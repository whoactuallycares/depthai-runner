from pathlib import Path
import depthai as dai
import open3d as o3d
import numpy as np
import threading
import cv2
import os

from projector_device import PointCloudVisualizer

def getPath(resolution):
  (width, heigth) = resolution
  path = Path("models", "out")
  path.mkdir(parents=True, exist_ok=True)
  name = f"pointcloud_{width}x{heigth}"

  return_path = str(path / (name + '.blob'))
  if os.path.exists(return_path):
    return return_path
  print(f"ERROR: Path '{return_path}' not found")


def create_xyz(width, height, camera_matrix):
  xs = np.linspace(0, width - 1, width, dtype=np.float32)
  ys = np.linspace(0, height - 1, height, dtype=np.float32)

  # generate grid by stacking coordinates
  base_grid = np.stack(np.meshgrid(xs, ys)) # WxHx2
  points_2d = base_grid.transpose(1, 2, 0) # 1xHxWx2

  # unpack coordinates
  u_coord: np.array = points_2d[..., 0]
  v_coord: np.array = points_2d[..., 1]

  # unpack intrinsics
  fx: np.array = camera_matrix[0, 0]
  fy: np.array = camera_matrix[1, 1]
  cx: np.array = camera_matrix[0, 2]
  cy: np.array = camera_matrix[1, 2]

  # projective
  x_coord: np.array = (u_coord - cx) / fx
  y_coord: np.array = (v_coord - cy) / fy

  xyz = np.stack([x_coord, y_coord], axis=-1)
  return np.pad(xyz, ((0,0),(0,0),(0,1)), "constant", constant_values=1.0)

COLOR = True # Stream & display color frames

def configureDepthPostProcessing(stereoDepthNode):
  """
  In-place post-processing configuration for a stereo depth node
  The best combo of filters is application specific. Hard to say there is a one size fits all.
  They also are not free. Even though they happen on device, you pay a penalty in fps.
  """
  stereoDepthNode.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

  # stereoDepthNode.initialConfig.setBilateralFilterSigma(16)
  config = stereoDepthNode.initialConfig.get()
  config.postProcessing.speckleFilter.enable = True
  config.postProcessing.speckleFilter.speckleRange = 60
  config.postProcessing.temporalFilter.enable = True

  config.postProcessing.spatialFilter.holeFillingRadius = 2
  config.postProcessing.spatialFilter.numIterations = 1
  config.postProcessing.thresholdFilter.minRange = 700  # mm
  config.postProcessing.thresholdFilter.maxRange = 4000  # mm
  # config.postProcessing.decimationFilter.decimationFactor = 1
  config.censusTransform.enableMeanMode = True
  config.costMatching.linearEquationParameters.alpha = 0
  config.costMatching.linearEquationParameters.beta = 2
  stereoDepthNode.initialConfig.set(config)
  stereoDepthNode.setLeftRightCheck(True)
  stereoDepthNode.setExtendedDisparity(False)
  stereoDepthNode.setSubpixel(True)
  stereoDepthNode.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

def get_resolution(width):
  if width==480: return dai.MonoCameraProperties.SensorResolution.THE_480_P
  elif width==720: return dai.MonoCameraProperties.SensorResolution.THE_720_P
  elif width==800: return dai.MonoCameraProperties.SensorResolution.THE_800_P
  else: return dai.MonoCameraProperties.SensorResolution.THE_400_P

class PointcloudNetwork():
  def __init__(self):
    self._running = True
    self._has = False
    self.resolution = (640,400)
    self.pcl_data = np.zeros(3)
    self.R_camera_to_world = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).astype(np.float64)
    pass

  def createNodes(self, pipeline: dai.Pipeline, camRgb: dai.node.ColorCamera, stereo: dai.node.StereoDepth = None, sync: bool = True) -> None:
    # Depth -> PointCloud
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(getPath(self.resolution))
    stereo.depth.link(nn.inputs["depth"])

    xyz_in = pipeline.createXLinkIn()
    xyz_in.setMaxDataSize(6144000)
    xyz_in.setStreamName("xyz_in")
    xyz_in.out.link(nn.inputs["xyz"])

    # Only send xyz data once, and always reuse the message
    nn.inputs["xyz"].setReusePreviousMessage(True)

    pointsOut = pipeline.createXLinkOut()
    pointsOut.setStreamName("pcl")
    nn.out.link(pointsOut.input)

  def frame(self) -> cv2.Mat:
    raise NotImplementedError()

  def processingThread(self):
    while self._running:
      pcl_data = np.array(self.queue.get().getFirstLayerFp16()).reshape(1, 3, self.resolution[1], self.resolution[0])
      pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0
      #self.pcl_converter.visualize_pcl(pcl_data, downsample=True)

      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(pcl_data)
      pcd.remove_non_finite_points()
      pcd = pcd.voxel_down_sample(voxel_size=0.03)
      pcd.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))
      self.pcl_data = np.asarray(pcd.points, dtype=np.float64)

  def start(self, device: dai.Device):
    # calibData = device.readCalibration()
    # M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT,
    #   dai.Size2f(self.resolution[0], self.resolution[1]),
    # )

    # # Creater xyz data and send it to the device - to the pointcloud generation model (NeuralNetwork node)
    # xyz = create_xyz(self.resolution[0], self.resolution[1], np.array(M_right).reshape(3,3))
    # matrix = np.array([xyz], dtype=np.float16).view(np.int8)
    # buff = dai.Buffer()
    # buff.setData(matrix)
    # device.getInputQueue("xyz_in").send(buff)

    # self.pcl_converter = PointCloudVisualizer()
    # self.queue = device.getOutputQueue("pcl", maxSize=8, blocking=False)
    # #if COLOR:
    #   #self.qRgb = device.getOutputQueue("color", maxSize=1, blocking=False)





    # # main stream loop
    # while True:
    #  self.draw(0)

    print("Opening device")
    self.pcl_data = np.zeros((3))
    # device.setLogLevel(dai.LogLevel.ERR)

    calibData = device.readCalibration()
    M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT,
      dai.Size2f(self.resolution[0], self.resolution[1]),
    )

    # Creater xyz data and send it to the device - to the pointcloud generation model (NeuralNetwork node)
    xyz = create_xyz(self.resolution[0], self.resolution[1], np.array(M_right).reshape(3,3))
    matrix = np.array([xyz], dtype=np.float16).view(np.int8)
    buff = dai.Buffer()
    buff.setData(matrix)
    device.getInputQueue("xyz_in").send(buff)

    #self.pcl_converter = PointCloudVisualizer()
    self.queue = device.getOutputQueue("pcl", maxSize=8, blocking=False)

    self.threads = [
      threading.Thread(target=self.processingThread)
    ]
    for thread in self.threads:
      thread.start()
    return
    # while True:
    #   pcl_data = np.array(self.queue.get().getFirstLayerFp16()).reshape(1, 3, self.resolution[1], self.resolution[0])
    #   pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(pcl_data)
    #   pcd.remove_non_finite_points()
    #   pcd = pcd.voxel_down_sample(voxel_size=0.03)

    #   cd.setPointcloud(np.asarray(pcd.points))
    #   print(f"YTT {pcl_data.shape}")
    if COLOR:
      qRgb = device.getOutputQueue("color", maxSize=1, blocking=False)
    qStereo = device.getOutputQueue("stereo", maxSize=8, blocking=False)

    resolution = self.resolution

    # main stream loop
    while True:
      print("Start")
      if qStereo.has():
        qStereo.get()
      if COLOR and qRgb.has():
        cv2.imshow("color", qRgb.get().getCvFrame())


      print("GotColor")
      pcl_data = np.array(queue.get().getFirstLayerFp16()).reshape(1, 3, resolution[1], resolution[0])
      pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0
      pcl_converter.visualize_pcl(pcl_data, downsample=True)
      print("Visualized")

      if cv2.waitKey(1) == "q":
        pcl_converter.close_window()
        break
