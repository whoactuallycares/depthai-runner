from nnGazeEstimation import GazeEstimationNetwork
from nnHumanPose import HumanPoseNetwork
from threadHelper import ImportantThread
from cameraData import CameraData
from nnRgbd import RGBDNetwork
from nnYolo import YoloNetwork
import depthai as dai
import numpy as np
import threading
import cv2

def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
  '''
  Map a 16-bit image trough a lookup table to convert it to 8-bit.

  Parameters
  ----------
  img: numpy.ndarray[np.uint16]
      image that should be mapped
  lower_bound: int, optional
      lower bound of the range that should be mapped to ``[0, 255]``,
      value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
      (defaults to ``numpy.min(img)``)
  upper_bound: int, optional
      upper bound of the range that should be mapped to ``[0, 255]``,
      value must be in the range ``[0, 65535]`` and larger than `lower_bound`
      (defaults to ``numpy.max(img)``)

  Returns
  -------
  numpy.ndarray[uint8]
  '''
  if not(0 <= lower_bound < 2**16) and lower_bound is not None:
    raise ValueError(
      '"lower_bound" must be in the range [0, 65535]')
  if not(0 <= upper_bound < 2**16) and upper_bound is not None:
    raise ValueError(
      '"upper_bound" must be in the range [0, 65535]')
  if lower_bound is None:
    lower_bound = np.min(img)
  if upper_bound is None:
    upper_bound = np.max(img)
  if lower_bound >= upper_bound:
    raise ValueError(
      '"lower_bound" must be smaller than "upper_bound"')
  lut = np.concatenate([
    np.zeros(lower_bound, dtype=np.uint16),
    np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
    np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
  ])
  return lut[img].astype(np.uint8)

class DeviceRunner():
  def __init__(self):
    self.shouldStop_ = threading.Event()
    self.shouldRestart_ = threading.Event()
    self.enableCV2_ = False
    self.enableFoxglove_ = True
    self.enableIMU_ = False
    self.enablePointcloud_ = False
    self.enableLR_ = False
    self.enableSync_ = False
    self.activeNetwork_ = "gaze"
    # Real values currently on the camera
    self.has_ = {
      "IMU": self.enableIMU_,
      "Pointcloud": self.enablePointcloud_,
      "LR": self.enableLR_,
      "Sync": self.enableSync_,
      "NN": self.activeNetwork_,
    }
    self.cmdLock_ = threading.Lock()
    self.commands_ = []
    self.irValue_ = 1000
    self.oldIrValue_ = 1000
    self.device_ = None
    self.networks_ = {
      "pose" : HumanPoseNetwork(),
      "yolo" : YoloNetwork(),
      "gaze" : GazeEstimationNetwork(),
    }
    self.pointcloud_ = RGBDNetwork()

  def restart(self):
    self.shouldRestart_.set()

  def stop(self):
    self.shouldStop_.set()

  def setIR(self, value: int):
    self.irValue_ = value

  def enableCV2(self, value: bool):
    self.enableCV2_ = value

  def enableFoxglove(self, value: bool):
    self.enableFoxglove_ = value

  def enableIMU(self, value: bool):
    self.enableIMU_ = value

  def enablePointcloud(self, value: bool):
    self.enablePointcloud_ = value

  def enableLR(self, value: bool):
    self.enableLR_ = value

  def enableSync(self, value: bool):
    self.enableSync_ = value

  def setActiveNN(self, value: str):
    self.activeNetwork_ = value

  def createPipeline(self):
    pipeline = dai.Pipeline()

    # Color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    camRgb.setIspScale(1, 2)
    camRgb.initialControl.setManualFocus(130)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    colorOut = pipeline.createXLinkOut()
    colorOut.setStreamName("color")
    camRgb.isp.link(colorOut.input)

    # Left camera output
    left = pipeline.createMonoCamera()
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    if self.enableLR_:
      leftOut = pipeline.createXLinkOut()
      leftOut.setStreamName("left")
      left.out.link(leftOut.input)

    # Right camera output
    right = pipeline.createMonoCamera()
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    if self.enableLR_:
      rightOut = pipeline.createXLinkOut()
      rightOut.setStreamName("right")
      right.out.link(rightOut.input)

    # Stereo
    stereo = pipeline.createStereoDepth()
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    stereoOut = pipeline.createXLinkOut()
    stereoOut.setStreamName("stereo")
    stereo.depth.link(stereoOut.input)

    # IMU
    if self.enableIMU_:
      imu = pipeline.create(dai.node.IMU)
      imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
      imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
      imu.setBatchReportThreshold(1)
      imu.setMaxBatchReports(10)

      xlinkOut = pipeline.create(dai.node.XLinkOut)
      xlinkOut.setStreamName("imu")
      imu.out.link(xlinkOut.input)

    # NN
    self.networks_[self.has_["NN"]].createNodes(pipeline, camRgb, stereo=stereo, sync=self.enableSync_)

    # Pointcloud
    if self.enablePointcloud_:
      self.pointcloud_.createNodes(pipeline, camRgb, stereo=stereo, sync=False)

    return pipeline

  @ImportantThread("Device Runner")
  def run(self, cameraData: CameraData):
    while not self.shouldStop_.is_set(): # To allow for pipeline recreation
      with dai.Device(self.createPipeline()) as device:
        self.device_ = device

        if self.enablePointcloud_:
          self.pointcloud_.start(device)

        self.networks_[self.has_["NN"]].start(device)
        qColor = device.getOutputQueue("color", maxSize=1, blocking=False)
        qStereo = device.getOutputQueue("stereo", maxSize=1, blocking=False)
        if self.enableLR_:
          qLeft = device.getOutputQueue("left", maxSize=1, blocking=False)
          qRight = device.getOutputQueue("right", maxSize=1, blocking=False)
        if self.enableIMU_:
          qIMU = device.getOutputQueue("imu", maxSize=1, blocking=False)

        colorFrame = None
        hasColor = False

        while not self.shouldStop_.is_set() and not self.shouldRestart_.is_set():
          if self.oldIrValue_ != self.irValue_:
            self.device_.setIrLaserDotProjectorBrightness(self.irValue_)
            self.oldIrValue_ = self.irValue_
          if self.has_["Pointcloud"]:
            cameraData.setPointcloud(self.pointcloud_.pc_data)
            pass
          if qColor.has():
            colorFrame = qColor.get().getCvFrame()
            cameraData.setColorFrame(colorFrame)
            hasColor = True
            #networks[activeNetwork].draw(colorFrame)
            if self.enableCV2_:
              cv2.imshow("color", colorFrame)
          if hasColor:
            w = colorFrame.shape[1]
            h = colorFrame.shape[0]
            nnFrame = colorFrame[0: h, (w-h)//2: w-(w-h)//2, 0:3]
            self.networks_[self.has_["NN"]].draw(nnFrame)
            cameraData.setNnFrame(nnFrame)
            if self.enableCV2_:
              cv2.imshow("nn", nnFrame)
          if self.has_["LR"]:
            if qLeft.has():
              leftFrame = qLeft.get().getCvFrame()
              cameraData.setLeftFrame(leftFrame)
              if self.enableCV2_:
                cv2.imshow("left", leftFrame)
            if qRight.has():
              rightFrame = qRight.get().getCvFrame()
              cameraData.setRightFrame(rightFrame)
              if self.enableCV2_:
                cv2.imshow("right", rightFrame)
          if qStereo.has():
            stereoFrame = qStereo.get().getCvFrame() * 10
            in8bit = map_uint16_to_uint8(stereoFrame, 0, 2**16 - 1)
            lut = np.arange(256, dtype=np.uint8)[::-1]
            pretty8bit = cv2.applyColorMap(cv2.LUT(in8bit, lut), cv2.COLORMAP_PLASMA)
            cameraData.setStereoFrame(pretty8bit)
            if self.enableCV2_:
              cv2.imshow("stereo", pretty8bit)
          if self.has_["IMU"] and qIMU.has():
            for packet in qIMU.get().packets:
              imuData = {
                "timestamp": 0,
                "gyroscope": {
                  "x": packet.gyroscope.x,
                  "y": packet.gyroscope.y,
                  "z": packet.gyroscope.z,
                },
                "accelerometer": {
                  "x": packet.acceleroMeter.x,
                  "y": packet.acceleroMeter.y,
                  "z": packet.acceleroMeter.z,
                }
              }
              cameraData.setIMU(imuData)

          if cv2.waitKey(1) == "q":
            break
      self.device_ = None
      self.shouldRestart_.clear()

      # Apply settings
      self.has_["NN"] = self.activeNetwork_
      self.has_["Pointcloud"] = self.enablePointcloud_
      self.has_["IMU"] = self.enableIMU_
      self.has_["LR"] = self.enableLR_
      self.has_["Sync"] = self.enableSync_
    return not self.shouldStop_.is_set()
