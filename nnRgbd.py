import depthai as dai
import open3d as o3d
import numpy as np
import threading
import cv2

class HostSync:
  def __init__(self, numMessages: int = 2):
    self.arrays = {}
    self.numMessages = numMessages

  def add_msg(self, name, msg):
    if not name in self.arrays:
      self.arrays[name] = []
    # Add msg to array
    self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

    synced = {}
    for name, arr in self.arrays.items():
      for i, obj in enumerate(arr):
        if msg.getSequenceNum() == obj["seq"]:
          synced[name] = obj["msg"]
          break
    # If there are 5 (all) synced msgs, remove all old msgs
    # and return synced msgs
    if len(synced) == self.numMessages:  # pc_color, pc_stereo
      # Remove old msgs
      for name, arr in self.arrays.items():
        for i, obj in enumerate(arr):
          if obj["seq"] < msg.getSequenceNum():
            arr.remove(obj)
          else:
            break
      return synced
    return False

class RGBDNetwork():
  def __init__(self):
    self._running = True
    self._has = False
    self.pc_data = None
    pass

  def createNodes(self, pipeline: dai.Pipeline, camRgb: dai.node.ColorCamera, stereo: dai.node.StereoDepth = None, sync: bool = True) -> None:
    self.w, self.h = camRgb.getIspSize()

    colorManip = pipeline.createImageManip()
    camRgb.isp.link(colorManip.inputImage)

    colorOut = pipeline.createXLinkOut()
    colorOut.setStreamName("pc_color")
    colorManip.out.link(colorOut.input)

    stereoManip = pipeline.createImageManip()
    stereo.depth.link(stereoManip.inputImage)

    stereoOut = pipeline.createXLinkOut()
    stereoOut.setStreamName("pc_stereo")
    stereoManip.out.link(stereoOut.input)

  def frame(self) -> cv2.Mat:
    raise NotImplementedError()

  def draw(self, frame: cv2.Mat) -> None:
    raise NotImplementedError()

  def processingThread(self):
    qs = []
    qs.append(self.device.getOutputQueue("pc_stereo", maxSize=1, blocking=False))
    qs.append(self.device.getOutputQueue("pc_color", maxSize=1, blocking=False))

    calibData = self.device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(self.w, self.h))

    self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
      self.w,
      self.h,
      intrinsics[0][0],
      intrinsics[1][1],
      intrinsics[0][2],
      intrinsics[1][2])

    self.R_camera_to_world = np.array([
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0]]).astype(np.float64)
    self.pcl = o3d.geometry.PointCloud()

    sync = HostSync()
    depth_vis, color = None, None

    while self._running:
      for q in qs:
        new_msg = q.tryGet()
        if new_msg is not None:
          msgs = sync.add_msg(q.getName(), new_msg)
          if msgs:
            depth = msgs["pc_stereo"].getFrame()
            color = msgs["pc_color"].getCvFrame()
            depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_vis = cv2.equalizeHist(depth_vis)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), o3d.geometry.Image(depth), convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=20000, depth_scale=1000.0
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)

            # downsample
            pcd = pcd.voxel_down_sample(voxel_size=0.01)

            # remove_noise:
            #pcd = pcd.remove_statistical_outlier(30, 0.1)[0]

            pcd.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))

            arrPoints = np.asarray(pcd.points, dtype=np.float64).astype(np.float32).reshape((-1,3)).transpose()
            arrColors = np.asarray(pcd.colors, dtype=np.float64).astype(np.float32).reshape((-1,3)).transpose()
            alpha = np.ones((arrColors.shape[1]), dtype=np.float32)
            done = np.vstack((arrPoints, arrColors, alpha))
            self.pc_data = done.transpose()

      key = cv2.waitKey(1)
      if key == ord("q"):
        break

  def start(self, device: dai.Device):
    self.device = device

    self.threads = [
        threading.Thread(target=self.processingThread),
    ]

    for thread in self.threads:
        thread.start()

  def stop(self) -> None:
    self._running = False

  def has(self) -> bool:
    return self._has
