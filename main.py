from nnGazeEstimation import GazeEstimationNetwork
from nnPointcloud import PointcloudNetwork
from foxgloveComms import FoxgloveUploader
from nnHumanPose import HumanPoseNetwork
from cameraData import CameraData
from nnYolo import YoloNetwork
import depthai as dai
import numpy as np
import websockets
import threading
import asyncio
import time
import json
import cv2

ENABLE_CV2 = False
ENABLE_FOXGLOVE = True
ENABLE_IMU = False
ENABLE_POINTCLOUD = True
ENABLE_LR = False
ENABLE_SYNC = False

allNetworks = [
  HumanPoseNetwork(),
  YoloNetwork(),
  GazeEstimationNetwork(),
]

networkMap = {
  "pose" : HumanPoseNetwork(),
  "yolo" : YoloNetwork(),
  "gaze" : GazeEstimationNetwork(),
}

activeNetwork = HumanPoseNetwork()
pointcloud = PointcloudNetwork(ENABLE_CV2)

def configureDepthPostProcessing(stereoDepthNode: dai.node.StereoDepth, lrcheck: bool = True, extended: bool = False, subpixel: bool = True) -> None:
  """
  In-place post-processing configuration for a stereo depth node
  The best combo of filters is application specific. Hard to say there is a one size fits all.
  They also are not free. Even though they happen on device, you pay a penalty in fps.
  """
  stereoDepthNode.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

  #stereoDepthNode.initialConfig.setBilateralFilterSigma(16)
  config = stereoDepthNode.initialConfig.get()
  config.postProcessing.speckleFilter.enable = True
  config.postProcessing.speckleFilter.speckleRange = 60
  config.postProcessing.temporalFilter.enable = True

  config.postProcessing.spatialFilter.holeFillingRadius = 2
  config.postProcessing.spatialFilter.numIterations = 1
  config.postProcessing.thresholdFilter.minRange = 700  # mm
  config.postProcessing.thresholdFilter.maxRange = 4000  # mm
  #config.postProcessing.decimationFilter.decimationFactor = 1
  config.censusTransform.enableMeanMode = True
  config.costMatching.linearEquationParameters.alpha = 0
  config.costMatching.linearEquationParameters.beta = 2
  stereoDepthNode.initialConfig.set(config)
  stereoDepthNode.setLeftRightCheck(lrcheck)
  stereoDepthNode.setExtendedDisparity(extended)
  #stereoDepthNode.setSubpixel(subpixel)
  stereoDepthNode.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

def createPipeline():
  pipeline = dai.Pipeline()

  camRgb = pipeline.createColorCamera()
  camRgb.setIspScale(1, 3)
  #camRgb.setFps(40)
  #camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

  colorOut = pipeline.createXLinkOut()
  colorOut.setStreamName("color")
  camRgb.isp.link(colorOut.input)

  # Configure Camera Properties
  left = pipeline.createMonoCamera()
  left.setBoardSocket(dai.CameraBoardSocket.LEFT)
  left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

  # left camera output
  if ENABLE_LR:
    leftOut = pipeline.createXLinkOut()
    leftOut.setStreamName("left")
    left.out.link(leftOut.input)

  right = pipeline.createMonoCamera()
  right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
  right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

  # right camera output
  if ENABLE_LR:
    rightOut = pipeline.createXLinkOut()
    rightOut.setStreamName("right")
    right.out.link(rightOut.input)

  stereo = pipeline.createStereoDepth()
  configureDepthPostProcessing(stereo)
  #stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
  #stereo.setLeftRightCheck(True)

  left.out.link(stereo.left)
  right.out.link(stereo.right)

  stereoOut = pipeline.createXLinkOut()
  stereoOut.setStreamName("stereo")
  stereo.depth.link(stereoOut.input)

  if ENABLE_IMU:
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    xlinkOut = pipeline.create(dai.node.XLinkOut)
    xlinkOut.setStreamName("imu")
    imu.out.link(xlinkOut.input)

  activeNetwork.createNodes(pipeline, camRgb, stereo=stereo, sync=ENABLE_SYNC)

  if ENABLE_POINTCLOUD:
    pointcloud.createNodes(pipeline, camRgb, stereo=stereo, sync=False)

  return pipeline

irValue = 0
irValueLast = 0
selectedNN = 0
selectedNNLast = 0

async def messageHandler(websocket, path):
  global activeNetwork
  async for message in websocket:
    try:
      print(f"Got json command {message}")
      if message[:6] == "option":
        print("selecting option")
        global selectedNN
        selectedNN = int(message[-1]) - 1
      if message[:6] == "Slider":
        global irValue
        irValue = int(message[8:])
        print(f"IR value now {irValue}")
      command = json.dumps(message)


      if command["command"] == None:
        continue
      elif command["command"] == "set-nn":
        activeNetwork = networkMap[command["args"]]
      elif command["command"] == "set-ir":
        irValue = int(command["args"])

    except:
      print(f"Received invalid json: {message}")

async def startServer():
  server = await websockets.serve(messageHandler, "localhost", 8766)
  while not shouldStop.is_set():
    time.sleep(0.5)

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

def main(cameraData: CameraData) -> None:
  global activeNetwork
  while not shouldStop.is_set(): # To allow for pipeline recreation
    with dai.Device(createPipeline()) as device:
      device.setIrLaserDotProjectorBrightness(800)
      if ENABLE_POINTCLOUD:
        pointcloud.start(device)

      activeNetwork.start(device)
      qColor = device.getOutputQueue("color", maxSize=1, blocking=False)
      if ENABLE_LR:
        qLeft = device.getOutputQueue("left", maxSize=1, blocking=False)
        qRight = device.getOutputQueue("right", maxSize=1, blocking=False)
      qStereo = device.getOutputQueue("stereo", maxSize=1, blocking=False)
      if ENABLE_IMU:
        qIMU = device.getOutputQueue("imu", maxSize=1, blocking=False)

      colorFrame = None
      hasColor = False

      while not shouldStop.is_set() or shouldRestart.is_set():
        if ENABLE_POINTCLOUD:
          cameraData.setPointcloud(pointcloud.pcl_data)
        if qColor.has():
          colorFrame = qColor.get().getCvFrame()
          cameraData.setColorFrame(colorFrame)
          hasColor = True
          #activeNetwork.draw(colorFrame)
          if ENABLE_CV2:
            cv2.imshow("color", colorFrame)
        if hasColor:
          w = colorFrame.shape[1]
          h = colorFrame.shape[0]
          nnFrame = colorFrame[0: h, (w-h)//2: w-(w-h)//2, 0:3]
          activeNetwork.draw(nnFrame)
          #cameraData.setNnFrame(nnFrame)
          if ENABLE_CV2:
            cv2.imshow("nn", nnFrame)
        if ENABLE_LR:
          if qLeft.has():
           leftFrame = qLeft.get().getCvFrame()
           cameraData.setLeftFrame(leftFrame)
           if ENABLE_CV2:
             cv2.imshow("left", leftFrame)
          if qRight.has():
           rightFrame = qRight.get().getCvFrame()
           cameraData.setRightFrame(rightFrame)
           if ENABLE_CV2:
             cv2.imshow("right", rightFrame)
        if qStereo.has():
          stereoFrame = qStereo.get().getCvFrame() * 10
          print(stereoFrame.shape)
          print(stereoFrame.dtype)
          in8bit = map_uint16_to_uint8(stereoFrame, 0, 2**16 - 1)

          cameraData.setStereoFrame(cv2.applyColorMap(in8bit, cv2.COLORMAP_JET))
          cv2.imshow("stereo", cv2.applyColorMap(in8bit, cv2.COLORMAP_JET))
          if ENABLE_CV2:
            cv2.imshow("stereo", stereoFrame)
        if ENABLE_IMU and qIMU.has():
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


        global irValue, irValueLast, selecteNN, selectedNNLast
        if irValue != irValueLast:
          print(f"Setting IR value to {irValue*2}")
          irValueLast = irValue
          device.setIrLaserDotProjectorBrightness(2*irValue)

        # Check for commands

        if cv2.waitKey(1) == "q":
          break
    shouldRestart.clear()


# Start the server in a separate thread

if __name__ == "__main__":

  cd = CameraData()
  shouldStop = threading.Event()
  shouldRestart = threading.Event()
  receiverThread = threading.Thread(target=asyncio.run, args=(startServer(),))
  receiverThread.start()

  senderThread = threading.Thread(target=main, args=(cd,))
  senderThread.start()

  if ENABLE_FOXGLOVE:
    fgUp = FoxgloveUploader()
    fgUp.run(cd)
  else:
    while True:
      try:
        time.sleep(1)
      except KeyboardInterrupt:
        break
  shouldStop.set()
  senderThread.join()
  receiverThread.join()
