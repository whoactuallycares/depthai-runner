from foxgloveComms import FoxgloveUploader
from cameraData import CameraData
import depthai as dai
import websockets
import threading
import asyncio
import time
import json
import cv2

ENABLE_CV2 = False
ENABLE_FOXGLOVE = True

def createPipeline():
  pipeline = dai.Pipeline()

  camRgb = pipeline.createColorCamera()
  camRgb.setIspScale(1, 3)

  colorOut = pipeline.createXLinkOut()
  colorOut.setStreamName("color")
  camRgb.isp.link(colorOut.input)

  # Configure Camera Properties
  left = pipeline.createMonoCamera()
  left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
  left.setBoardSocket(dai.CameraBoardSocket.LEFT)

  # left camera output
  leftOut = pipeline.createXLinkOut()
  leftOut.setStreamName("left")
  left.out.link(leftOut.input)

  right = pipeline.createMonoCamera()
  right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
  right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

  # right camera output
  rightOut = pipeline.createXLinkOut()
  rightOut.setStreamName("right")
  right.out.link(rightOut.input)

  stereo = pipeline.createStereoDepth()
  # configureDepthPostProcessing(stereo)
  left.out.link(stereo.left)
  right.out.link(stereo.right)

  stereoOut = pipeline.createXLinkOut()
  stereoOut.setStreamName("stereo")
  stereo.disparity.link(stereoOut.input)

  imu = pipeline.create(dai.node.IMU)
  imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
  imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
  imu.setBatchReportThreshold(1)
  imu.setMaxBatchReports(10)

  xlinkOut = pipeline.create(dai.node.XLinkOut)
  xlinkOut.setStreamName("imu")
  imu.out.link(xlinkOut.input)

  return pipeline

irValue = 0
irValueLast = 0

async def messageHandler(websocket, path):
  async for message in websocket:
    try:
      command = json.dumps(message)
      print(f"Got json command {command}")
      if message[:6] == "Slider":
        global irValue
        irValue = int(message[8:])
        print(f"IR value now {irValue}")
    except:
      print(f"Received invalid json: {message}")

async def startServer():
  server = await websockets.serve(messageHandler, "localhost", 8766)
  await server.wait_closed()

def main(cameraData: CameraData) -> None:
  with dai.Device(createPipeline()) as device:
    qColor = device.getOutputQueue("color", maxSize=1, blocking=False)
    qLeft = device.getOutputQueue("left", maxSize=1, blocking=False)
    qRight = device.getOutputQueue("right", maxSize=1, blocking=False)
    qStereo = device.getOutputQueue("stereo", maxSize=1, blocking=False)
    qIMU = device.getOutputQueue("imu", maxSize=1, blocking=False)

    while not shouldStop.is_set():
      if qColor.has():
        cameraData.setColorFrame(qColor.get().getCvFrame())
        if ENABLE_CV2:
          cv2.imshow("color", qColor.get().getCvFrame())
      if qLeft.has():
        cameraData.setLeftFrame(qLeft.get().getCvFrame())
        if ENABLE_CV2:
          cv2.imshow("left", qLeft.get().getCvFrame())
      if qRight.has():
        cameraData.setRightFrame(qRight.get().getCvFrame())
        if ENABLE_CV2:
          cv2.imshow("right", qRight.get().getCvFrame())
      if qStereo.has():
        cameraData.setStereoFrame(qStereo.get().getCvFrame())
        if ENABLE_CV2:
          cv2.imshow("stereo", qStereo.get().getCvFrame())
      if qIMU.has():
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
      global irValue, irValueLast
      if irValue != irValueLast:
        print(f"Setting IR value to {irValue*2}")
        irValueLast = irValue
        device.setIrLaserDotProjectorBrightness(2*irValue)

      # Check for commands

      if cv2.waitKey(1) == "q":
        break



# Start the server in a separate thread

if __name__ == "__main__":

  cd = CameraData()
  shouldStop = threading.Event()
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
