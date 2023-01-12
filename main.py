from foxgloveComms import FoxgloveUploader
from deviceRunner import DeviceRunner
from cameraData import CameraData
import websockets
import threading
import asyncio
import time
import json

ENABLE_FOXGLOVE = True

dr = DeviceRunner()
async def messageHandler(websocket, path):
  global dr
  async for message in websocket:
    try:
      print(f"Got json command {message}")
      command = json.loads(message)
      if command["cmd"] == "setIRBrightness":
        dr.setIR(int(command["value"]))
      elif command["cmd"] == "setActiveNN":
        dr.setActiveNN(command["value"])
      elif command["cmd"] == "enableIMU":
        dr.enableIMU(command["value"])
      elif command["cmd"] == "enableSync":
        dr.enableSync(command["value"])
      elif command["cmd"] == "enablePointcloud":
        dr.enablePointcloud(command["value"])
      elif command["cmd"] == "enableLR":
        dr.enableLR(command["value"])
      elif command["cmd"] == "restart":
        dr.restart()
    except:
      print(f"Received invalid json: {message}")

async def startServer():
  async with websockets.serve(messageHandler, "localhost", 8766):
    while not shouldStop.is_set():
      await asyncio.sleep(0.1)

# Start the server in a separate thread
if __name__ == "__main__":

  cd = CameraData()
  shouldStop = threading.Event()
  shouldRestart = threading.Event()
  receiverThread = threading.Thread(target=asyncio.run, args=(startServer(),))
  receiverThread.start()

  drThread = dr.run(cd)

  if ENABLE_FOXGLOVE:
    fgUp = FoxgloveUploader()
    fgUp.run(cd)
    time.sleep(1)
    #fgUp.add_video_channel("color")
    #fgUp.add_video_channel("stereo")
    #fgUp.add_video_channel("nn")
    #fgUp.add_imu_channel()
    #fgUp.add_pointcloud_channel()
  else:
    while True:
      try:
        time.sleep(1)
      except KeyboardInterrupt:
        break
  dr.stop()
  shouldStop.set()
  receiverThread.join()

  drThread.set()
