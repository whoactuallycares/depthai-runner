from foxgloveComms import FoxgloveUploader
from foxgloveRunner import FoxgloveRunner
from deviceRunner import DeviceRunner
from cameraData import CameraData
import websockets
import threading
import asyncio
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
ENABLE_FOXGLOVE = True

dr = DeviceRunner()
fr = None


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
        if command["value"] == True:
          fr.add_channel("imu")
          print("Add")
        else:
          fr.remove_channel("imu")
          print("Rem")
      elif command["cmd"] == "enableSync":
        dr.enableSync(command["value"])
      elif command["cmd"] == "enablePointcloud":
        dr.enablePointcloud(command["value"])
        if command["value"] == True:
          fr.add_channel("pointcloud")
        else:
          fr.remove_channel("pointcloud")
      elif command["cmd"] == "enableLR":
        dr.enableLR(command["value"])
      elif command["cmd"] == "restart":
        #dr.restart()
        pass
    except Exception as e:
      logging.error(f"error : {e}")

async def startServer():
  async with websockets.serve(messageHandler, "localhost", 8766):
    while not shouldStop.is_set():
      await asyncio.sleep(0.1)

# Start the server in a separate thread
if __name__ == "__main__":

  cd = CameraData()
  fr = FoxgloveRunner(cd)
  shouldStop = threading.Event()
  shouldRestart = threading.Event()
  receiverThread = threading.Thread(target=asyncio.run, args=(startServer(),))
  receiverThread.start()

  drThread = dr.run(cd)

  frThread = fr.run(asyncio.new_event_loop())
  time.sleep(3)
  fr.add_channel("color")
  fr.add_channel("nn")
  fr.add_channel("stereo")


  if ENABLE_FOXGLOVE and False:
    #fgUp = FoxgloveUploader()
    #fgUp.run(cd)
    time.sleep(1)
  else:
    while True:
      try:
        time.sleep(1)
      except KeyboardInterrupt:
        break
  dr.stop()
  fr.stop()
  shouldStop.set()
  receiverThread.join()

  drThread.set()
  frThread.set()
