from threadHelper import ImportantThread
from foxgloveRunner import FoxgloveRunner
from deviceRunner import DeviceRunner
import websockets
import threading
import asyncio
import logging
import json


class CommsRunner():
  def __init__(self, dr: DeviceRunner, fr: FoxgloveRunner):
    self.shouldStop_ = asyncio.Event()
    self.dr_ = dr
    self.fr_ = fr

  async def messageHandler_(self, websocket, path):
    async for message in websocket:
      try:
        print(f"Got json command {message}")
        command = json.loads(message)
        if command["cmd"] == "setIRBrightness":
          self.dr_.setIR(int(command["value"]))
        elif command["cmd"] == "setActiveNN":
          self.dr_.setActiveNN(command["value"])
        elif command["cmd"] == "enableIMU":
          self.dr_.enableIMU(command["value"])
          if command["value"] == True:
            self.fr_.add_channel("imu")
          else:
            self.fr_.remove_channel("imu")
        elif command["cmd"] == "enableSync":
          self.dr_.enableSync(command["value"])
        elif command["cmd"] == "enablePointcloud":
          self.dr_.enablePointcloud(command["value"])
          if command["value"] == True:
            self.fr_.add_channel("pointcloud")
          else:
            self.fr_.remove_channel("pointcloud")
        elif command["cmd"] == "enableLR":
          self.dr_.enableLR(command["value"])
        elif command["cmd"] == "restart":
          #self.dr_.restart()
          pass
      except Exception as e:
        logging.error(f"error : {e}")

  async def _run(self):
    async with websockets.serve(self.messageHandler_, "localhost", 8766):
      while not self.shouldStop_.is_set():
        await asyncio.sleep(0.1)
    return not self.shouldStop_.is_set()

  @ImportantThread("Comms Runner")
  def run(self, loop):
    asyncio.set_event_loop(loop)
    retval = loop.run_until_complete(self._run())
    asyncio.sleep(2)

    return retval

  def stop(self):
    self.shouldStop_.set()
