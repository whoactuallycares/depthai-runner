from foxgloveRunner import FoxgloveRunner
from deviceRunner import DeviceRunner
from commsRunner import CommsRunner
from cameraData import CameraData
import asyncio
import logging
import time

logging.basicConfig(level=logging.INFO)
ENABLE_FOXGLOVE = True

dr = DeviceRunner()
fr = None
cr = None

# Start the server in a separate thread
if __name__ == "__main__":
  cd = CameraData()
  fr = FoxgloveRunner(cd)
  cr= CommsRunner(dr, fr)

  drThread = dr.run(cd)
  frThread = fr.run(asyncio.new_event_loop())

  time.sleep(3)
  fr.add_channel("color")
  fr.add_channel("nn")
  fr.add_channel("stereo")

  crThread =cr.run(asyncio.new_event_loop())

  while True:
    try:
      time.sleep(1)
    except KeyboardInterrupt:
      break
  dr.stop()
  fr.stop()
  cr.stop()

  drThread.set()
  frThread.set()
  crThread.set()
