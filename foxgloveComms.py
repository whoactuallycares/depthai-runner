from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_websocket import run_cancellable
from foxglove_websocket.types import ChannelId
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf.descriptor import FileDescriptor
import google.protobuf.message
from base64 import b64encode
from typing import Set, Type
import logging
import asyncio
import time

from cameraData import CameraData

logging.basicConfig(level=logging.INFO)



def build_file_descriptor_set(
  message_class: Type[google.protobuf.message.Message],
) -> FileDescriptorSet:
  """
  Build a FileDescriptorSet representing the message class and its dependencies.
  """
  file_descriptor_set = FileDescriptorSet()
  seen_dependencies: Set[str] = set()

  def append_file_descriptor(file_descriptor: FileDescriptor):
    for dep in file_descriptor.dependencies:
      if dep.name not in seen_dependencies:
        seen_dependencies.add(dep.name)
        append_file_descriptor(dep)
    file_descriptor.CopyToProto(file_descriptor_set.file.add())  # type: ignore

  append_file_descriptor(message_class.DESCRIPTOR.file)
  return file_descriptor_set



class FoxgloveUploader():
  def __init__(self, name: str = "Camera Streamer", address: str = "0.0.0.0", port: int = 8765):
    self.name = name
    self.address = address
    self.port = port

  class Listener(FoxgloveServerListener):
    def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
      logging.debug("First client subscribed to", channel_id)

    def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
      logging.debug("Last client unsubscribed from", channel_id)

  async def _run(self, camData: CameraData):
    async with FoxgloveServer(self.address, self.port, self.name) as server:
      server.set_listener(self.Listener())
      videoStreams = ["color", "nn", "left", "right", "stereo"]
      channels = {}

      for videoStream in videoStreams:
        channels[videoStream] = await server.add_channel(
          {
            "topic": f"raw_{videoStream}",
            "encoding": "protobuf",
            "schemaName": CompressedImage.DESCRIPTOR.full_name,
            "schema": b64encode(
              build_file_descriptor_set(CompressedImage).SerializeToString()
            ).decode("ascii"),
          }
        )

      while True:
        await asyncio.sleep(1 / 60)

        for videoStream in videoStreams:
          raw_image = CompressedImage()
          raw_image.frame_id = f"raw_{videoStream}"
          raw_image.format = "jpeg"

          if videoStream == "color":
            frame = camData.getColorFrame()
          elif videoStream == "nn":
            frame = camData.getNnFrame()
          elif videoStream == "left":
            frame = camData.getLeftFrame()
          elif videoStream == "right":
            frame = camData.getRightFrame()
          elif videoStream == "stereo":
            frame = camData.getStereoFrame()

          raw_image.data = bytes(frame)
          raw_image.timestamp.FromNanoseconds(time.time_ns())

          await server.send_message(channels[videoStream], time.time_ns(), raw_image.SerializeToString())

  def run(self, camData: CameraData):
    run_cancellable(self._run(camData))