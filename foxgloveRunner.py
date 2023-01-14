from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from google.protobuf.descriptor import FileDescriptor
from foxglove_websocket import run_cancellable
from foxglove_websocket.types import ChannelId
from threadHelper import ImportantThread
from cameraData import CameraData
import google.protobuf.message
from base64 import b64encode
from typing import Set, Type
import logging
import asyncio
import json
import time

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

class FoxgloveRunner():
  def __init__(self, cameraData: CameraData):
    self.shouldStop_ = asyncio.Event()
    self.cameraData_ = cameraData
    self.channels_ = []
    pass

  class Listener(FoxgloveServerListener):
    def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
      logging.debug("Client subscribed to", channel_id)

    def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
      logging.debug("Client unsubscribed from", channel_id)

  def add_channel(self, name: str):
    self.channels_ += [{"id": None, "name": name}]

  def remove_channel(self, name: str):
    channel = next(filter(lambda chan: chan["name"] == name, self.channels_))
    channel["name"] = None # Clear name to signal that it's supposed to be removed by the main thread

  async def send_pointcloud(self, camData, channel):
    pcData = camData.getPointcloud()
    if pcData is None:
      return
    pointcloud = PointCloud()
    pointcloud.timestamp.FromNanoseconds(time.time_ns())
    pointcloud.frame_id = "test"
    pointcloud.data = pcData.tobytes()
    pointcloud.point_stride = 4 * 7
    pointcloud.fields.append(PackedElementField(name="x", offset=0, type=7)) # FLOAT32
    pointcloud.fields.append(PackedElementField(name="y", offset=4, type=7)) # FLOAT32
    pointcloud.fields.append(PackedElementField(name="z", offset=8, type=7)) # FLOAT32
    pointcloud.fields.append(PackedElementField(name="red", offset=12, type=7)) # FLOAT32
    pointcloud.fields.append(PackedElementField(name="green", offset=16, type=7)) # FLOAT32
    pointcloud.fields.append(PackedElementField(name="blue", offset=20, type=7)) # FLOAT32
    pointcloud.fields.append(PackedElementField(name="alpha", offset=24, type=7)) # FLOAT32
    await self.server.send_message(channel["id"], time.time_ns(), pointcloud.SerializeToString())

  def _channel_description(self, name: str):
    if name == "pointcloud":
      return {
        "topic": "pointcloud",
        "encoding": "protobuf",
        "schemaName": PointCloud.DESCRIPTOR.full_name,
        "schema": b64encode(
          build_file_descriptor_set(PointCloud).SerializeToString()
        ).decode("ascii"),
      }
    elif name == "imu":
      return {
        "topic": "imu",
        "encoding": "json",
        "schemaName": "com.luxonis.imu",
        "schema": json.dumps(
          {
            "type": "object",
            "properties": {
              "timestamp": {"type": "integer"},
              "gyroscope": {
                "type": "object",
                "properties": {
                  "x": {"type": "number"},
                  "y": {"type": "number"},
                  "z": {"type": "number"},
                },
              },
              "accelerometer": {
                "type": "object",
                "properties": {
                  "x": {"type": "number"},
                  "y": {"type": "number"},
                  "z": {"type": "number"},
                },
              },
            },
          },
        ),
      }
    else:
      return {
        "topic": f"vid_{name}",
        "encoding": "protobuf",
        "schemaName": CompressedImage.DESCRIPTOR.full_name,
        "schema": b64encode(
          build_file_descriptor_set(CompressedImage).SerializeToString()
        ).decode("ascii"),
      }

  async def _run(self):
    async with FoxgloveServer("127.0.0.1", 8765, "Depthai-Foxglove") as server:
      self.server = server
      server.set_listener(self.Listener())

      while not self.shouldStop_.is_set():
        await asyncio.sleep(1 / 60)

        for channel in self.channels_:
          if channel["id"] == None:
            channel["id"] = await server.add_channel(self._channel_description(channel["name"]))
          elif channel["name"] == None:
            await server.remove_channel(channel["id"])
            self.channels_.remove(channel)
            continue
          elif channel["name"] == "pointcloud":
            await self.send_pointcloud(self.cameraData_, channel)
          elif channel["name"] == "imu":
            if self.cameraData_.getIMU() != {}:
              await server.send_message(channel["id"], time.time_ns(), json.dumps(self.cameraData_.getIMU()).encode("utf8"))
          else:
            raw_image = CompressedImage()
            raw_image.frame_id = f"vid_{channel['name']}"
            raw_image.format = "jpeg"

            if channel["name"] == "color":
              frame = self.cameraData_.getColorFrame()
            elif channel["name"] == "nn":
              frame = self.cameraData_.getNnFrame()
            elif channel["name"] == "left":
              frame = self.cameraData_.getLeftFrame()
            elif channel["name"] == "right":
              frame = self.cameraData_.getRightFrame()
            elif channel["name"] == "stereo":
              frame = self.cameraData_.getStereoFrame()

            raw_image.data = bytes(frame)
            raw_image.timestamp.FromNanoseconds(time.time_ns())

            await server.send_message(channel["id"], time.time_ns(), raw_image.SerializeToString())
    return not self.shouldStop_.is_set()

  @ImportantThread("Foxglove Runner")
  def run(self, loop):
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(self._run())

  def stop(self):
    self.shouldStop_.set()
