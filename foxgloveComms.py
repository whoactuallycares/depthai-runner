from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_websocket import run_cancellable
from foxglove_websocket.types import ChannelId
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf.descriptor import FileDescriptor
import google.protobuf.message
from base64 import b64encode
from typing import Set, Type
import logging
import asyncio
import json
import time
import struct

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
    self.channels = []

  class Listener(FoxgloveServerListener):
    def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
      logging.debug("First client subscribed to", channel_id)

    def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
      logging.debug("Last client unsubscribed from", channel_id)

  async def add_video_channel(self, name: str):
    self.channels += [{"id": await self.server.add_channel(
      {
        "topic": f"vid_{name}",
        "encoding": "protobuf",
        "schemaName": CompressedImage.DESCRIPTOR.full_name,
        "schema": b64encode(
          build_file_descriptor_set(CompressedImage).SerializeToString()
        ).decode("ascii"),
      }
    ), "name": name}]
    print(f"Adding video channel {name}")

  async def add_imu_channel(self):
    self.channels += [{"id": await self.server.add_channel(
      {
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
    ), "name": "imu"}]

  async def add_pointcloud_channel(self):
    self.channels += [{"id": await self.server.add_channel(
      {
        "topic": "pointcloud",
        "encoding": "protobuf",
        "schemaName": PointCloud.DESCRIPTOR.full_name,
        "schema": b64encode(
          build_file_descriptor_set(PointCloud).SerializeToString()
        ).decode("ascii"),
      }
    ), "name": "pointcloud"}]

  async def remove_channel(self, name: str):
    channel = next(filter(lambda chan: chan["name"] == name, self.channels))
    await self.server.remove_channel(channel["id"])

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

  async def _run(self, camData: CameraData):
    async with FoxgloveServer(self.address, self.port, self.name) as server:
      self.server = server
      server.set_listener(self.Listener())
      await self.add_video_channel("color")
      await self.add_video_channel("nn")
      await self.add_video_channel("stereo")
      await self.add_imu_channel()
      await self.add_pointcloud_channel()

      while True:
        await asyncio.sleep(1 / 60)

        for channel in self.channels:
          if channel["name"] == "pointcloud":
            await self.send_pointcloud(camData, channel)
          elif channel["name"] == "imu":
            if camData.getIMU() != {}:
              await server.send_message(channel["id"], time.time_ns(), json.dumps(camData.getIMU()).encode("utf8"))
          else:
            raw_image = CompressedImage()
            raw_image.frame_id = f"vid_{channel['name']}"
            raw_image.format = "jpeg"

            if channel["name"] == "color":
              frame = camData.getColorFrame()
            elif channel["name"] == "nn":
              frame = camData.getNnFrame()
            elif channel["name"] == "left":
              frame = camData.getLeftFrame()
            elif channel["name"] == "right":
              frame = camData.getRightFrame()
            elif channel["name"] == "stereo":
              frame = camData.getStereoFrame()

            raw_image.data = bytes(frame)
            raw_image.timestamp.FromNanoseconds(time.time_ns())

            await server.send_message(channel["id"], time.time_ns(), raw_image.SerializeToString())


  def run(self, camData: CameraData):
    run_cancellable(self._run(camData))
