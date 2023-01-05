import depthai as dai
import cv2

class Network():
  def __init__(self):
    self._running = True
    self._has = False
    pass

  def createNodes(self, pipeline: dai.Pipeline, camRgb: dai.node.ColorCamera, sync: bool = True) -> None:
    raise NotImplementedError()

  def frame(self) -> cv2.Mat:
    raise NotImplementedError()

  def draw(self, frame: cv2.Mat) -> None:
    raise NotImplementedError()

  def start(self) -> None:
    raise NotImplementedError()

  def stop(self) -> None:
    self._running = False

  def has(self) -> bool:
    return self._has
