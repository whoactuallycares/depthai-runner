import numpy as np
import threading
import cv2

class CameraData():
  def __init__(self):
    #squareSize = 3
    #squareCount = 2
    #colorA = [255,255,255]
    #colorB = [255,255,255]
    #checkerboard = np.tile(np.concatenate((np.tile(colorA*squareSize + colorB*squareSize, (squareSize,1)).reshape((6*(squareSize**2))), np.tile(colorB*squareSize + colorA*squareSize, (squareSize,1)).reshape((6*(squareSize**2))))).reshape((2*squareSize,6*squareSize)), (squareCount,squareCount))
    #self.colorFrame = np.full((1000, 1000, 3), 255, dtype=np.uint8)

    emptyFrame = np.full((256, 256, 3), 0, dtype=np.uint8)

    self.colorFrame =  emptyFrame
    self.stereoFrame = emptyFrame
    self.leftFrame = emptyFrame
    self.rightFrame = emptyFrame
    self.nnFrame = emptyFrame
    self.imu = {}
    self.lock = threading.Lock()

  def __getData(self, name: str) -> any:
    self.lock.acquire()
    data = self.__dict__[name]
    self.lock.release()
    return data

  def __setData(self, name: str, value: any) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encimg = cv2.imencode('.jpg', value, encode_param)
    self.lock.acquire()
    self.__dict__[name] = encimg
    self.lock.release()

  def getColorFrame(self) -> np.ndarray:
    return self.__getData("colorFrame")

  def setColorFrame(self, value: np.ndarray) -> None:
    self.__setData("colorFrame", value)

  def getLeftFrame(self) -> np.ndarray:
    return self.__getData("leftFrame")

  def setLeftFrame(self, value: np.ndarray) -> None:
    self.__setData("leftFrame", value)

  def getRightFrame(self) -> np.ndarray:
    return self.__getData("rightFrame")

  def setRightFrame(self, value: np.ndarray) -> None:
    self.__setData("rightFrame", value)

  def getStereoFrame(self) -> np.ndarray:
    return self.__getData("stereoFrame")

  def setStereoFrame(self, value: np.ndarray) -> None:
    self.__setData("stereoFrame", value)

  def getNnFrame(self) -> np.ndarray:
    return self.__getData("nnFrame")

  def setNnFrame(self, value: np.ndarray) -> None:
    self.__setData("nnFrame", value)

  def getIMU(self) -> dict:
    return self.imu
    #return self.__getData("imu")

  def setIMU(self, value: dict) -> None:
    self.imu = value
    #self.__setData("imu", value)
