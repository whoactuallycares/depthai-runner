from enum import IntEnum
import depthai as dai
import blobconverter
import numpy as np
import cv2

class NNType(IntEnum):
  New = -1
  Yolo = 0
  YoloSpatial = 1
  MobileNet = 2
  MobileNetSpatial = 3

labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

class NNHandler():
  def __init__(self, blobpath, nnType=NNType.Yolo, inputSize: tuple[int, int] = (400, 400), interleaved = False):
    self.nnType = nnType
    self.interleaved = interleaved
    self.blobpath = blobpath
    self.inputSize = inputSize
    self.network = None

  def create(self, pipeline: dai.Pipeline, inImg: dai.node.ImageManip, lastHandler = None):
    if lastHandler != None and lastHandler.nnType == self.nnType:
      self.network = lastHandler.network
    else:
      if self.nnType == NNType.Yolo:
        self.network = pipeline.createYoloDetectionNetwork()
      elif self.nnType == NNType.YoloSpatial:
        self.network = pipeline.createYoloSpatialDetectionNetwork()
      elif self.nnType == NNType.MobileNet:
        self.network = pipeline.createMobileNetDetectionNetwork()
      elif self.nnType == NNType.MobileNetSpatial:
        self.network = pipeline.createMobileNetSpatialDetectionNetwork()

    self.network.setConfidenceThreshold(0.5)
    self.network.setNumClasses(80)
    self.network.setCoordinateSize(4)
    self.network.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    self.network.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
    self.network.setIouThreshold(0.5)
    self.network.setBlobPath(self.blobpath)
    self.network.setNumInferenceThreads(2)
    self.network.input.setBlocking(False)

    inImg.out.link(self.network.input)
    inImg.initialConfig.setResize(self.inputSize[0], self.inputSize[1])
    if self.interleaved:
      inImg.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888i)
    else:
      inImg.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

    return self.network

  def draw(self, detections, frame: cv2.Mat):
    color = (255, 0, 0)
    for detection in detections:
      bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
      cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
      cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
      cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
