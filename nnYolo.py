import depthai as dai
import blobconverter
import numpy as np
import threading
import cv2

from networkBase import Network

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

class YoloNetwork(Network):
  def __init__(self):
    super().__init__()

  def nnThread(self):
    qNN = self.device.getOutputQueue("nn")
    while self._running:
      try:
        self.detections = qNN.get().detections
      except RuntimeError as ex:
        continue

  def createNodes(self, pipeline: dai.Pipeline, camRgb: dai.node.ColorCamera, sync: bool = True) -> None:
    nn = pipeline.createYoloDetectionNetwork()

    nn.setConfidenceThreshold(0.5)
    nn.setNumClasses(80)
    nn.setCoordinateSize(4)
    nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
    nn.setIouThreshold(0.5)
    nn.setBlobPath(blobconverter.from_zoo(name="yolo-v4-tiny-tf", shaves=6))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")
    nn.out.link(nnOut.input)

    inImage = pipeline.createImageManip()
    camRgb.preview.link(inImage.inputImage)
    inImage.out.link(nn.input)
    inImage.initialConfig.setResize(416, 416)
    inImage.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

  def draw(self, frame: cv2.Mat):
    color = (255, 0, 0)
    for detection in self.detections:
      bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
      cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
      cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
      cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

  def start(self, device: dai.Device):
    self.device = device
    self.detections = []
    self.threads = [
      threading.Thread(target=self.nnThread),
    ]
    for thread in self.threads:
      thread.start()
