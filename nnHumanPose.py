from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
from depthai_sdk.managers import PipelineManager, NNetManager
from syncNodes import createSyncNodes
from networkBase import Network
from pathlib import Path
import depthai as dai
import blobconverter
import numpy as np
import threading
import cv2



colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

class PhonyCamera():
  def __init__(self, manip):
    self.preview = manip.out

class HumanPoseNetwork(Network):
  def __init__(self):
    self.pose = None
    self.keypoints_list = None
    self.detected_keypoints = None
    self.personwiseKeypoints = None

    self.nm = NNetManager(inputSize=(456, 256))
    self.pm = PipelineManager()
    self.pm.setNnManager(self.nm)
    self.hase = False
    super().__init__()

  def decode_thread(self, in_queue):
    while self._running:
      try:
        raw_in = in_queue.get()
        print(f"SeqNR {raw_in.getSequenceNum()}")
        self._has = True
      except RuntimeError:
        return
      heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
      pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
      heatmaps = heatmaps.astype('float32')
      pafs = pafs.astype('float32')
      outputs = np.concatenate((heatmaps, pafs), axis=1)

      new_keypoints = []
      new_keypoints_list = np.zeros((0, 3))
      keypoint_id = 0

      for row in range(18):
        probMap = outputs[0, row, :, :]
        probMap = cv2.resize(probMap, self.nm.inputSize)  # (456, 256)
        keypoints = getKeypoints(probMap, 0.3)
        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
        keypoints_with_id = []

        for i in range(len(keypoints)):
          keypoints_with_id.append(keypoints[i] + (keypoint_id,))
          keypoint_id += 1

        new_keypoints.append(keypoints_with_id)

      valid_pairs, invalid_pairs = getValidPairs(outputs, w=self.nm.inputSize[0], h=self.nm.inputSize[1], detected_keypoints=new_keypoints)
      newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

      self.detected_keypoints, self.keypoints_list, self.personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)

  def createNodes(self, pipeline: dai.Pipeline, camRgb: dai.node.ColorCamera, sync: bool = True) -> None:
    manip = pipeline.createImageManip()
    camRgb.preview.link(manip.inputImage)
    manip.initialConfig.setResize(456, 256)
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

    self.pm.nodes.camRgb = PhonyCamera(manip)
    nn = self.nm.createNN(pipeline, self.pm.nodes, source="color", blobPath=Path(blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=6)), fullFov=True)
    nn.setNumInferenceThreads(3)
    nn.setNumPoolFrames(32)
    self.pm.addNn(nn=nn)

    manip.out.link(nn.input)

    if sync:
      createSyncNodes(pipeline, [nn.out, camRgb.video], ["nn", "rgb"])
    else:
      nnOut = pipeline.create(dai.node.XLinkOut)
      nnOut.setStreamName("nn_out")
      nn.out.link(nnOut.input)

    #inImage.out.link(nn.input)
    #inImage.initialConfig.setResize(416, 416)
    #inImage.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

  def draw(self, frame: cv2.Mat):
    if self.keypoints_list is not None and self.detected_keypoints is not None and self.personwiseKeypoints is not None:
      scale_factor = frame.shape[0] / self.nm.inputSize[1]
      offset_w = int(frame.shape[1] - self.nm.inputSize[0] * scale_factor) // 2

      def scale(point):
        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

      for i in range(18):
        for j in range(len(self.detected_keypoints[i])):
          cv2.circle(frame, scale(self.detected_keypoints[i][j][0:2]), 5, colors[i], -1, cv2.LINE_AA)
      for i in range(17):
        for n in range(len(self.personwiseKeypoints)):
          index = self.personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
          if -1 in index:
            continue
          B = np.int32(self.keypoints_list[index.astype(int), 0])
          A = np.int32(self.keypoints_list[index.astype(int), 1])
          cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)
      self._has = False

  def start(self, device: dai.Device):
    self.device = device
    self.detections = []
    self.nm.createQueues(device)
    outputQueue = device.getOutputQueue("nn_out")
    self.threads = [
      threading.Thread(target=self.decode_thread, args=(outputQueue, ))
    ]
    for thread in self.threads:
      thread.start()
