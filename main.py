from depthai_sdk import OakCamera, DetectionPacket, Visualizer
from foxgloveComms import FoxgloveUploader
from cameraData import CameraData
import depthai as dai
import threading
import cv2

ENABLE_CV2 = True
ENABLE_FOXGLOVE = True

def main(cameraData: CameraData) -> None:
  with OakCamera(usbSpeed=dai.UsbSpeed.HIGH) as oak:
    color = oak.create_camera('color')
    left = oak.create_camera('left')
    right = oak.create_camera('right')
    nn = oak.create_nn('mobilenet-ssd', color)
    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    # Create visualization callbacks
    def colorCb(packet: DetectionPacket, visualizer: Visualizer):
      cameraData.setColorFrame(packet.frame)
      if ENABLE_CV2:
        cv2.imshow("colorFrame", packet.frame)

    def leftCb(packet: DetectionPacket, visualizer: Visualizer):
      cameraData.setLeftFrame(packet.frame)
      if ENABLE_CV2:
        cv2.imshow("leftFrame", packet.frame)

    def rightCb(packet: DetectionPacket, visualizer: Visualizer):
      cameraData.setRightFrame(packet.frame)
      if ENABLE_CV2:
        cv2.imshow("rightFrame", packet.frame)

    def nnCb(packet: DetectionPacket, visualizer: Visualizer):
      cameraData.setNnFrame(packet.frame)
      if ENABLE_CV2:
        cv2.imshow("nnFrame", packet.frame)

    def imuCb(packet: DetectionPacket, visualizer: Visualizer):
      cameraData.setIMU(packet.frame)


    oak.visualize(nn, fps=True, callback=nnCb)
    oak.visualize(color, fps=True, callback=colorCb)
    oak.visualize(left, fps=True, callback=leftCb)
    oak.visualize(right, fps=True, callback=rightCb)
    #oak.visualize(imu, fps=True, callback=imuCb)


    pipeline = oak.build()

    nn.node.setNumInferenceThreads(2)

    features = pipeline.create(dai.node.FeatureTracker)
    color.node.video.link(features.inputImage)

    out = pipeline.create(dai.node.XLinkOut)
    out.setStreamName('features')
    features.outputFeatures.link(out.input)

    # Start the pipeline (upload it to the OAK)
    oak.start()


    while running and oak.running():
      oak.poll()


if __name__ == "__main__":

  cd = CameraData()

  running = threading.Event()
  thread = threading.Thread(target=main, args=(cd,))
  thread.start()

  if ENABLE_FOXGLOVE:
    fgUp = FoxgloveUploader()
    fgUp.run(cd)
    running = False
    thread.join()