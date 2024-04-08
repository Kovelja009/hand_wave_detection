import cv2
import depthai as dai
import time


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0

        self._coordinates = (20, 20)
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.7
        self._color = (0, 0, 255)
        self._thickness = 1

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

    def show_fps(self, frame, fps):
        return cv2.putText(frame, fps.__str__(), self._coordinates, self._font, self._font_scale, self._color,
                           self._thickness, cv2.LINE_AA)


class OAK_D:
    def __init__(self, fps=24, width=1920, height=1080):
        # Create pipeline
        self._pipeline = dai.Pipeline()

        # Define source and output
        self._camRgb = self._pipeline.create(dai.node.ColorCamera)
        self._xoutVideo = self._pipeline.create(dai.node.XLinkOut)

        self._xoutVideo.setStreamName("video")

        # Properties
        self._camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self._camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self._camRgb.setVideoSize(width, height)
        self._camRgb.setFps(fps)

        self._xoutVideo.input.setBlocking(False)
        self._xoutVideo.input.setQueueSize(1)

        # Linking
        self._camRgb.video.link(self._xoutVideo.input)

        # Connect to device and start pipeline
        self._device = dai.Device(self._pipeline)
        self._video = self._device.getOutputQueue(name="video", maxSize=1, blocking=False)
        self.fps_handler = FPSHandler()
        self.height = self._camRgb.getVideoHeight()
        self.width = self._camRgb.getVideoWidth()

    '''Returns the color frame with number of fps from the camera with or without FPS overlay'''
    def get_color_frame(self, show_fps=False):
        video_in = self._video.get()
        # convert from 
        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        cv_frame = video_in.getCvFrame()
        self.fps_handler.next_iter()
        fps = self.fps_handler.fps()
        
        if show_fps:
            # return video_in.getCvFrame()
            return self.fps_handler.show_fps(cv_frame, round(fps, 2)), fps
        else:
            return cv_frame, fps


if __name__ == '__main__':
    oak_d = OAK_D(fps=60, width=300, height=300)
    while True:
        frame = oak_d.get_color_frame(show_fps=True)
        cv2.imshow("VidraCar", frame)
        if cv2.waitKey(1) == ord('q'):
            break
