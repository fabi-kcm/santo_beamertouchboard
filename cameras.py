from cv2 import VideoCapture
import cv2

import config as C


class IriunCameraInput:
    def __init__(self):
        self.cap = VideoCapture(C.IRIUN_CAMERA_NO)
        _ = self.__call__()
        self.width = _.shape[1]
        self.height = _.shape[0]
        self.resolution = (self.width, self.height)
    def __call__(self):
        ret, frame = self.cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        self.cap.release()
