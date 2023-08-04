import pyautogui # I do not know why, but however, this line seems to be REALLY important ?!?!?!

import config as C
import processing
from calibration import Calibration
from cameras import IriunCameraInput
from filterapp import FilterApp
from screen import ScreenInput


class Main:
    def __init__(self):
        # self.screen_input = ScreenInput()

        self.cameras = [IriunCameraInput(), ]

        self.calibration = Calibration(camera_sources=self.cameras)
        self.transformation = self.calibration()

        self.fingertip_detector = processing.DetectIndexfingertip(self.cameras[0].resolution, self.transformation)
        self.gesture_detector = processing.GestureDetection()

        self.filter_app = FilterApp()


    def __call__(self):
        while True:
            screen_img = self.screen_input()
            camera_img = self.cameras[0]()

            try:
                onscreen_fingertip_pos = self.fingertip_detector(self.transformation.region_of_interest(camera_img))
                latest_fingertip_positions, action_point = self.gesture_detector.detect(onscreen_fingertip_pos)

            except C.NoFingerDetectedError:
                self.gesture_detector.clear()
                continue

            if action_point != -1:
                print(f"circle at:", str(action_point))
                self.filter_app.draw_click((abs(action_point[0]), abs(action_point[1])))


            self.filter_app.overlay_at(onscreen_fingertip_pos)
            self.filter_app.draw_arrows(latest_fingertip_positions)
            self.filter_app.update()


           # TODO: the action_point has negative coordinates. for now its jst abs()t out, but that needs to be fixed
            # -> TODO: possibly there could be added a timer so than only all n seconds screen is updated (fps limiter)


            # self.gesture_detector.detect # TODO

            #rel_pos = indexfingertip_relpos[0]*C.MONITOR_RESOLUTION  # TODO: Check how indexfingertip_relpos is being spit out
            #transformed_pos = processing.transform_relative_pos(rel_pos)
            #pyautogui.moveTo(*transformed_pos)
            




_ = Main()
_()