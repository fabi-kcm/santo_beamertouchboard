import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame as pyg
import win32api
import win32con
import win32gui

import config as C
from cameras import IriunCameraInput


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def calc_difference(imga, imgb):
    assert imga.shape[0] == imgb.shape[0]
    res = cv2.subtract(imga, imgb) # difference of images
    #cv2.imshow("res", res)
    #cv2.waitKey(0)

    red_only = res[:,:,2] #red only

def transform_relative_pos(relpos, transformation, camera_res):
    pos = (
        relpos[0] * camera_res[0],
        relpos[1] * camera_res[1]
    )

    return transformation(pos)


class _FilterApp:
    def __init__(self):
        self.screen = pyg.display.set_mode(C.MONITOR_RESOLUTION, pyg.FULLSCREEN)
        self.TRANSPARENCY_COLOR = (1, 1, 1)
        self.WHITE = (255, 255, 255)

        # for borderless window use pygame.Noframe
        # size of the pygame window will be of width 700 and height 450
        hwnd = pyg.display.get_wm_info()["window"]
        # Getting information of the current active window
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(
            hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED
        )

        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*self.TRANSPARENCY_COLOR), 0, win32con.LWA_COLORKEY)

        self.screen.fill((0, 0, 255))
        self.screen.fill(self.TRANSPARENCY_COLOR)

        pyg.display.update()

    def overlay_at(self, pos: tuple[int, int]):
        self.screen.fill(self.TRANSPARENCY_COLOR)
        pyg.draw.circle(self.screen, self.WHITE, pos, int(0.2*C.MONITOR_RESOLUTION[1]))
        pyg.draw.circle(self.screen, (255, 0, 0), pos, 5)

        pyg.display.update()

class DetectIndexfingertip:
    def __init__(self, camera_res, transformation):
        self.transformation = transformation

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.1,
            max_num_hands=1
        )

        self.img_width = camera_res[0]
        self.img_height = camera_res[1]

        self.filter_window = _FilterApp()

    def __call__(self, image):
        results, landmark = self.find_finger(image)
        relpos = landmark.x, landmark.y
        self.show_detected_fingertip(results, landmark, image)
        point_on_display = self.transform_relative_pos(relpos)
        return point_on_display
        # self.filter_window.overlay_at(point_on_display)


    def find_finger(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # image.flags.writeable = False
        _ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(_)
        # Draw the hand annotations on the image.
        #image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not results.multi_hand_landmarks:
            raise C.NoFingerDetectedError

        normalizedLandmark = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return results, normalizedLandmark

    def show_detected_fingertip(self, results, landmark, image):
        pixelCoordinatesLandmark = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x,
                                                                                    landmark.y,
                                                                                    self.img_width,
                                                                                    self.img_height)
        cv2.circle(image, pixelCoordinatesLandmark, 2, (255, 0, 0), -1)

        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(1)

    def transform_relative_pos(self, relpos: tuple[int, int]):
        return self.transformation.transform_point((relpos[0]*self.img_width, relpos[1]*self.img_height))


class GestureDetection:
    def __init__(self):
        self.clear()

    def clear(self):
        self.coords = deque([(0, 0), ], maxlen=C.GESTURE_DETECTION_CONTEXT_LEN)

    def detect(self, new_pos): # new_pos should be a not-relative position on the ACTUAL SCREEN!
        if not distance(new_pos, self.coords[-1]) >= C.NEW_HAND_MOVE_DIVIATION_TRESHOLD:
            return self.coords, -1

        self.coords.append(new_pos)
        circle_detected, center = self.find_circular_subchain()
        if circle_detected:
            self.clear()
            return self.coords, center
        else:
            return self.coords, -1

    def find_circular_subchain(self, min_length=C.GESTURE_DETECTION_CONTEXT_LEN, tolerance=C.GESTURE_DETECTION_TOLERANCE):
        """
        Check for any subchain of points that forms a circular shape within a given tolerance.

        :param self.coords: np.array with shape (n, 2)
        :param min_length: minimum length of a circular subchain
        :param tolerance: maximum allowed average deviation from the circle (as a percentage of the radius)
        :returns: Boolean indicating if a circular subchain exists, and the center of the circle if it exists, None otherwise.
        """

        coords = np.array(self.coords)
        n = len(coords)

        for i in range(n - min_length + 1):
            subchain = coords[i:i + min_length]

            # Fit a circle to the points
            (h, k), r = self.least_squares_circle(subchain)

            # Compute distances from the points to the circle center
            distances = np.sqrt((subchain[:, 0] - h) ** 2 + (subchain[:, 1] - k) ** 2)

            # Compare the distances to the circle radius
            deviations = np.abs(distances - r)

            # If the average deviation is within the tolerance, it's a circle
            if np.mean(deviations) <= tolerance * r:
                return True, (h, k)

        return False, None

    @staticmethod
    def least_squares_circle(coords):
        """
        Fit a circle using the least squares method.

        :param coords: np.array with shape (n, 2)
        :returns: center (h, k), radius
        """
        x, y = coords[:, 0], coords[:, 1]
        b = -2 * np.array([x, y, np.ones(x.shape)]).T
        c = (x ** 2 + y ** 2)[:, np.newaxis]

        # Solve the linear system to get circle parameters
        h, k, r_sq = np.linalg.lstsq(b, c, rcond=None)[0].flatten()
        r = np.sqrt(r_sq)

        return (h, k), r



if __name__ == '__main__':
    t = IriunCameraInput()
    p = DetectIndexfingertip((t.width, t.height))
    while True:
        p(t())


#https://techtutorialsx.com/2021/04/24/python-mediapipe-finger-roi/