import numpy as np
import cv2


#Function for Iris Detection and Pupil Detection convert it into gray scale
#For Noise Removal, We have use erosion and apply thresholding to count the number of pixel for Black and White part of the eye
#Applied contouring and get the iris and pupil boundary region

class Pupil(object):

    def __init__(self, eyefr, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        self.detect_iris(eyefr)

    @staticmethod
    def image_processing(eyefr, threshold):
        kernel = np.ones((3, 3), np.uint8)
        newframe = cv2.bilateralFilter(eyefr, 10, 15, 15)
        newframe = cv2.erode(newframe, kernel, iterations=3)
        newframe = cv2.threshold(newframe, threshold, 255, cv2.THRESH_BINARY)[1]

        return newframe

    def detect_iris(self, eyefr):

        self.iris_frame = self.image_processing(eyefr, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moment = cv2.moments(contours[-2])
            self.x = int(moment['m10'] / moment['m00'])
            self.y = int(moment['m01'] / moment['m00'])
        except (IndexError, ZeroDivisionError):
            pass
