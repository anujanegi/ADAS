import cv2
import numpy as np
import math

class HeadMovement:

    def __init__(self):
        # cascade path
        faceCascPath = "detection/haarcascade_frontalface_alt2.xml"
		# create the cascade
        self.faceCascade = cv2.CascadeClassifier(faceCascPath)
        # parameters for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
        # parameters for Lucas-Kanade method
        self.lkPara = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        #define movement thresholds
        self.max_head_movement = 10
        self.movement_threshold = 50
        self.gesture_threshold = 100
        # other global variables
        self.face_found = False
        self.frame_num = 0
        self.x_movement = 0; self.y_movement = 0
        self.old_face = None

    def distance(self, x, y):
        return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

    #Function to get coordinates
    def get_coords(self, p1):
        try: return int(p1[0][0][0]), int(p1[0][0][1])
        except: return int(p1[0][0]), int(p1[0][1])

    def detectMovement(self, image):
        x=0;y=0;w=0;h=0
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.frame_num == 0:
            # Take first frame and find corners
            self.frame_num += 1
            faces = self.faceCascade.detectMultiScale(frame_gray, 1.3, 5)
            for (x,y,w,h) in faces:
                self.face_found = True
            self.old_face = frame_gray.copy()
            face_center = x+w/2, y+h/3
            self.p0 = np.array([[face_center]], np.float32)
            return False

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_face, frame_gray, self.p0, None, **self.lkPara)

        #get the xy coordinates for points p0 and p1
        a,b = self.get_coords(self.p0), self.get_coords(p1)
        self.x_movement += abs(a[0]-b[0])
        self.y_movement += abs(a[1]-b[1])

        if self.y_movement > self.gesture_threshold:
            return True
        if self.x_movement > self.gesture_threshold:
            return False

        self.p0 = p1

        return False
