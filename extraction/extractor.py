import cv2
import numpy as np
import sys
import time
from extraction.extraction_utils import *

class Extractor:

    def __init__(self):
        # cascade path
        faceCascPath = "extraction/haarcascade_frontalface_default.xml"
        eyeCascPath = "extraction/haarcascade_eye.xml"

        # create the haar cascades
        self.faceCascade = cv2.CascadeClassifier(faceCascPath)
        self.eyeCascade = cv2.CascadeClassifier(eyeCascPath)

        # default box config x,y,w,h
        self.box_config = [0,0,0,0]
        self.best_eyes = None
        self.selectedFace = [0,0,0,0]

    # returns face, eyes and config
    def getFacialData(self, gray_image):
        # detect faces
        faces = list(self.faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        ))
        # select the largest face - most probable
        try:
            faces.sort(reverse=True, key=lambda face: face[2]*face[3])
            selectedFace = faces[0]
        except:
            # no face found
            return [[], [], []]
        x, y, w, h = selectedFace
        roi_gray = gray_image[y:y+h, x:x+w]
        # detect eyes
        eyes = list(self.eyeCascade.detectMultiScale(
    	   roi_gray,
           scaleFactor=1.1,
	       minSize=(10, 10),
        ))
        [best_eyes, cropped] = getBestEyes(selectedFace, list(eyes))
        if(cropped[2]/cropped[3]<2):
            return [[], [], []]
        if(compareRectangles(self.selectedFace, selectedFace)<1 and
            compareRectangles(cropped, self.box_config)<(2/3)):
                best_eyes = self.best_eyes
        self.box_config = cropped
        self.best_eyes = best_eyes
        self.selectedFace = selectedFace
        return [selectedFace, self.best_eyes, self.box_config]
