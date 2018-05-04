import cv2
import numpy as np
import time
import sys

class YawnDetector:

	def __init__(self):
		# cascade path
		faceCascPath = "detection/haarcascade_frontalface_alt2.xml"

		# create the cascade
		self.faceCascade = cv2.CascadeClassifier(faceCascPath)

		# class variables
		self.ratio = 0 #contour area and ROI ratio
		self.yawnAvgTime = 0
		self.yawnStartTime = True
		self.yawnRatioCount = [] # List to hold yawn ratio count and timestamp
		self.yawnCounter = 0
		self.yawnTime = 1 # fix yawn time

	def makeContours(self, image, contours):
		cv2.drawContours(image, contours, -1, (0,255,0), 3)
		maxArea = 0
		secondMax = 0
		maxCount = 0
		secondmaxCount = 0
		for i in contours:
			count = i
			area = cv2.contourArea(count)
			if maxArea < area:
				secondMax = maxArea
				maxArea = area
				secondmaxCount = maxCount
				maxCount = count
			elif (secondMax < area):
				secondMax = area
				secondmaxCount = count
		return [secondmaxCount, secondMax]

	def thresholdContours(self, mouthRegion, rectArea):
		imgray = cv2.equalizeHist(cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY))
		# Thresholding each pixel to 255 if that pixel each exceeds 64 else convert it to 0
		ret, thresh = cv2.threshold(imgray, 64, 255, cv2.THRESH_BINARY)
		# Contouring binary image by having the contoured region made by of small rectangles and storing only the end points
		# of the rectangle
		contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,offset=(2,2))[1]
		returnValue = self.makeContours(mouthRegion, contours)
		secondMaxCount = returnValue[0]
		contourArea = returnValue[1]
		self.ratio = contourArea / rectArea
		# Draw contours in the image passed. The contours are stored as vectors in the array.
		# -1 indicates the thickness of the contours
		if(isinstance(secondMaxCount, np.ndarray) and len(secondMaxCount) > 0):
			cv2.drawContours(mouthRegion, [secondMaxCount], 0, (255,0,0), -1)

	#Isolates the region of interest and detects if a yawn has occured
	def detectYawn(self, image):
	   	# Capture frame-by-frame
		gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		faces = self.faceCascade.detectMultiScale(
        	gray,
        	scaleFactor=1.1,
        	minNeighbors=5,
        	minSize=(50, 50),
        	flags=cv2.CASCADE_SCALE_IMAGE
    	)

		for (x, y, w, h) in faces:
			# Isolate the ROI as the mouth region
			widthOneCorner = int(x + (w / 4))
			widthOtherCorner = int(x + ((3 * w) / 4))
			heightOneCorner = int(y + (11 * h / 16))
			heightOtherCorner = int(y + h)
			# Indicate the ROI as the mouth by highlighting it
			cv2.rectangle(image, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner),(0,0,255), 2)
			# Mouth region
			mouthRegion = image[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]
			# Area of the bottom half of the face rectangle
			rectArea = (w*h)/2
			if(len(mouthRegion) > 0):
				self.thresholdContours(mouthRegion, rectArea)
			# print ("Current probablity of yawn: " + str(round(self.ratio*1000, 2)) + "%")
			# print ("Length of yawnCounter: " + str(len(self.yawnRatioCount)))
			if(self.ratio > 0.06):
				if(self.yawnStartTime is True):
					self.yawnStartTime = False
					self.yawnAvgTime = time.time()
				# If the mouth is open for more than yawnTime seconds, classify it as a yawn
				if((time.time() - self.yawnAvgTime) >= self.yawnTime):
					self.yawnCounter += 1
					self.yawnRatioCount.append(self.yawnCounter)
					self.yawnStartTime = True
					self.yawnAvgTime = 0
					return True
		return False
