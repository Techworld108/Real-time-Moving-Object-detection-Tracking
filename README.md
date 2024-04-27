# Real-time-Moving-Object-detection-Tracking
## Moving Object Detection
Moving object detection is a technique used in computer vision and image processing. Multiple consecutive frames from a video are compared by various methods to determine if any moving object is detected.
# Image Resize
import cv2

import imutils

img = cv2.imread(‘sample2.jpg')

resizedImg = imutils.resize(img, width=500)

cv2.imwrite(‘resizedImage.jpg', resizedImg)

# Gaussian Blur - Smoothening
import cv2

img = cv2.imread(‘sample2.jpg')

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#dst = cv2.GaussianBlur(src, (kernel),borderType)

gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

cv2.imwrite(“GaussianBlur.jpg”, gaussianImg)

# Threshold
### dst = cv2.threshold(src, threshold, maxValueForThreshold,binary,type)[1] 

import cv2

img=cv2.imread("sample.jpg")

grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gaussBlur = cv2.GaussianBlur(grayImg,(21,21),0)

thresholdImg = cv2.threshold(grayImg,150,255,cv2.THRESH_BINARY)[1]

cv2.imwrite("threshold.jpg",thresholdImg)

# Drawing Rectangle
### cv2.rectangle(src,startpoint,endpoint,color,thickness)

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Putting Text in Image
### cv2.putText(src, text, position,font,fontSize,color,thickness)

cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# findContours
### dst =cv2.findContours(srcImageCopy, contourRetrievalMode, contourApproximationMethod)
 
cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Reading frame from camera – video streaming
import cv2

vs = cv2.VideoCapture(0)

while True:

	_,img = vs.read()
 
	cv2.imshow("VideoStream", img)
 
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
 
		break
  
vs.release()

cv2.destroyAllWindows()

# Moving Object detection
import imutils

import time

import cv2

vs = cv2.VideoCapture(0)

firstFrame = None

area=500

while True:

	_,img = vs.read()
 
	text = "Normal"
 
	img = imutils.resize(img, width=500)
 
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
	grayImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
 
	if firstFrame is None:
 
		firstFrame = grayImg
  
		continue
  
	imgDiff = cv2.absdiff(firstFrame, grayImg)
 
	threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
 
	threshImg = cv2.dilate(threshImg, None, iterations=2)
 
	cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
 
	for c in cnts:
 
		if cv2.contourArea(c) < area:
  
			continue
   
		(x, y, w, h) = cv2.boundingRect(c)
  
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
		text = "Moving Object detected"
  
		print(text)
  
	cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
	cv2.imshow("VideoStream", img)
 
	cv2.imshow("Thresh", threshImg)
 
	cv2.imshow("Image Difference", imgDiff)
 
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
 
		break
  
vs.release()

cv2.destroyAllWindows()
























