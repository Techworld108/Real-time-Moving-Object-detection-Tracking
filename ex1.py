import cv2
import imutils
img = cv2.imread("sample2.jpg")
resizeImg = imutils.resize(img,width=20)
cv2.imwrite("resizedImage.jpg",resizeImg)
