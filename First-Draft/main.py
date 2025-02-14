import cv2                    
import numpy as np
from functions import empty        
from functions import stackImages       # Import stackImages function from function file
from functions import getContours       # Import getContours function from function file

frameWidth = 640                  
frameHeight = 480

cap = cv2.VideoCapture(1)               # Parameter represents an index for video camera to use for capture (Default 0)
cap.set(3, frameWidth)                  # Set video capture width
cap.set(4, frameHeight)                 # Set video capture height

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 1000, 120)

cv2.createTrackbar("1", "Parameters", 150, 255, empty)                # Create first threshold slider
cv2.createTrackbar("2", "Parameters", 150, 255, empty)                # Create second threshold slider
cv2.createTrackbar("Area", "Parameters", 3000, 15000, empty)          # Create area sldier

while(cap.isOpened()):                                                # Open video capture
     _, frame = cap.read()
     kernel = np.ones((5,5))
     imgContour = frame.copy()
     count = 0

     threshold1 = cv2.getTrackbarPos("1", "Parameters")               # Threshold sliders are used to change the intensity of noise in the frame
     threshold2 = cv2.getTrackbarPos("2", "Parameters")

     imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                # Converts frames to grayscale images
     imgCanny = cv2.Canny(imgGray, threshold1, threshold2)            # Takes the grayscale images and uses Canny Edge Detector to find and project edges
     imgDil = cv2.dilate(imgCanny, kernel, 1)                         # Dilates the projected edges 

     count = getContours(imgDil, imgContour, count)                   # Calls getContours function and assigns shape count

     cv2.putText(imgContour, "Number of Shapes: " + str(count), (20, 460), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 1)

     imgStack = stackImages(0.9, ([frame, imgDil, imgContour]))       # Calls stackImages
     cv2.imshow('Camera', imgStack)                                   # Show only the stacked images called in previous function

     if cv2.waitKey(1) & 0xFF == 27:                                  # Press 'Esc' key to close video capture
          break

cap.release()  
cv2.destroyAllWindows()                                               # Exit program