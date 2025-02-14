import cv2
import numpy as np

fontSize = 0.7                # Define size of text
fontThick = 1                 # Define thickness of text

def empty(a):
      pass

# The purpose of this function is to take an array of images and to stack them next
# to each other so that to project them at the same time

# The function takes two parameters (scale, imgArray)
# scale is a variable that represents the scale of the final stacked image
# imgArray is an array of images to be stacked

def stackImages(scale, imgArray):
     rows = len(imgArray)
     cols = len(imgArray[0])
     rowsAvailable = isinstance(imgArray[0], list)
     width = imgArray[0][0].shape[1]
     height = imgArray[0][0].shape[0]

     if rowsAvailable:
          for x in range (0, rows):
               for y in range (0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                         imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None, scale, scale)
                    else:
                         imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    
                    if len(imgArray[x][y].shape) == 2:
                         imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

          imageBlank = np.zeros((height, width, 3), np.uint8)
          hor = [imageBlank] * rows
          hor_con = [imageBlank] * rows

          for x in range(0, rows):
               hor[x] = np.hstack(imgArray[x])
          
          ver = np.vstack(hor)
     else:
          for x in range(0, rows):
               if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
               else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)

               if len(imgArray[x].shape) == 2:
                    imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
          
          hor = np.hstack(imgArray)
          ver = hor

     return ver



# The purpose of this function is to identify a shape by projecting its contour,
# name, and the total number of shapes detected

# The function takes three parameters (img, imgContour, count)
# img represents the image to be utilized for projecting
# imgContour represents the final image with contours, names, and shape count
# count is a variable that holds the shape count

def getContours(img, imgContour, count):
      
     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

     for cnt in contours:
          area = cv2.contourArea(cnt)
          areaMin = cv2.getTrackbarPos("Area", "Parameters")
          if area > areaMin:
               cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
               count = count + 1

               peri = cv2.arcLength(cnt, True)
               approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

               x, y, w, h = cv2.boundingRect(approx)

               # Hard code portion where shapes are determined based on number of edges drawn

               if len(approx) == 3:
                    cv2.putText(imgContour, "Triangle", (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0, 255, 0), fontThick)
               elif len(approx) == 4:
                    cv2.putText(imgContour, "Square", (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0, 255, 0), fontThick)
               elif len(approx) == 5:
                    cv2.putText(imgContour, "Pentagon", (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0, 255, 0), fontThick)
               elif len(approx) == 6:
                    cv2.putText(imgContour, "Hexagon", (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0, 255, 0), fontThick)  
               elif len(approx) == 8:
                    cv2.putText(imgContour, "Octagon", (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0, 255, 0), fontThick)
               elif len(approx) == 10:
                    cv2.putText(imgContour, "Star", (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0, 255, 0), fontThick)
               else:
                    cv2.putText(imgContour, "Circle", (x, y + 50), cv2.FONT_HERSHEY_COMPLEX, fontSize, (0, 255, 0), fontThick)            

     return count            