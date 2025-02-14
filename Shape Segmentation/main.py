import cv2
import os
import time
import shapely
import threading
import numpy as np
from shapely import Polygon
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Create list objects to hold x and y coordinate for center point
pointX = []
pointY = []
point = []

def pointExtraction(coords, frame):
    max = len(coords)                               # Get max num of objects to create

    if max == 0 or max > 1:
        print("----No object detected or too many objects----")
        return

    matrix = [[] for _ in range(max)]               # Create empty list with max size
    i = -1
    for x in coords:
        i = i + 1
        for y in x:
            for z in y:
                matrix[i].append((z[0],z[1]))       # Append xy coord to matrix list

    for i in range(max):
        polygon = Polygon(matrix[i])                                # Convert array of coordinate points to a polygon
        simple = polygon.simplify(8, preserve_topology=False)       # Simplify polygon shape

        centerX = int(simple.centroid.x)                            # Extract xy coordinates for center point
        centerY = int(simple.centroid.y)
        center = (centerX, centerY)
        cv2.circle(frame, center, 5, (255, 0, 0), -2)               # Draw center point
    
    pointX.append(centerX)    
    pointY.append(centerY)                          # Append x and y coordinate to list
    point.append(center)

# Call segmentation model
model = YOLO("Shape Segmentation\shape-segv4.pt")  

# Var to hold center point to send out to Robot Arm
myCenter = 0

# Set capture resolution
frameWidth = 640                  
frameHeight = 480

cap = cv2.VideoCapture(0)               # Parameter represents an index for video camera to use for capture (Default 0)
cap.set(3, frameWidth)                  # Set video capture width
cap.set(4, frameHeight)                 # Set video capture height
cap.set(cv2.CAP_PROP_POS_FRAMES, 10)

# Output run to video
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('instance-segmentation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Main Program
while(cap.isOpened()):                  # Open video capture
    _, frame = cap.read()             # Read frame

    # Throw error
    if not _:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Apply smoothing to frame
    blur = cv2.GaussianBlur(frame, (5,5), 0)           

    # Call YOLO prediction
    results = model.predict(blur, conf=.2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        
        # Draw masks onto frame
        for mask in masks:
            cv2.polylines(frame, [np.int32([mask])], isClosed=True, color=(0, 255, 0), thickness=2)

    # Convert the BGR image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set the lower and upper bounds for green hue
    lower_green = np.array([50,100,50])
    upper_green = np.array([70,255,255])

    # Create a mask for green color using inRange function
    color = cv2.inRange(hsv, lower_green, upper_green)

    # Perform bitwise and on the original image arrays using the mask
    extract = cv2.bitwise_and(frame, frame, mask=color)
    extract = cv2.cvtColor(extract, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(extract, 0, 255, cv2.THRESH_BINARY)[1]

    # Extract coordinate points from mask into object
    coords, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assign center point extraction function to another thread
    t1 = threading.Thread(target=pointExtraction(coords, frame))

    t1.start()                                      # Start new thread
    t1.join()                                       # When main program continues after thread has finished

    # Begin current center point and previous center point comparison
    if len(pointX) == 2 and len(pointY) == 2:
        # Extract x1, x2, y1, y2
        x1 = pointX[0]
        x2 = pointX[1]
        y1 = pointY[0]
        y2 = pointY[1]

        # Find difference
        r1 = x1 - x2
        r2 = y1 - y2

        # Define insignificance range between -1.5 and 1.5
        if (r1 > -1.5 and r1 < 1.5) and (r2 > -1.5 and r2 < 1.5):
            # Difference between two points IS NOT significant enough to conclude object is moving
            print("----Motion not detected, retreiving center----")
            myCenter = point[0]                                         # Assign center point to be sent to Robot Arm
            print(myCenter)

            print("----PROGRAM FALLING ASLEEP TO WAIT FOR ARM MOTION----")
            time.sleep(5)                                               # Program must sleep for duration to let arm pick up
                                                                        # and move object before sending another center point
            pointX.clear()
            pointY.clear()                                              # Clear lists for new center point comparison
            point.clear()
        else:
            # Difference between two points IS significant enough to conclude object is moving
            print("----Motion detected, cannot retrieve center----")
            time.sleep(1.5)                                             # Program must sleep for duration to let object continue moving
                
            pointX.clear()
            pointY.clear()                                              # Clear lists for new center point comparison
            point.clear()


    out.write(frame)                                    # Write frame to video format
    cv2.imshow("instance-segmentation", frame)          # Show frame

    # When user wants to exit program, hit 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        print("----Quitting program----")
        break

out.release()
cap.release()
cv2.destroyAllWindows()