# Date: 09/24/24
# Author: Reinaldo Moraga
import cv2
import time
import math
import numpy as np
import pyrealsense2 as rs
from shapely import Polygon
from ultralytics import YOLO
from pycpd import RigidRegistration
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R


class Robot_Cam:
    "A class that is defined to contain all camera related functions and variables"
    def __init__(self):

        # Create list objects to hold x and y coordinate for center point
        self.x_center = []
        self.y_center = []
        self.obj_center = []
        self.obj_corner = []

        # Important variables that user will need
        self.CV_model = None
        self.obj_output = None
        self.cal_points = []
        self.task_list = []

        

        self.out1 = cv2.VideoWriter('boundary-calibration.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
        self.out2 = cv2.VideoWriter('instance-segmentation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))

        


    # This function resets camera hardware

    def resetHardware(self):

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()


    # This function loads a pre-trained model for segmentation

    def load_CV_model(self, model_path):

        self.CV_model = YOLO(model_path)  


    # This function takes a cordinate array to create a simplified polygon. From that polygon,
    # the center point is found, drawn to frame, and added to list objects

    def pointExtraction(self, coords, frame, xcord, ycord, center, corners):

        # Get max number of objects to create
        max = len(coords)    

        # If there are no objects...                 
        if max == 0:
            print("----No object detected----")
            return

        # Create empty list with size max
        matrix = [[] for _ in range(max)]  

        i = -1
        for x in coords:
            i = i + 1
            for y in x:
                for z in y:
                    # Append xy coord to list
                    matrix[i].append((z[0],z[1]))     

        for i in range(max):

            # Convert array of points to a polygon
            if len(matrix[i]) > 3:
                #print("Points in matrix[i]", matrix)
                polygon = Polygon(matrix[i])       

                # Take polygon and simplify shape                   
                simple = polygon.simplify(10, preserve_topology=False)    

                # Extract xy coord from center point of simplified shape
                if simple:
                    center_x = int(simple.centroid.x)                       
                    center_y = int(simple.centroid.y)
                    center_pt = (center_x, center_y)

                    # Draw center point to frame
                    cv2.circle(frame, center_pt, 3, (192, 203, 255), -1)  
                    cv2.putText(frame, str(center_pt), (center_x + 5, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) #Draw the text

                    if simple.geom_type == 'Polygon':
                        corner_pt = tuple(simple.exterior.coords)

                    smooth_corners = applySmoothing(corner_pt)

                    for c in smooth_corners:
                        cv2.circle(frame, vector(c), 3, (0, 0, 255), -1)
                        #cv2.putText(frame, str(vector(c)), vector(c), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA) #Draw the text

                    # Append xy coord to list objects       
                    xcord.append(center_x)    
                    ycord.append(center_y)                     
                    center.append(center_pt)
                    corners.append(smooth_corners)

    
    # This function creates a pipeline to open video capture, detect the star markers for calibration,
    # calls centerExtraction() to extract center points from star markers, detect if any movement is present,
    # and set the 4 calibration points

    def detect_CalibrationPonts(self):
        # Hold previous frame for comparison
        prev_frame = None
        camera_stable = False

        # Set the duration in seconds for how long the loop should run
        time_limit = 3  
        start_time = time.time()  # Record the start time

        # When a camera is initated, the video stream starts
        self.pipe = rs.pipeline() # Create a pipeline
        cfg = rs.config() # Create a config and configure the pipeline to stream
        cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)

        # Start pipeline
        self.pipe.start(cfg)

        # Open video capture
        while(True):     
            # Initialize holder objects for each frame
            coords = []                               
            x = []
            y = []
            center = []                                 
            corners = []

            frames = self.pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Check if the camera is stable by comparing current frame with previous one
            if prev_frame is not None:
                camera_stable = find_motion(prev_frame, color_image)

            prev_frame = color_image.copy()

            # If motion detected, print message and restart start time
            if not camera_stable:
                print("----Camera is not stable. Waiting...----")
                start_time = time.time()
                continue

            final_image = applyZoom(color_image)
            
            # Call YOLO prediction
            results = self.CV_model.predict(final_image, conf=.75, classes=[1])
            if results[0].masks is not None:
                masks = results[0].masks.xy
                    
                # Draw masks onto frame
                for mask in masks:
                    cv2.polylines(final_image, [np.int32([mask])], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Save mask coordinates and object name
                    coords.append(np.int32([mask]))

            # Call centerExtraction to get center of object
            self.pointExtraction(coords, final_image, x, y, center, corners)     

            # Write frame to video format
            self.out1.write(final_image)                

            # Show frame
            cv2.imshow("boundary-calibration", final_image)  
 
            if cv2.waitKey(1) & 0xFF == 27:
                print("----Quitting program----")
                break  
 
            # If there are not 4 objects detected, restart start time
            if len(center) != 4:
                start_time = time.time()

            # Calculate how much time has passed
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                # Create list with object names, color, center, corners from last frame
                cv2.imwrite('cal_image.png', final_image)
                list = [None] * len(coords)
                for r in range(len(coords)):
                    list[r] = center[r]

                list = sorted(list, key=lambda x: x[0])

                self.cal_points = list
                break          

        self.out2.release()
        self.pipe.stop()  # Stop RealSense pipeline
        cv2.destroyAllWindows()
        
        
    # This function creates a pipeline to open video capture, detects an object,
    # calls pointExtraction() to extract attributes of the detected object, detect if any movement is present,
    # and save information of each object to a dictionary

    def detect_ObjectCamCoords(self):
        # Hold previous frame for comparison
        prev_frame = None
        camera_stable = False

        # Set the duration in seconds for how long the loop should run
        time_limit = 3  
        start_time = time.time()  # Record the start time

        # When a camera is initated, the video stream starts
        self.pipe = rs.pipeline() # Create a pipeline
        cfg = rs.config() # Create a config and configure the pipeline to stream
        cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)

        # Start pipeline
        self.pipe.start(cfg)

        # Open video capture
        while(True):     
            # Initialize holder objects for each frame
            coords = []                               
            x = []
            y = []
            names = [] 
            color_code = []                                
            color_name = []
            depth = []
            center = []
            corners = []                         

            frames = self.pipe.wait_for_frames()

            # Get color frame and depth frame
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.5), cv2.COLORMAP_JET)

            # Check if the camera is stable by comparing current frame with previous one
            if prev_frame is not None:
                camera_stable = find_motion(prev_frame, color_image)

            prev_frame = color_image.copy()

            # If motion detected, print message and restart start time
            if not camera_stable:
                print("----Camera is not stable. Waiting...----")
                start_time = time.time()
                continue
            
            final_image = applyZoom(color_image)
            
            # Call YOLO prediction
            results = self.CV_model.predict(final_image, conf=.3, classes=[0,2])

            # Commence information extraction only if there are detections
            if results[0].masks is not None:
                clss = results[0].boxes.cls.cpu().tolist()                
                masks = results[0].masks.xy
                    
                for mask, cls in zip(masks, clss):
                    # Draw mask onto frame
                    cv2.polylines(final_image, [np.int32([mask])], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Save mask coordinates and object name
                    coords.append(np.int32([mask]))
                    names.append(self.CV_model.names[int(cls)])

                    # Create a mask for the current object
                    mask_img = np.zeros(final_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_img, [np.int32([mask])], 1)

                    # Extract the color information using mask and save
                    mean_color = cv2.mean(final_image, mask=mask_img)[:3]
                    mean_color = (round(mean_color[2]), round(mean_color[1]), round(mean_color[0]))
                    hsv_color = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_RGB2HSV)[0][0]

                    # Save information to our lists
                    color = find_color(mean_color)
                    color_code.append(hsv_color.tolist())
                    color_name.append(color)

                    # Extract depth information using mask and save
                    masked_depth = cv2.bitwise_and(depth_cm, depth_cm, mask=mask_img)
                    mean_depth = cv2.mean(masked_depth, mask=mask_img)[0]
                    depth.append(round(mean_depth))

            # Call centerExtraction to get center of object
            self.pointExtraction(coords, final_image, x, y, center, corners)   

            # Write frame to video format
            self.out2.write(color_image)                

            # Show frame
            cv2.imshow("instance-segmentation", final_image)   
            #cv2.imshow("test", depth_cm)
 
            if cv2.waitKey(1) & 0xFF == 27:
                print("----Quitting program----")

                # Save last frame for notes
                cv2.imwrite('quit_image_c.png', color_image)
                cv2.imwrite('quit_image_d.png', depth_cm)
                break   
            
            # If objects detected is less than or equal to 1, restart start time
            if len(center) == 0:
                start_time = time.time()

            # Calculate how much time has passed
            elapsed_time = time.time() - start_time

            if elapsed_time > time_limit:                
                # Save last frame for notes
                cv2.imwrite('final_frame_c.png', final_image)
                cv2.imwrite('final_frame_d.png', depth_cm)

                # Create list with object names, color, center, corners from last frame
                detections_list = [None] * len(coords)
                for r in range(len(coords)):
                    # Assign of type object or destination to each detection
                    if color_name[r] == "gray":
                        detections_list[r] = {"name": names[r], "type": "destination", "hsv_code":color_code[r], "color": color_name[r], "center": center[r], "corners": corners[r]}
                    else:
                        detections_list[r] = {"name": names[r], "type": "part", "hsv_code":color_code[r], "color": color_name[r], "depth": depth[r], "center": center[r], "corners": corners[r]}

                match_list  = []

                for part in detections_list:
                    if part['type'] == "part":
                        # For each object, iterate through the list to find destinations
                        for dest in detections_list:
                            if dest['type'] == "destination" and part['name'] == dest['name']:
                                # Calculate residual error and area difference for the object-destination pair
                                error = find_matchError(part['corners'], dest['corners'])
                                #print(error)
                
                                # Store the object-destination pair, metrics, and combined metric
                                match_list.append((part['name'], part['hsv_code'], part['color'], part['depth'], part['center'], part['corners'], dest['center'], dest['corners'], error))

                # Sort the pairs by the combined metric in ascending order
                match_list = sorted(match_list, key=lambda x: x[8])
              
                # Set to track already matched objects and destinations
                matched_objects = set()
                matched_destinations = set()

                parts_list = []

                # Greedily assign the best matches based on lowest metric
                for name, hsv, color, depth, center, corners, d_center, d_corners, error in match_list:
                    if center not in matched_objects and d_center not in matched_destinations:
                        matched_objects.add(center)
                        matched_destinations.add(d_center)  
                        new_entry = {"name": name, "hsv": hsv, "color": color, "depth": depth, "center": center, "corners": corners,
                                        "dest_center": d_center, "dest_corners": d_corners}
                        parts_list.append(new_entry)

                # Save list for output
                self.obj_output = parts_list
                break          

        self.out2.release()
        self.pipe.stop()  # Stop RealSense pipeline
        cv2.destroyAllWindows()
        
        
    # This function generates a task list from the parts list by matching objects
    # to their corresponding destinations, and saving that information as a task

    def generate_TaskList(self):
        task_list = []
        parts_list = self.obj_output
        for item in parts_list:
            task_list.append((item['center'], item['dest_center'], item['corners']))

        self.task_list = task_list       


    # This function returns the calibration points

    def get_CalibrationPonts(self):

        return self.cal_points
    

    # This function returns the parts list

    def get_ObjectCamCoords(self):

        return self.obj_output
    
    
    # This function returns the task list

    def get_TaskList(self):

        return self.task_list


COLORS_LIST = {
    (0, 0, 0): 'black',
    (255, 255, 255): 'white',
    (255, 0, 0): 'red',
    (0, 255, 0): 'lime',
    (0, 0, 255): 'blue',
    (255, 255, 0): 'yellow',
    (0, 255, 255): 'aqua',
    (255, 0, 255): 'fuchsia',
    (192, 192, 192): 'light gray',
    (128, 128, 128): 'gray',
    (128, 0, 0): 'maroon',
    (128, 128, 0): 'mustard',
    (0, 128, 0): 'green',
    (0, 128, 128): 'teal',
    (255, 165, 0): 'gold',
    (255, 192, 203): 'bubblegum',
    (139, 69, 19): 'orange',
    (255, 105, 180): 'pink',
    (0, 0, 139): 'dark blue',
    (85, 107, 47): 'olive green',
    (160, 82, 45): 'sienna',
    (255, 50, 0): 'red orange',
    (75, 0, 130): 'indigo',
    (0, 250, 154): 'mint green',
    (72, 61, 139): 'purple',
    (135, 206, 250): 'baby blue',
    (210, 180, 140): 'beige',
    (240, 128, 128): 'pastel red',
    (70, 130, 180): 'sky blue',
    (102, 205, 170): 'pastel green',
    (255, 20, 147): 'hot pink',
    (16, 62, 90): 'cobalt blue',
    (104, 76, 22): 'gold brown'
}

def closest_color(requested_color):
    min_colors = {}

    # Calculate the Euclidean distance between the RGB values of the two colors
    for rgb_value, name in COLORS_LIST.items():
        rd = (rgb_value[0] - requested_color[0]) ** 2
        gd = (rgb_value[1] - requested_color[1]) ** 2
        bd = (rgb_value[2] - requested_color[2]) ** 2

        # Objective is to find lowest score
        min_colors[(rd + gd + bd)] = name
        
    return min_colors[min(min_colors.keys())]

def find_color(rgb_color):
    # Directly check if the RGB color is in our dictionary
    if rgb_color in COLORS_LIST:
        return COLORS_LIST[rgb_color]
    else:
        # If the exact color name is not found, find the closest color
        return closest_color(rgb_color)

def find_motion(prev_frame, current_frame, threshold=250000):
    # Convert frames to grayscale for simplicity
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two frames
    frame_diff = cv2.absdiff(prev_gray, current_gray)

    # Count the number of non-zero pixels in the difference image
    non_zero_count = np.count_nonzero(frame_diff)
    #print(non_zero_count)

    # Check if the number of changed pixels is below the threshold
    return non_zero_count < threshold

def find_area(corners):
    # Get x and y points from corners
    x = [pt[0] for pt in corners]
    y = [pt[1] for pt in corners]

    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def find_matchError(object_corners, destination_corners):
    # Convert corners to numpy arrays
    object_points = np.array(object_corners)
    destination_points = np.array(destination_corners)

    error = []

    for i in range(100):
    
        # Calculate areas
        object_area = find_area(object_corners)
        destination_area = find_area(destination_corners)
        
        # Calculate area difference
        area_difference = abs(object_area - destination_area)
        
        # Ensure both sets have the same number of points by using Nearest Neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(destination_points)
        _, indices = nbrs.kneighbors(object_points)
        
        # Match points by selecting the closest points in the destination
        matched_destination_points = destination_points[indices.flatten()]

        # Initialize the Rigid Registration
        reg = RigidRegistration(X=matched_destination_points, Y=object_points)
        
        # Perform registration to align object with destination
        transform, (s, R, t) = reg.register()
        
        # Apply the transformation to the object's points
        transformed_object_points = s * object_points.dot(R) + t
        
        # Calculate the difference (residual error) between the transformed object and the destination
        residual_error = np.linalg.norm(transformed_object_points - matched_destination_points, axis=1).mean()

        error.append(residual_error + area_difference)
    
    error.sort()
    return error[0]

def applyZoom(color_image):
    # Get the dimensions of the frame
    height, width, _ = color_image.shape
    zoom_factor = 1.3
    
    # Calculate the region of interest (ROI)
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2

    # Crop the image to the ROI
    zoomed_frame = color_image[start_y:start_y+new_height, start_x:start_x+new_width]
    zoomed_frame = cv2.resize(zoomed_frame, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_frame

def apply_colorMask(hsv, zoomed_frame):
    # Define the lower and upper bounds for the HSV color
    lower_bound = np.array([hsv[0] - 60, 50, 50])
    upper_bound = np.array([hsv[0] + 60, 255, 255])

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2HSV)

    # Create an inverted mask: change black (0) to white (255)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    inverted_mask = cv2.bitwise_not(mask)

    # Convert the mask to 3 channels to apply it to the original image
    color_mask = cv2.bitwise_and(zoomed_frame, zoomed_frame, mask=mask)

    # Create a white background with the same size as the original image
    white_background = np.full_like(zoomed_frame, 255)

    # Apply the inverted mask to the white background
    result = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

    # Add the colored mask to the white background to keep the detected color
    final_image = cv2.add(result, color_mask)

    return final_image

def find_rotationAngle(source_points, target_points):
    """
    Detects the orientation of an object using PyCPD.

    Parameters:
    - source_points: A (N x 3) numpy array of points from the detected object (e.g., center + corners)
    - target_points: A (N x 3) numpy array of reference points with known orientation (e.g., a standard shape)

    Returns:
    - rotation_matrix: A 3x3 numpy array representing the rotation matrix.
    - euler_angles: A numpy array of Euler angles (in degrees) representing the rotation in 'xyz' axes.
    - rotation_angle: The angle of rotation in degrees.
    """
    
    # Step 1: Set up the registration (source = detected object, target = reference object)
    reg = RigidRegistration(X=target_points, Y=source_points)
    
    # Step 2: Perform the registration
    transformed_source, transformation_params = reg.register()
    
    # Step 3: Extract the rotation matrix
    rotation_matrix = reg.R
    
    # Step 4: Convert the rotation matrix to Euler angles
    rotation_object = R.from_matrix(rotation_matrix)
    euler_angles = rotation_object.as_euler('xyz', degrees=True)  # Convert to degrees for easier interpretation
    
    # Step 5: Calculate the rotation angle from the rotation matrix
    rotation_angle = np.degrees(np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0]))

    return rotation_matrix, euler_angles, rotation_angle

def find_angle(p1, p2, p3):
    # Calculate vectors p1 -> p2 and p2 -> p3
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    
    # Calculate angle in radians and then convert to degrees
    if mag_v1 * mag_v2 == 0:  # Avoid division by zero
        return 0
    angle_rad = math.acos(dot_product / (mag_v1 * mag_v2))
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def applySmoothing(coords, min_angle=140, max_angle=200):
    # Remove the duplicate last point to avoid processing it twice
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    filtered_coords = []
    n = len(coords)

    for i in range(n):
        # Select the previous, current, and next points
        p1 = coords[i - 1] if i > 0 else coords[-1]  # Use the last point for the first point's previous
        p2 = coords[i]
        p3 = coords[(i + 1) % n]  # Wraps around to the first point if i is n-1

        # Calculate the angle at p2
        angle = find_angle(p1, p2, p3)

        
        # Keep the point if the angle is outside the specified range
        if not (min_angle <= angle <= max_angle):
            filtered_coords.append(p2)

    # Close the polygon by re-adding the first point as the last
    if filtered_coords[0] != filtered_coords[-1]:
        filtered_coords.append(filtered_coords[0])

    return filtered_coords

vector = np.vectorize(np.int_)

'''
#------------DEMO PROGRAM------------#

# Initiate Robot_cam object
test = Robot_Cam()

# Restart camera hardware
test.resetHardware

# Load model
test.load_CV_model("shape-segv1.pt")

# Detect calibration points
#test.detect_CalibrationPonts()

# Get calibration points
#cal_points = test.get_CalibrationPonts()

# Print calibration points
#print("calibration points: ")
#for i in cal_points:
#    print(i)

# Detect objects
test.detect_ObjectCamCoords()

# Get objects
my_object = test.get_ObjectCamCoords()

# Print objects detected
print("my object:")
for o in my_object:
    print(o)

# Generate Task List
test.generate_TaskList()

# Get task list
task_list = test.get_TaskList()

# Print Task List
print("Task List:")
for i in task_list:
    print(i)

# Place objects
test.place_Object()
'''