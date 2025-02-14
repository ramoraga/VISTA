# Date: 11/5/24
# Author: Dr Niechen Chen
# Most recent change by: Reinaldo

from database_module import *
from robotic_module import *
from camera_module import *
from math_util import *
import sys, signal

def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)
 
signal.signal(signal.SIGINT, signal_handler)

################### Initialization Phase ###################
if __name__ == "__main__":
    db = robodb("robot_assembly.db")
    db.connect()

db.delete_all_data()                    # Delete previous data from DetectedShapes

robot_1_port_name = '/dev/ttyUSB0'      # Initialize Robot object
robot1 = Robot(robot_1_port_name)
Cam1 = Robot_Cam()                      # Initialize Camera object

################### Calibration Phase ###################
#robot1.robo_home()
robot1.move_aside()
 
Cam1.load_CV_model("shape-segv1.pt")    # Load Trained YOLO model

'''fetched_matrix = [[-0.0046034, 0.41117, 105.76],
                  [0.39578, 0.0056358, -123.78],
                  [0, 0, 1]]'''
 
Cam1.detect_CalibrationPonts()          # Call for calibration point detection
time.sleep(1)

cam_cal_pts = Cam1.get_CalibrationPonts()           # Get camera calibration points & robot calibration points
print("Calibration Points: ", cam_cal_pts)
time.sleep(2)
print("Robot Coordinates: ", robot1.rcoords)
time.sleep(2) 

robot1.setRobotCalibrationMatrix(find_calibration_matrix(cam_cal_pts, robot1.rcoords))       # Calculate the calibration matrix 
db.insert_calibration_matrix(robot1.calibrationMaxtrix)                                      # and store to database

fetched_matrix = db.get_calibration_matrix()                    # Save matrix for future use
fetched_matrix = np.array(fetched_matrix).reshape(3, 3)
print("Calibration Matrix:\n", fetched_matrix)
time.sleep(2)

################### Part Detection Phase ###################
Cam1.detect_ObjectCamCoords()
time.sleep(1)

my_detections = Cam1.get_ObjectCamCoords()      # Get detection results
print("\nmy detections:")
for i in my_detections:
    print(i)

for i in my_detections:                 # Insert parts list into database
    db.insert_part_list(i)
a = db.get_part_list_all()
db.insert_parts_list()
time.sleep(2)

Cam1.generate_TaskList()                # Generate task list from Part List
my_tasks = Cam1.get_TaskList()
for i in my_tasks:
    # rbcpart = np.round(transform_point(fetched_matrix,i[0]))
    # rbcpart = rbcpart.tolist()
    # rbcdest = np.round(transform_point(fetched_matrix,i[1]))
    # rbcdest = rbcdest.tolist()
    task = {"part center":i[0], "dest center":i[1], "part corners": i[2:]} #, "RPart":rbcpart, "Rdest":rbcdest
    db.insert_task_list(task)
db.update_part_id()
a = db.get_task_list_all()

print("my tasks:")
for i in my_tasks:
    print(i)
time.sleep(2)

################### Main Loop ###################
# Hold previous frame for comparison
prev_frame = None
camera_stable = False
task_number = 1

# While the task list is not empty
while(my_tasks):

    # Save variables from the task at top of list
    obj_center, dest_center, obj_corners = my_tasks[0]

    # Extract object name from output list
    for item in my_detections:
        if item['dest_center'] == dest_center:
            obj_name = item['name']
            hsv = item['hsv']
            dest_corners = (list(item['dest_corners']))

    # Convert center points from 2D to 3D
    obj_x, obj_y = obj_center
    dest_x, dest_y = dest_center

    R_obj_center, R_obj_corners, source_points = convertPoints(obj_x, obj_y, obj_corners, fetched_matrix)
    R_dest_center, R_dest_corners, target_points = convertPoints(dest_x, dest_y, dest_corners, fetched_matrix)
    rotation_matrix, euler_angles, rotation_angle = find_rotationAngle(source_points, target_points)

    # Print task number
    #**********************in future will get from TasksList table**********
    print("Task", task_number, ": ", my_tasks[0])
    time.sleep(2)

    robot1.pickNorient_arm_p2p(R_obj_center[0], R_obj_center[1], 52, 0,0,0, R_dest_center[0], R_dest_center[1], 75, 0,0,(rotation_angle)) ##for old block z is 57,  for new z is 69
    robot1.move_aside()

    R_obj_center = list(R_obj_center)
    R_dest_center = list(R_dest_center)
    log = {"Item Center": obj_center, "Dest Center": dest_center, "R Item ": R_obj_center, "R Dest ": R_dest_center ,"angle" :rotation_angle, "timestamp" : datetime.datetime.now().isoformat(sep=" ", timespec="seconds")}
    db.insert_history_log(log, task_number)
    time.sleep(1)

    #*****************add the picknplace information to HistoryLog table************************
        
    # Set the duration in seconds for how long the loop should run
    time_limit = 3  # 5 seconds
    start_time = time.time()  # Record the start time

    # When a camera is initated, the video stream starts
    Cam1.pipe = rs.pipeline() # Create a pipeline
    cfg = rs.config() # Create a config and configure the pipeline to stream
    cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)

    # Start pipeline
    Cam1.pipe.start(cfg)

    # Open video to check if object location and orientation match destination
    while(True):
        # Variables to hold attributes
        center = []
        coords = []
        corners = []
        x = []

        frames = Cam1.pipe.wait_for_frames()

        # Get color frame and depth frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Check if the camera is stable by comparing current frame with previous one
        if prev_frame is not None:
            camera_stable = find_motion(prev_frame, color_image)

        prev_frame = color_image.copy()

        # If motion detected, restart start time
        if not camera_stable:
            print("----Motion detected. Waiting...----")
            start_time = time.time()
            continue

        zoomed_frame = applyZoom(color_image)
        final_image = apply_colorMask(hsv, zoomed_frame)

        # Call YOLO prediction
        results = Cam1.CV_model.predict(final_image, conf=.3)
        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()                
            masks = results[0].masks.xy
                    
            for mask, cls in zip(masks, clss):
                # Draw mask onto frame
                cv2.polylines(final_image, [np.int32([mask])], isClosed=True, color=(0, 255, 0), thickness=2)

                # Save mask coordinates and object name
                coords.append(np.int32([mask]))
                
                
        # Call center extraction. We only need the center
        Cam1.pointExtraction(coords, final_image, x, x, center, corners)  

        # Show frame
        cv2.imshow("color-mask", final_image)   # For viewing color-mask
                
        # User input exit program
        if cv2.waitKey(1) & 0xFF == 27:
            print("----Quitting program----")
            my_tasks.pop(0)
            break  
                
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        if elapsed_time > time_limit:
            time.sleep(1)

            if len(coords) == 1:
                print("...Performing First Layer Color Check")
                time.sleep(2)
                obj_x, obj_y = center[0]
                x_diff = abs(obj_x - dest_x)
                y_diff = abs(obj_y - dest_y)

                R_obj_center, R_obj_corners, source_points = convertPoints(obj_x, obj_y, corners[0], fetched_matrix)
                rotation_matrix, euler_angles, rotation_angle = find_rotationAngle(source_points, target_points)

                value = (center[0], dest_center, corners[0])
                my_tasks[0] = value
                print("...Updating Task[0]")
                #db.update_task_list()
                time.sleep(1)
            else:
                print("...Performing Second Layer Registration Check")
                time.sleep(2)

                r_corners = []

                for i in corners:
                    x, y = (0, 0)
                    _, converted, _ = convertPoints(obj_x, obj_y, i, fetched_matrix)
                    r_corners.append(converted)

                
                for i in r_corners:
                    print(i)





                error_list = []
                match_list = []
                for i in range(len(r_corners)):
                    error = find_matchError(r_corners[i], R_dest_corners)
                    error_list.append(error)
                    match_list.append((center[i], corners[i], error))

                print(error_list)

                match_count = 0

                for i in error_list:
                    if i <= 250:
                        match_count += 1

                if match_count == 1:
                    match_list = sorted(match_list, key=lambda x: x[2])

                    center, corners, _ = match_list[0]
                    obj_x, obj_y = center
                    x_diff = abs(obj_x - dest_x)
                    y_diff = abs(obj_y - dest_y)

                    R_obj_center, R_obj_corners, source_points = convertPoints(obj_x, obj_y, corners, fetched_matrix)
                    rotation_matrix, euler_angles, rotation_angle = find_rotationAngle(source_points, target_points)

                    value = (center, dest_center, corners)
                    my_tasks[0] = value
                    print("...Updating Task[0]")
                    #db.update_task_list()
                    time.sleep(1)
                else:
                    print("...Performing Third Layer Center Check")
                    time.sleep(2)
                    for point in range(len(center)):
                        obj_x, obj_y = center[point]

                        x_diff = abs(obj_x - dest_x)
                        y_diff = abs(obj_y - dest_y)

                        if center[point] == obj_center or (x_diff < 30 and y_diff < 30):
                    # Update current task with new values
                    #***************do if for the TasksList table************************************

                            R_obj_center, R_obj_corners, source_points = convertPoints(obj_x, obj_y, corners[point], fetched_matrix)
                            rotation_matrix, euler_angles, rotation_angle = find_rotationAngle(source_points, target_points)

                            value = (center[point], dest_center, corners[point])
                            my_tasks[0] = value
                            print("...Updating Task[0]")
                            #db.update_task_list()
                            time.sleep(1)
            break 

    Cam1.pipe.stop()  # Stop RealSense pipeline
    cv2.destroyAllWindows()

    # Extract new values from task list
    #***************Update the entry in History Log with Error*******************************
    updated_center, _, _= my_tasks[0]

    #obj_x, obj_y = updated_center

    # Find difference between current center and orientation with destination
    #x_diff = abs(obj_x - dest_x)
    #y_diff = abs(obj_y - dest_y)

    # If the difference is below a threshold
    #********compare the current task info with destination info saved in the HistoryLog table (two get statements here)

    if x_diff < 5 and y_diff < 5 and abs(rotation_angle) < 3:
        print("...Popping Task[0] from Task List")
        my_tasks.pop(0)
        time.sleep(2)
        # Iterate task number
        task_number += 1
    else:
        print("...Part does not fit Destination")
        time.sleep(1)

    print("...Quality Check Complete")
    time.sleep(1)

# Once all tasks are complete
print("***All Tasks Complete!***")
time.sleep(2)

#'''


'''while True:
    Cam1.detect_ObjectCamCoords()
    CP = Cam1.get_ObjectCamCoords()
    # populate parts list and task list

    #execute robot operation
    angle = math.atan(R[0,1]/R[0,0])*(180/math.pi)
    robot1.pickNorient_arm(cent_C[0],cent_C[1],52,0,0,0,Tr[0],Tr[1],56,0,0,angle) ##for old block z is 52,  for new z is 64
    robot1.move_aside()

    #plot graph   
    #plot_3d(C, D, B_transformed)



    for i in CP:
        db.insert_parts_list(i)
 
    CPSrc = []
    CPTgt = []
 
    fetched_CP = db.get_parts_list_all()
    for i in fetched_CP:
        if i['color'] == 'gray':
            CPTgt.append(i)
        else:
            CPSrc.append(i)

    #future code for multiple shapes

    TShape = []
    RShape = []
    TShapeT = []
    RShapeT = []
 
    for i in CPSrc:
        if len(i['corners']) > 5:
            a = db.extract_CP(i)
            TShape.append(a)
        else:
            b = db.extract_CP(i)
            RShape.append(b)
 
    for i in CPTgt:
        if len(i['corners']) > 5:
            c = db.extract_CP(i)
            TShapeT.append(c)
        else:
            d = db.extract_CP(i)
            RShapeT.append(d)

    #defining the points A and B for registration
    Aa = RShape
    Bb = RShapeT

    A = Aa[0]
    B = Bb[0]

    # A = np.array(A)
    # B = np.array(B)

    # A_3d = np.hstack((A, np.zeros((A.shape[0], 1), dtype=np.float32)))
    # B_3d = np.hstack((B, np.zeros((B.shape[0], 1), dtype=np.float32)))

    #converting camera coordinates to robot coordinates
    C = []
    D = []


    for i in A:
        transformed_point = transform_point(fetched_matrix, i)
        C.append((int(round(transformed_point[0])), int(round(transformed_point[1])), 0))
    for i in B:
        transformed_point = transform_point(fetched_matrix, i)
        D.append((int(round(transformed_point[0])), int(round(transformed_point[1])), 0))
 
    C = np.array(C)
    cent_C = find_centroid(C)
    D = np.array(D)
 
    #Apply rigid registration to align A to B
    registration = RigidRegistration(X=D, Y=C)
    transform_output = registration.register()
    s,R,t = registration.get_registration_parameters()
    R = registration.R
    t = registration.t
    B_transformed = registration.TY




 
    '''



