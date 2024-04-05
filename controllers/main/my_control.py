# Examples of basic methods for simulation competition
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
All available ground truth measurements can be accessed by calling sensor_data[item], where "item" can take the following values:
"x_global": Global X position
"y_global": Global Y position
"z_global": Global Z position
"roll": Roll angle (rad)
"pitch": Pitch angle (rad)
"yaw": Yaw angle (rad)
"v_x": Global X velocity
"v_y": Global Y velocity
"v_z": Global Z velocity
"v_forward": Forward velocity (body frame)
"v_left": Leftward velocity (body frame)
"v_down": Downward velocity (body frame)
"ax_global": Global X acceleration
"ay_global": Global Y acceleration
"az_global": Global Z acceleration
"range_front": Front range finder distance
"range_down": Donward range finder distance
"range_left": Leftward range finder distance 
"range_back": Backward range finder distance
"range_right": Rightward range finder distance
"range_down": Downward range finder distance
"rate_roll": Roll rate (rad/s)
"rate_pitch": Pitch rate (rad/s)
"rate_yaw": Yaw rate (rad/s)
"""


# Global variables
on_ground = True
height_desired = 1.0
timer = None
start_pos = None
timer_done = None


# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, start_pos

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)
    
    # Take off
    if start_pos is None:
        start_pos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]    
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    
    control_command = [0.0, 0.0, height_desired, 1.0]
    on_ground = False
    map = update_occupancy_map(sensor_data)
    
    return control_command # [vx, vy, alt, yaw_rate]


# All lengths are in meters
MAP_X_MIN, MAP_X_MAX = 0.0, 5.0
MAP_Y_MIN, MAP_Y_MAX = 0.0, 3.0
MAP_RESOLUTION = 0.05
MAP_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / MAP_RESOLUTION)
MAP_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RESOLUTION)
SENSOR_RANGE_MAX = 2.0
CONFIDENCE = 0.2 # Certainty given by each measurement

def global_x_to_map(x: float) -> int:
    return int((x - MAP_X_MIN) / MAP_RESOLUTION)

def global_y_to_map(y: float) -> int:
    return int((y - MAP_Y_MIN) / MAP_RESOLUTION)

def is_index_x_valid(index: int) -> bool:
    return index >= 0 and index < MAP_SIZE_X

def is_index_y_valid(index: int) -> bool:
    return index >= 0 and index < MAP_SIZE_Y

# 0 = unknown, 1 = free, -1 = occupied
occupancy_map = np.zeros((MAP_SIZE_Y, MAP_SIZE_X), dtype=np.float32)

t = 0 # Current timestep

cv2.namedWindow("map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("map", 500, 300)

def update_occupancy_map(sensor_data):
    global occupancy_map, t

    x_global = sensor_data['x_global']
    y_global = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
    for j in range(4):
        yaw_sensor = yaw + j * np.pi * 0.5
        sensor = ['range_front', 'range_left', 'range_back', 'range_right'][j]
        measurement = sensor_data[sensor]

        for i in range(int(SENSOR_RANGE_MAX / MAP_RESOLUTION)):
            distance = i * MAP_RESOLUTION
            index_x = global_x_to_map(x_global + distance * np.cos(yaw_sensor))
            index_y = global_y_to_map(y_global + distance * np.sin(yaw_sensor))

            if distance < measurement:
                if is_index_x_valid(index_x) and is_index_y_valid(index_y):
                    occupancy_map[index_y, index_x] += CONFIDENCE
            else:
                if is_index_x_valid(index_x) and is_index_y_valid(index_y):
                    occupancy_map[index_y, index_x] -= CONFIDENCE
                break
        
    occupancy_map = np.clip(occupancy_map, -1.0, 1.0)

    # only plot every Nth time step (comment out if not needed)
    if t % 10 == 0:
        map_gray = np.array(np.clip((occupancy_map + 1.0) * 0.5 * 255.0, 0.0, 255.0), dtype=np.uint8)
        map_gray = np.flip(map_gray, axis=0)
        cv2.imshow("map", map_gray)
        cv2.waitKey(1)
    t += 1

    return map


# Control from the exercises
index_current_setpoint = 0
def path_to_setpoint(path,sensor_data,dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, start_pos

    # Take off
    if start_pos is None:
        start_pos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]    
    if on_ground and sensor_data['z_global'] < 0.49:
        current_setpoint = [start_pos[0], start_pos[1], height_desired, 0.0]
        return current_setpoint
    else:
        on_ground = False

    # Start timer
    if (index_current_setpoint == 1) & (timer is None):
        timer = 0
        print("Time recording started")
    if timer is not None:
        timer += dt
    # Hover at the final setpoint
    if index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [start_pos[0], start_pos[1], start_pos[2]-0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer,1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm([current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone, clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return current_setpoint

    return current_setpoint

def clip_angle(angle):
    angle = angle % (2.0 * np.pi)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    elif angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

