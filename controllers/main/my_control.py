# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    map = update_occupancy_grid(sensor_data)
    
    return control_command # [vx, vy, alt, yaw_rate]


# All lengths are in meters
MAP_X_MIN, MAP_X_MAX = 0.0, 5.0
MAP_Y_MIN, MAP_Y_MAX = 0.0, 3.0
MAP_RESOLUTION = 0.05
MAP_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / MAP_RESOLUTION)
MAP_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RESOLUTION)
SENSOR_RANGE_MAX = 2.0
CONFIDENCE = 0.2 # Certainty given by each measurement

def global_to_map(x: float, y: float) -> tuple[int, int]:
    return (int((x - MAP_X_MIN) / MAP_RESOLUTION),
            int((y - MAP_Y_MIN) / MAP_RESOLUTION))

# 0 = unknown, 1 = free, -1 = occupied
occupancy_grid = np.zeros((MAP_SIZE_Y, MAP_SIZE_X), dtype=np.float32)

t = 0 # Current timestep

cv2.namedWindow("map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("map", 500, 300)

def update_occupancy_grid(sensor_data):
    global occupancy_grid, t

    x_global = sensor_data['x_global']
    y_global = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
    # Measurements to add to the occupancy grid
    measurement_grid = np.zeros_like(occupancy_grid)

    line_start = global_to_map(x_global, y_global)

    print(sensor_data['range_front'], sensor_data['range_left'], sensor_data['range_back'], sensor_data['range_right'])

    for j in range(4):
        yaw_sensor = yaw + j * np.pi * 0.5
        sensor = ['range_front', 'range_left', 'range_back', 'range_right'][j]
        measurement = sensor_data[sensor]

        x_end_global = x_global + measurement * np.cos(yaw_sensor)
        y_end_global = y_global + measurement * np.sin(yaw_sensor)
        line_end = global_to_map(x_end_global, y_end_global)

        print(MAP_SIZE_X, MAP_SIZE_Y)
        print(line_start)
        print(line_end)
        exit()
        
        # Increase the confidence on the whole line, end-point included
        cv2.line(measurement_grid, line_start, line_end, color=CONFIDENCE)
        # Decrease twice the confidence on the end-point
        measurement_grid[line_end[1], line_end[0]] -= 2 * CONFIDENCE

    # The point at the drone location wrongly has 4x confidence
    measurement_grid[line_start[1], line_start[0]] -= 3 * CONFIDENCE

    occupancy_grid = np.clip(occupancy_grid + measurement_grid, -1.0, 1.0)

    # only plot every Nth time step (comment out if not needed)
    if t % 10 == 0:
        map_gray = np.array(np.clip((occupancy_grid + 1.0) * 0.5 * 255.0, 0.0, 255.0), dtype=np.uint8)
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
