# Examples of basic methods for simulation competition
import cv2
import matplotlib.pyplot as plt
import numpy as np

# The available ground truth state measurements can be accessed by calling sensor_data[item].
# All values of "item" are provided as defined in main.py lines 296-323.
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "range_down": Downward range finder distance
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "yaw": Yaw angle (rad)

# In meters
MAP_X_MIN, MAP_X_MAX = 0.0, 5.0
MAP_Y_MIN, MAP_Y_MAX = 0.0, 3.0
MAP_RESOLUTION = 0.05
SENSOR_RANGE_MAX = 2.0
# In pixels
MAP_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / MAP_RESOLUTION)
MAP_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RESOLUTION)

CONFIDENCE = 0.2


# Global variables
on_ground = True
height_desired = 1.0
timer = None
start_pos = None
timer_done = None
t = 0

# 0 = unknown, 1 = free, -1 = occupied
occupancy_map = np.zeros((MAP_SIZE_Y, MAP_SIZE_X), dtype=np.float32)

cv2.namedWindow("map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("map", 500, 300)


# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, start_pos, t

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)

    # Take off
    if start_pos is None:
        start_pos = [
            sensor_data["x_global"],
            sensor_data["y_global"],
            sensor_data["range_down"],
        ]
    if on_ground and sensor_data["range_down"] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----

    control_command = [0.0, 0.0, height_desired, 2.0]
    target = np.array([4.8, 0.5])

    occupancy_map = update_occupancy_map(sensor_data)
    # Velocity command in world frame
    vel = get_velocity_command(sensor_data, occupancy_map, target)
    vel_frame = rotate(vel, -sensor_data["yaw"])
    control_command[:2] = vel_frame

    if t % 10 == 0:
        map_image = np.array(
            np.clip((occupancy_map + 1.0) * 0.5 * 255.0, 0.0, 255.0), dtype=np.uint8
        )
        map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
        x_map = x_global_to_map(sensor_data["x_global"])
        y_map = y_global_to_map(sensor_data["y_global"])
        if is_index_x_valid(x_map) and is_index_y_valid(y_map):
            map_image[y_map, x_map] = (255, 0, 0)
        map_image = np.flip(map_image, axis=0)
        cv2.imshow("map", map_image)
        cv2.waitKey(1)
    t += 1

    # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]
    return control_command


def get_velocity_command(sensor_data, occupancy_map, target) -> np.ndarray:
    # In (y, x) map indices
    obstacles = np.argwhere(occupancy_map <= -0.2)
    # In (y, x) global frame
    obstacles = map_to_global(obstacles)
    pos_global = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    # In (x, y) global frame
    vel_attractive = attraction(pos_global, target)
    vel_repulsive = repulsion(pos_global, np.flip(obstacles, axis=1))
    vel = np.clip((vel_attractive + vel_repulsive).squeeze(), -0.3, 0.3)
    return vel


def attraction(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    ATTENUATION_RADIUS = 0.2
    EPSILON = 0.001
    NOMINAL_VELOCITY = 0.3
    target_rel = target - pos
    distance = np.linalg.norm(target_rel)
    if distance >= ATTENUATION_RADIUS:
        return NOMINAL_VELOCITY * target_rel / distance
    elif distance >= EPSILON:
        return NOMINAL_VELOCITY * target_rel / ATTENUATION_RADIUS
    else:
        return np.zeros_like(target_rel)


def repulsion(pos: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    # 1/r repulsion:
    #  v = A*(r0-r)/(r-a)
    #  -A*r0/a = A0     A = -A0*a/r0
    #  v = A0/r0*a*(r-r0)/(r-a)    more "curvy" with `a` being a small negative value
    #
    # linear repulsion:
    #  v = v0/r0*(r0-r)
    #
    # FIXME: try to understand why it is unstable when obstacles appear suddenly
    REPULSION_STRENGTH = 0.2
    INFLUENCE_RADIUS = 0.4
    EPSILON = 0.001
    obstacles_rel = obstacles - pos.reshape((1, 2))
    distances = np.linalg.norm(obstacles_rel, axis=1)  # .reshape((-1, 1))
    close_indices = (distances >= EPSILON) & (distances < INFLUENCE_RADIUS)
    close_distances = distances[close_indices].reshape((-1, 1))
    close_obstacles = obstacles_rel[close_indices, :]
    repulsives = (
        REPULSION_STRENGTH
        / INFLUENCE_RADIUS
        * (close_distances - INFLUENCE_RADIUS)
        * close_obstacles
        / close_distances
    )
    return np.sum(repulsives, axis=0)


def x_global_to_map(x: float) -> int:
    return int(np.floor((x - MAP_X_MIN) / MAP_RESOLUTION))


def y_global_to_map(y: float) -> int:
    return int(np.floor((y - MAP_Y_MIN) / MAP_RESOLUTION))


def x_map_to_global(x: int) -> float:
    return (float(x) + 0.5) * MAP_RESOLUTION + MAP_X_MIN


def y_map_to_global(y: int) -> float:
    return (float(y) + 0.5) * MAP_RESOLUTION + MAP_Y_MIN


def map_to_global(coords: np.ndarray) -> np.ndarray:
    return (coords.astype(np.float32) + 0.5) * MAP_RESOLUTION + np.array(
        [MAP_Y_MIN, MAP_X_MIN]
    ).reshape((1, 2))


def is_index_x_valid(index: int) -> bool:
    return index >= 0 and index < MAP_SIZE_X


def is_index_y_valid(index: int) -> bool:
    return index >= 0 and index < MAP_SIZE_Y


def update_occupancy_map(sensor_data):
    global occupancy_map

    x_global = sensor_data["x_global"]
    y_global = sensor_data["y_global"]
    yaw = sensor_data["yaw"]

    for j in range(4):
        yaw_sensor = yaw + j * np.pi * 0.5
        sensor = "range_" + ["front", "left", "back", "right"][j]
        measurement = sensor_data[sensor]

        for i in range(int(SENSOR_RANGE_MAX / MAP_RESOLUTION)):
            distance = i * MAP_RESOLUTION
            index_x = x_global_to_map(x_global + distance * np.cos(yaw_sensor))
            index_y = y_global_to_map(y_global + distance * np.sin(yaw_sensor))

            if distance < measurement:
                if is_index_x_valid(index_x) and is_index_y_valid(index_y):
                    occupancy_map[index_y, index_x] += CONFIDENCE
            else:
                if is_index_x_valid(index_x) and is_index_y_valid(index_y):
                    occupancy_map[index_y, index_x] -= CONFIDENCE
                break

    occupancy_map = np.clip(occupancy_map, -1.0, 1.0)

    return occupancy_map


# --- Control from the exercises ---

index_current_setpoint = 0


def path_to_setpoint(path, sensor_data, dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, start_pos

    # Take off
    if start_pos is None:
        start_pos = [
            sensor_data["x_global"],
            sensor_data["y_global"],
            sensor_data["range_down"],
        ]
    if on_ground and sensor_data["range_down"] < 0.49:
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
        control_command = [start_pos[0], start_pos[1], start_pos[2] - 0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer, 1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = (
        sensor_data["x_global"],
        sensor_data["y_global"],
        sensor_data["range_down"],
        sensor_data["yaw"],
    )
    distance_drone_to_goal = np.linalg.norm(
        [
            current_setpoint[0] - x_drone,
            current_setpoint[1] - y_drone,
            current_setpoint[2] - z_drone,
            clip_angle(current_setpoint[3]) - clip_angle(yaw_drone),
        ]
    )

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

def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ vec