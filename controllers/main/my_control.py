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

IMG_RESOLUTION = 0.01
IMG_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / IMG_RESOLUTION)
IMG_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / IMG_RESOLUTION)

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
cv2.resizeWindow("map", IMG_SIZE_X, IMG_SIZE_Y)


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
    vel_world = get_velocity_command(sensor_data, occupancy_map, target)
    # Velocity command in local frame
    vel_local = rotate(vel_world, -sensor_data["yaw"])
    control_command[:2] = vel_local

    if t % 10 == 0:
        map_image = create_image(occupancy_map, sensor_data)
        cv2.imshow("map", map_image)
        cv2.waitKey(1)
    t += 1

    # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]
    return control_command


def create_image(occupancy_map: np.ndarray, sensor_data: dict) -> np.ndarray:
    map_grayscale = np.clip((occupancy_map + 1.0) * 0.5 * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
    img = cv2.cvtColor(
        cv2.resize(
            map_grayscale,
            dsize=(IMG_SIZE_X, IMG_SIZE_Y),
            interpolation=cv2.INTER_NEAREST,
        ),
        cv2.COLOR_GRAY2BGR,
    )

    pos_global = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    yaw = sensor_data["yaw"]
    pos_to_tip = np.array([np.cos(yaw), np.sin(yaw)]) * 0.08
    tip_global = pos_global + pos_to_tip
    pos_to_left = np.array([pos_to_tip[1], -pos_to_tip[0]]) * 0.3
    left_global = pos_global + pos_to_left
    right_global = pos_global - pos_to_left
    pts = global_to_img(
        np.array(
            [
                tip_global,
                left_global,
                right_global,
            ]
        )
    )
    cv2.polylines(
        img,
        pts=[pts.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2,
    )

    return np.flip(img, axis=0)


def get_velocity_command(sensor_data, occupancy_map, target) -> np.ndarray:
    # In (y, x) map indices
    obstacles = np.argwhere(occupancy_map <= -0.2)
    # In (x, y) global frame
    obstacles = map_to_global(obstacles)
    pos_global = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    # In (x, y) global frame
    vel_attractive = attraction(pos_global, target)
    vel_repulsive = repulsion(pos_global, obstacles)
    vel = np.clip((vel_attractive + vel_repulsive).squeeze(), -0.3, 0.3)
    return vel


def attraction(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Generates an attractive velocity to the target.
    The velocity profile is linear near the target and constant further away:
        ||v(r)|| = v0 * r / r0    if r < r0
        ||v(r)|| = v0             if r >= r0

    Arguments:
        pos: (2,) position of the drone in world space
        target: (2,) position of the target in world space
    """

    ATTENUATION_RADIUS = 0.2  # r0 [m/s]
    MAX_ATTRACTION_VELOCITY = 0.3  # v0 [m/s]
    EPSILON = 0.001  # [m]

    target_rel = target - pos
    distance = np.linalg.norm(target_rel)
    if distance >= ATTENUATION_RADIUS:
        return MAX_ATTRACTION_VELOCITY * target_rel / distance
    elif distance >= EPSILON:
        return MAX_ATTRACTION_VELOCITY * target_rel / ATTENUATION_RADIUS
    else:
        return np.zeros_like(target_rel)


def repulsion(pos: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """
    Generates a repulsive velocity from obstacles and map borders.
    The velocity profile is linear:
        ||v(r)|| = v0 / r0 * (r0 - r)    if r < r0
        ||v(r)|| = 0                     if r >= r0

    Arguments:
        pos: (2,) position of the drone in world space
        obstacles: (N, 2) positions of obstacles in world space
    """
    # FIXME: we need to add repulsive walls !!
    # FIXME: implement local minima avoidance from paper
    MAX_REPULSION_VELOCITY = 0.2  # v0 [m/s]
    INFLUENCE_RADIUS = 0.4  # r0 [m]
    EPSILON = 0.001  # [m]

    obstacles_rel = obstacles.reshape((-1, 2)) - pos.reshape((1, 2))
    distances = np.linalg.norm(obstacles_rel, axis=1)
    close_indices = (distances >= EPSILON) & (distances < INFLUENCE_RADIUS)
    close_distances = distances[close_indices].reshape((-1, 1))
    close_obstacles = obstacles_rel[close_indices, :]
    obstacle_repulsions = (
        MAX_REPULSION_VELOCITY
        / INFLUENCE_RADIUS
        * (close_distances - INFLUENCE_RADIUS)
        * close_obstacles
        / close_distances
    )

    wall_repulsion_x_min = (
        MAX_REPULSION_VELOCITY
        / INFLUENCE_RADIUS
        * (MAP_X_MIN + INFLUENCE_RADIUS - pos[0])
    )
    wall_repulsion_x_max = (
        -MAX_REPULSION_VELOCITY
        / INFLUENCE_RADIUS
        * (MAP_X_MAX + INFLUENCE_RADIUS - pos[0])
    )
    wall_repulsion_y_min = (
        MAX_REPULSION_VELOCITY
        / INFLUENCE_RADIUS
        * (MAP_Y_MIN + INFLUENCE_RADIUS - pos[1])
    )
    wall_repulsion_y_max = (
        -MAX_REPULSION_VELOCITY
        / INFLUENCE_RADIUS
        * (MAP_Y_MAX + INFLUENCE_RADIUS - pos[1])
    )
    # FIXME
    wall_repulsion = np.array(
        [
            [wall_repulsion_x_min, 0.0],
            [wall_repulsion_x_max, 0.0],
            [0.0, wall_repulsion_y_min],
            [0.0, wall_repulsion_y_max],
            [0.0, 0.0],
        ]
    ).max(axis=0)

    return np.sum(obstacle_repulsions, axis=0) + wall_repulsion


def map_to_global(indices: np.ndarray) -> np.ndarray:
    """
    (Y, X) map frame -> (X, Y) global frame
    """
    offset = np.array([MAP_Y_MIN, MAP_X_MIN]).reshape((1, 2))
    return np.flip(
        (indices.reshape((-1, 2)).astype(np.float32) + 0.5) * MAP_RESOLUTION + offset,
        axis=1,
    ).reshape(indices.shape)


def global_to_map(pts: np.ndarray) -> np.ndarray:
    """
    (X, Y) global frame -> (Y, X) map frame
    """
    offset = np.array([MAP_Y_MIN, MAP_X_MIN]).reshape((1, 2))
    return (
        (np.floor((np.flip(pts.reshape((-1, 2)), axis=1) - offset) / MAP_RESOLUTION))
        .astype(np.int32)
        .reshape((pts.shape))
    )


def global_to_img(pts: np.ndarray) -> np.ndarray:
    """
    (X, Y) global frame -> (X, Y) image frame
    """
    offset = np.array([MAP_Y_MIN, MAP_X_MIN]).reshape((1, 2))
    return (
        (np.floor((pts.reshape((-1, 2)) - offset) / IMG_RESOLUTION))
        .astype(np.int32)
        .reshape(pts.shape)
    )


def is_index_valid_map(row: int, col: int) -> bool:
    return (row >= 0) and (row < MAP_SIZE_Y) and (col >= 0) and (col < MAP_SIZE_X)


def update_occupancy_map(sensor_data: dict) -> np.ndarray:
    global occupancy_map

    x_global = sensor_data["x_global"]
    y_global = sensor_data["y_global"]
    xy_global = np.array([x_global, y_global])
    yaw = sensor_data["yaw"]

    for j in range(4):
        yaw_sensor = yaw + j * np.pi * 0.5
        yaw_direction = np.array([np.cos(yaw_sensor), np.sin(yaw_sensor)])
        sensor = "range_" + ["front", "left", "back", "right"][j]
        measurement = sensor_data[sensor]

        for i in range(int(SENSOR_RANGE_MAX / MAP_RESOLUTION)):
            distance = i * MAP_RESOLUTION
            index = global_to_map(xy_global + distance * yaw_direction)

            if distance < measurement:
                if is_index_valid_map(row=index[0], col=index[1]):
                    occupancy_map[index[0], index[1]] += CONFIDENCE
            else:
                if is_index_valid_map(row=index[0], col=index[1]):
                    occupancy_map[index[0], index[1]] -= CONFIDENCE
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
    return (
        np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        @ vec
    )
