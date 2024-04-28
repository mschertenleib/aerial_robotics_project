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

    path = np.array(
        [
            [1.0, 0.5, 1.0, 0.0],
            [1.0, 2.5, 1.0, 0.0],
            [start_pos[0], start_pos[1], 1.0, 0.0],
        ]
    )
    # target = np.array([4.8, 0.5])
    setpoint = path_to_setpoint(path, sensor_data, dt)
    target = setpoint[:2]

    occupancy_map = update_occupancy_map(sensor_data)

    control_command = get_control_command(
        sensor_data=sensor_data, target=target, occupancy_map=occupancy_map
    )

    if t % 10 == 0:
        map_image = create_map_image(
            sensor_data=sensor_data,
            occupancy_map=occupancy_map,
        )
        cv2.imshow("map", map_image)
        cv2.waitKey(1)
    t += 1

    # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]
    return control_command


def create_map_image(
    sensor_data: dict,
    occupancy_map: np.ndarray,
) -> np.ndarray:
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
    DRONE_SIZE = 0.1  # [m]
    pos_to_tip = np.array([np.cos(yaw), np.sin(yaw)]) * DRONE_SIZE * 0.5
    left = np.array([-pos_to_tip[1], pos_to_tip[0]])
    tip_global = pos_global + pos_to_tip
    left_global = pos_global - pos_to_tip + left
    right_global = pos_global - pos_to_tip - left
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
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return np.flip(img, axis=0)


def get_control_command(
    sensor_data: dict, target: np.ndarray, occupancy_map: np.ndarray
) -> np.ndarray:
    CORRECTION_FACTOR = 1.0

    pos = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    yaw = sensor_data["yaw"]

    obstacles = map_to_global(np.argwhere(occupancy_map <= -0.2))

    vel_attractive = attraction_to_target(
        pos, target, min_radius=0.01, max_radius=0.2, max_value=0.2
    )
    vel_repulsive = repulsion(pos, obstacles)
    norm_attractive = np.linalg.norm(vel_attractive)
    norm_repulsive = np.linalg.norm(vel_repulsive)
    if norm_repulsive > 0.001 and norm_attractive > 0.001:
        cos_angle = np.dot(vel_attractive, vel_repulsive) / (
            norm_attractive * norm_repulsive
        )
        vel_corrective = (
            CORRECTION_FACTOR
            * cos_angle
            * np.array([-vel_repulsive[1], vel_repulsive[0]])
        )
    else:
        vel_corrective = np.zeros(2)
    vel = clip_norm(
        (vel_attractive + vel_repulsive + vel_corrective).squeeze(),
        max_norm=0.3,
        epsilon=0.001,
    )
    print(f"{vel_attractive} {vel_repulsive} {vel_corrective}")

    vel = rotate(vel, -yaw)
    control_command = [vel[0], vel[1], 1.0, 2.0]
    return control_command


def get_control_command_old(sensor_data: dict, target: np.ndarray) -> np.ndarray:
    """
    Returns the control command [v_forward, v_left, alt, yaw_rate]
    """

    MAX_ATTRACTION = 0.3
    ATTRACTION_ATTENUATION_RADIUS = 0.2
    FRONT_INFLUENCE_RADIUS = 0.3
    FRONT_MAX_REPULSION = 1.0
    SIDE_INFLUENCE_RADIUS = 0.15
    SIDE_MAX_REPULSION = 1.0
    MAX_VELOCITY = 0.3
    YAW_KP = 2.0
    MAX_YAW_RATE = 2.0

    pos = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    yaw = sensor_data["yaw"]
    range_front = sensor_data["range_front"]
    range_left = sensor_data["range_left"]
    range_right = sensor_data["range_right"]

    target_relative = target - pos
    target_distance = np.linalg.norm(target_relative)
    if target_distance < 0.01:
        return np.zeros(4)

    target_heading = np.arctan2(target_relative[1], target_relative[0])
    yaw_error = clip_angle(target_heading - yaw)
    yaw_rate = YAW_KP * yaw_error

    v_forward = np.clip(
        MAX_ATTRACTION / ATTRACTION_ATTENUATION_RADIUS * target_distance,
        -MAX_ATTRACTION,
        MAX_ATTRACTION,
    ) - linear_repulsion(
        distance=range_front,
        radius=FRONT_INFLUENCE_RADIUS,
        max_value=FRONT_MAX_REPULSION,
    )

    v_left = (
        -linear_repulsion(
            distance=range_left,
            radius=SIDE_INFLUENCE_RADIUS,
            max_value=SIDE_MAX_REPULSION,
        )
        + linear_repulsion(
            distance=range_right,
            radius=SIDE_INFLUENCE_RADIUS,
            max_value=SIDE_MAX_REPULSION,
        )
        + linear_repulsion(
            distance=range_front,
            radius=FRONT_INFLUENCE_RADIUS,
            max_value=FRONT_MAX_REPULSION,
        )
        * get_prefered_repulsion_side(pos, yaw)
    )

    v_forward_cmd = np.clip(v_forward, -MAX_VELOCITY, MAX_VELOCITY)
    v_left_cmd = np.clip(v_left, -MAX_VELOCITY, MAX_VELOCITY)
    alt_cmd = 1.0
    yaw_rate_cmd = np.clip(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE)

    print(f"{v_forward_cmd:7.4f} {v_left_cmd:7.4f} {alt_cmd:7.4f} {yaw_rate_cmd:7.4f}")

    return [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]


def attraction_to_target(
    pos: np.ndarray,
    target: np.ndarray,
    min_radius: float,
    max_radius: float,
    max_value: float,
) -> np.ndarray:
    """
    Generates an attractive velocity to the target, in world frame.
    The velocity profile is linear near the target and constant further away:
        ||v(r)|| = v0 * r / r0    if r < r0
        ||v(r)|| = v0             if r >= r0

    Arguments:
        pos: position of the drone in world space
        target: position of the target in world space
    """
    # FIXME: min_radius
    return clip_norm(
        max_value / max_radius * (target - pos),
        max_norm=max_value,
        epsilon=0.001,
    )


def linear_repulsion(distance: float, radius: float, max_value: float) -> float:
    return max(max_value / radius * (radius - distance), 0.0)


def get_prefered_repulsion_side(pos: np.ndarray, yaw: float) -> float:
    # FIXME
    return 1.0


def distance_to_wall(pos: np.ndarray, dir: np.ndarray) -> float:
    if np.abs(dir[0]) > 0.0001:
        d_x_min = (MAP_X_MIN - pos[0]) / dir[0]
        d_x_max = (MAP_X_MAX - pos[0]) / dir[0]
    else:
        d_x_min, d_x_max = np.inf, np.inf
    if np.abs(dir[1]) > 0.0001:
        d_y_min = (MAP_Y_MIN - pos[1]) / dir[1]
        d_y_max = (MAP_Y_MAX - pos[1]) / dir[1]
    else:
        d_y_min, d_y_max = np.inf, np.inf
    distances = np.array([d_x_min, d_x_max, d_y_min, d_y_max])
    return np.min(distances[distances > 0.0])


def repulsion(pos: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """
    Generates a repulsive velocity from obstacles and map borders, in world frame.
    The velocity profile is linear:
        ||v(r)|| = v0 / r0 * (r0 - r)    if r < r0
        ||v(r)|| = 0                     if r >= r0

    Arguments:
        pos: position(s) of the drone in world space
        obstacles: positions of obstacles in world space
    """

    OBSTACLE_MAX_REPULSION = 0.2
    OBSTACLE_INFLUENCE_RADIUS = 0.2
    BORDER_MAX_REPULSION = 0.5
    BORDER_INFLUENCE_RADIUS = 0.2
    EPSILON = 0.001

    # Repulsion from obstacles
    obstacles_rel = obstacles.reshape((1, -1, 2)) - pos.reshape((-1, 1, 2))
    distances = np.linalg.norm(obstacles_rel, axis=-1, keepdims=1)
    close_mask = (distances >= EPSILON) & (distances < OBSTACLE_INFLUENCE_RADIUS)
    obstacle_repulsion = np.sum(
        np.where(
            close_mask,
            OBSTACLE_MAX_REPULSION
            / OBSTACLE_INFLUENCE_RADIUS
            * (distances - OBSTACLE_INFLUENCE_RADIUS)
            * obstacles_rel
            / distances,
            0.0,
        ),
        axis=-2,
    ).squeeze()

    # Repulsion from map borders
    delta_min = np.array([MAP_X_MIN, MAP_Y_MIN]) + BORDER_INFLUENCE_RADIUS - pos
    delta_max = np.array([MAP_X_MAX, MAP_Y_MAX]) - BORDER_INFLUENCE_RADIUS - pos
    border_repulsion = (
        BORDER_MAX_REPULSION
        / BORDER_INFLUENCE_RADIUS
        * (np.maximum(delta_min, 0.0) + np.minimum(delta_max, 0.0))
    )

    return obstacle_repulsion + border_repulsion


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
            clip_angle(current_setpoint[3] - yaw_drone),
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


def clip_norm(
    vec: np.ndarray, max_norm: float | np.ndarray, epsilon: float = 0.0
) -> np.ndarray:
    """
    Clips the norm of the input vector(s) to a maximum value.
    If epsilon is greater than zero, sets the input vector(s) to zero
    if its norm is smaller than epsilon.
    """
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.where(
        norm > max_norm,
        vec * (max_norm / norm),
        np.where(norm > epsilon, vec, 0.0) if epsilon > 0.0 else vec,
    )


def max_norm(vecs: np.ndarray) -> np.ndarray:
    if any([dim == 0 for dim in vecs.shape]):
        return np.zeros(2)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs[:, np.argmax(norms, axis=-2), :]


def smooth_max(vecs: np.ndarray, alpha: float) -> np.ndarray:
    """
    Applies the Boltzmann operator on the input vectors:
    a sum of the vectors weighted by the softargmax of their norms.
    Alpha is the exponential parameter.
    The operation is performed over the last two dimensions (of shape (N, 2))
    """
    if any([dim == 0 for dim in vecs.shape]):
        return np.zeros(2)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    exp_norms = np.exp(alpha * norms)
    return np.sum(vecs * exp_norms, axis=-2, keepdims=True) / np.sum(
        exp_norms, axis=-2, keepdims=True
    )
