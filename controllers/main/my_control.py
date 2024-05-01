# Examples of basic methods for simulation competition
import cv2
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

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

MAP_X_MIN, MAP_X_MAX = 0.0, 5.0  # [m]
MAP_Y_MIN, MAP_Y_MAX = 0.0, 3.0  # [m]
MAP_RESOLUTION = 0.05  # [m]
SENSOR_RANGE_MAX = 2.0  # [m]
MAP_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / MAP_RESOLUTION)
MAP_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RESOLUTION)
IMG_RESOLUTION = 0.005  # [m]
IMG_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / IMG_RESOLUTION)
IMG_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / IMG_RESOLUTION)
OBSTACLE_CONFIDENCE = 0.1
KERNEL_RADIUS = 9


class State(Enum):
    STARTUP = 0
    MOVE_TO_LANDING_ZONE = 1
    FIND_LANDING_PAD = 2


# Global variables
g_on_ground = True
g_height_desired = 1.0
g_timer = None
g_start_pos = None
g_timer_done = None
g_t = 0
# 0 = free, 0.5 = unknown, 1 = occupied
g_occupancy_map = np.zeros((MAP_SIZE_Y, MAP_SIZE_X), dtype=np.float32)
g_occupancy_map[:] = 0.5
g_first_map_update = True
g_state = State.STARTUP


# Visualization
g_drone_positions = []
g_mouse_x, g_mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global g_mouse_x, g_mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        g_mouse_x, g_mouse_y = x, y


cv2.namedWindow("map", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("map", mouse_callback)
cv2.resizeWindow("map", IMG_SIZE_X, IMG_SIZE_Y)


# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global g_on_ground, g_start_pos, g_t, g_drone_positions, g_first_map_update, g_state

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)

    # Take off
    if g_start_pos is None:
        g_start_pos = [
            sensor_data["x_global"],
            sensor_data["y_global"],
            sensor_data["range_down"],
        ]
    if g_on_ground and sensor_data["range_down"] < 0.49:
        control_command = [0.0, 0.0, g_height_desired, 0.0]
        return control_command
    else:
        g_on_ground = False

    # ---- YOUR CODE HERE ----

    path = np.array(
        [
            [3.7, 0.2, 1.0, 0.0],
            [4.8, 0.4, 1.0, 0.0],
            [3.7, 0.6, 1.0, 0.0],
            [4.8, 0.8, 1.0, 0.0],
            [3.7, 1.0, 1.0, 0.0],
            [4.8, 1.2, 1.0, 0.0],
            [3.7, 1.4, 1.0, 0.0],
            [4.8, 1.6, 1.0, 0.0],
            [3.7, 1.8, 1.0, 0.0],
            [4.8, 2.0, 1.0, 0.0],
            [3.7, 2.2, 1.0, 0.0],
            [4.8, 2.4, 1.0, 0.0],
            [3.7, 2.6, 1.0, 0.0],
            [4.8, 2.8, 1.0, 0.0],
            [g_start_pos[0], g_start_pos[1], 1.0, 0.0],
        ]
    )
    target = path_to_setpoint(path, sensor_data, dt)

    pos = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    yaw = sensor_data["yaw"]

    if g_first_map_update:
        cv2.circle(
            g_occupancy_map,
            global_to_map(pos)[::-1],
            4,
            color=0.0,
            thickness=-1,
        )
        g_first_map_update = False

    occupancy_map = update_occupancy_map(sensor_data)
    potential_field = build_potential_field(occupancy_map)

    print(g_state)

    if g_state == State.STARTUP:
        control_command = [0.0, 0.0, 1.0, 2.0]
        if np.abs(yaw) > 2.0 * np.pi / 3.0:
            g_state = State.MOVE_TO_LANDING_ZONE
    elif g_state == State.MOVE_TO_LANDING_ZONE:
        target = [4.5, 1.5, 1.0, 0.0]
        control_command = get_control_command(
            pos=pos, yaw=yaw, target=target, potential_field=potential_field
        )
        if pos[0] > 3.5:
            g_state = State.FIND_LANDING_PAD
    elif g_state == State.FIND_LANDING_PAD:
        control_command = [0.0, 0.0, 0.0, 0.0]
    else:
        control_command = [0.0, 0.0, 0.0, 0.0]

    if g_t % 5 == 0:
        g_drone_positions += pos.tolist()
        map_image = create_image(
            pos=pos,
            yaw=yaw,
            target=target,
            occupancy_map=occupancy_map,
            potential_field=potential_field,
        )
        cv2.imshow("map", map_image)
        cv2.waitKey(1)

    g_t += 1

    # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]
    return control_command


def build_potential_field(occupancy_map: np.ndarray) -> np.ndarray:
    potential = occupancy_map.copy()
    BORDER_POTENTIAL = 2.0
    potential[0, :] = BORDER_POTENTIAL
    potential[-1, :] = BORDER_POTENTIAL
    potential[:, 0] = BORDER_POTENTIAL
    potential[:, -1] = BORDER_POTENTIAL
    kernel = get_kernel(KERNEL_RADIUS)
    potential = cv2.filter2D(potential, -1, kernel)
    return potential


def get_kernel(kernel_radius: int) -> np.ndarray:
    kernel_size = 2 * kernel_radius + 1
    x = np.linspace(-kernel_radius, kernel_radius, kernel_size)
    y = np.linspace(-kernel_radius, kernel_radius, kernel_size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(np.square(xv) + np.square(yv))
    kernel = np.square(np.maximum(kernel_radius - r, 0.0))
    return kernel / np.sum(kernel)


def get_control_command(
    pos: np.ndarray, yaw: float, target: np.ndarray, potential_field: np.ndarray
) -> np.ndarray:
    vel, _, _, _ = get_world_velocity_command(
        pos=pos, target=target[:2], potential_field=potential_field
    )
    vel = rotate(vel, -yaw)
    control_command = [vel[0], vel[1], 1.0, 2.0]
    return control_command


def get_world_velocity_command(
    pos: np.ndarray, target: np.ndarray, potential_field: np.ndarray
) -> np.ndarray:
    vel_attractive = get_attraction(
        pos=pos, target=target, min_radius=0.01, max_radius=0.2, max_value=0.2
    )
    vel_repulsive = get_repulsion(pos=pos, potential_field=potential_field)
    vel_corrective = get_correction(
        pos=pos, attraction=vel_attractive, repulsion=vel_repulsive
    )
    vel = clip_norm(
        vel_attractive + vel_repulsive + vel_corrective,
        max_norm=0.3,
        epsilon=0.001,
    )
    return vel, vel_attractive, vel_repulsive, vel_corrective


def get_attraction(
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


def get_repulsion(
    pos: np.ndarray,
    potential_field: np.ndarray,
) -> np.ndarray:
    GRADIENT_SCALE = 4.0
    grad = get_gradient(pos, potential_field)
    return -grad * GRADIENT_SCALE


def get_gradient(
    pos: np.ndarray,
    potential_field: np.ndarray,
) -> np.ndarray:
    """
    Returns a bilinear interpolation of the gradients at each cell corner.
    The gradients at each cell corner are computed using the 2x2 adjacent cells.
    """

    map_pos = (pos - np.array([MAP_X_MIN, MAP_Y_MIN])) / MAP_RESOLUTION
    map_pos_fractional, map_pos_integral = np.modf(map_pos)
    i, j = int(map_pos_integral[1]), int(map_pos_integral[0])
    x, y = map_pos_fractional[0], map_pos_fractional[1]

    """
              p1  |  p2  |  p3 
             ----0,1----1,1----
              p4  |  p0  |  p5 
             ----0,0----1,0----
              p6  |  p7  |  p8 
    
    ^ y,i
    |
    o---> x,j
    """

    i_is_min = i == 0
    if i_is_min:
        y = 1.0
    i_is_max = i == potential_field.shape[0] - 1
    if i_is_max:
        y = 0.0
    j_is_min = j == 0
    if j_is_min:
        x = 1.0
    j_is_max = j == potential_field.shape[1] - 1
    if j_is_max:
        x = 0.0

    p0 = potential_field[i, j]
    p1 = potential_field[i + 1, j - 1] if not i_is_max and not j_is_min else 0.0
    p2 = potential_field[i + 1, j] if not i_is_max else 0.0
    p3 = potential_field[i + 1, j + 1] if not i_is_max and not j_is_max else 0.0
    p4 = potential_field[i, j - 1] if not j_is_min else 0.0
    p5 = potential_field[i, j + 1] if not j_is_max else 0.0
    p6 = potential_field[i - 1, j - 1] if not i_is_min and not j_is_min else 0.0
    p7 = potential_field[i - 1, j] if not i_is_min else 0.0
    p8 = potential_field[i - 1, j + 1] if not i_is_min and not j_is_max else 0.0

    grad_00 = 0.5 * np.array([p0 - p4 + p7 - p6, p4 - p6 + p0 - p7])
    grad_10 = 0.5 * np.array([p5 - p0 + p8 - p7, p0 - p7 + p5 - p8])
    grad_01 = 0.5 * np.array([p2 - p1 + p0 - p4, p1 - p4 + p2 - p0])
    grad_11 = 0.5 * np.array([p3 - p2 + p5 - p0, p2 - p0 + p3 - p5])
    grad = (
        grad_00 * (1.0 - x) * (1.0 - y)
        + grad_01 * (1.0 - x) * y
        + grad_10 * x * (1.0 - y)
        + grad_11 * x * y
    )
    return grad


def get_correction(
    pos: np.ndarray, attraction: np.ndarray, repulsion: np.ndarray
) -> np.ndarray:
    CORRECTION_FACTOR = 0.3

    norm_attractive = np.linalg.norm(attraction)
    norm_repulsive = np.linalg.norm(repulsion)
    if norm_repulsive < 0.001 or norm_attractive < 0.001:
        return np.zeros(2)

    cos_angle = np.dot(attraction, repulsion) / (norm_attractive * norm_repulsive)
    tangent = np.array([-repulsion[1], repulsion[0]])
    if np.cross(attraction, repulsion) >= 0.0:
        tangent *= -1.0
    # perpendicular = get_prefered_direction(pos, perpendicular)
    return CORRECTION_FACTOR * np.abs(cos_angle) * tangent


def create_image(
    pos: np.ndarray,
    yaw: float,
    target: np.ndarray,
    occupancy_map: np.ndarray,
    potential_field: np.ndarray,
) -> np.ndarray:

    # grayscale = np.clip((1.0 - occupancy_map) * 255.0, 0.0, 255.0).astype(np.uint8)
    grayscale = np.clip((1.0 - potential_field) * 255.0, 0.0, 255.0).astype(np.uint8)
    grayscale = cv2.resize(
        grayscale,
        dsize=(IMG_SIZE_X, IMG_SIZE_Y),
        interpolation=cv2.INTER_NEAREST,
    )
    img = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    # Draw drone path
    if len(g_drone_positions) > 0:
        pts = global_to_img(np.array(g_drone_positions))
        cv2.polylines(
            img,
            pts=[pts.reshape((-1, 1, 2))],
            isClosed=False,
            color=(192, 64, 192),
            lineType=cv2.LINE_AA,
        )

    # Draw drone
    DRONE_SIZE = 0.08
    pos_to_tip = np.array([np.cos(yaw), np.sin(yaw)]) * DRONE_SIZE * 0.5
    left = np.array([-pos_to_tip[1], pos_to_tip[0]]) * 0.5
    tip_global = pos + pos_to_tip
    left_global = pos - pos_to_tip + left
    right_global = pos - pos_to_tip - left
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
        color=(0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    def draw_arrows(pos: np.ndarray, target: np.ndarray) -> None:
        vel, vel_attractive, vel_repulsive, vel_corrective = get_world_velocity_command(
            pos=pos, target=target[:2], potential_field=potential_field
        )
        scale = 2.0
        pt1 = global_to_img(pos)
        pt2 = global_to_img(pos + vel_attractive * scale)
        cv2.arrowedLine(img, pt1, pt2, color=(64, 192, 64), line_type=cv2.LINE_AA)
        pt2 = global_to_img(pos + vel_repulsive * scale)
        cv2.arrowedLine(img, pt1, pt2, color=(64, 64, 192), line_type=cv2.LINE_AA)
        pt2 = global_to_img(pos + vel_corrective * scale)
        cv2.arrowedLine(img, pt1, pt2, color=(192, 64, 64), line_type=cv2.LINE_AA)
        pt2 = global_to_img(pos + vel * scale)
        cv2.arrowedLine(img, pt1, pt2, color=(0, 0, 0), line_type=cv2.LINE_AA)

    # Draw attraction/repulsion on drone
    draw_arrows(pos=pos, target=target)

    # Draw attraction/repulsion at mouse position
    offset = np.array([MAP_X_MIN, MAP_Y_MIN])
    mouse_x = min(g_mouse_x, IMG_SIZE_X - 1)
    mouse_y = max(IMG_SIZE_Y - 1 - g_mouse_y, 0)
    mouse_global = np.array([mouse_x, mouse_y]) * IMG_RESOLUTION + offset
    draw_arrows(pos=mouse_global, target=target)

    return np.flip(img, axis=0)


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


def get_prefered_direction(pos: np.ndarray, dir: np.ndarray) -> float:
    dist_pos = distance_to_wall(pos, dir)
    dist_neg = distance_to_wall(pos, -dir)
    if dist_pos > dist_neg:
        return dir
    else:
        return -dir


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
    global g_occupancy_map

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
                    g_occupancy_map[index[0], index[1]] -= OBSTACLE_CONFIDENCE
            else:
                if is_index_valid_map(row=index[0], col=index[1]):
                    g_occupancy_map[index[0], index[1]] += OBSTACLE_CONFIDENCE
                break

    g_occupancy_map = np.clip(g_occupancy_map, 0.0, 1.0)

    return g_occupancy_map


# --- Control from the exercises ---

g_index_current_setpoint = 0


def path_to_setpoint(path, sensor_data, dt):
    global g_on_ground, g_height_desired, g_index_current_setpoint, g_timer, g_timer_done, g_start_pos

    # Take off
    if g_start_pos is None:
        g_start_pos = [
            sensor_data["x_global"],
            sensor_data["y_global"],
            sensor_data["range_down"],
        ]
    if g_on_ground and sensor_data["range_down"] < 0.49:
        current_setpoint = [g_start_pos[0], g_start_pos[1], g_height_desired, 0.0]
        return current_setpoint
    else:
        g_on_ground = False

    # Start timer
    if (g_index_current_setpoint == 1) & (g_timer is None):
        g_timer = 0
        print("Time recording started")
    if g_timer is not None:
        g_timer += dt
    # Hover at the final setpoint
    if g_index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [g_start_pos[0], g_start_pos[1], g_start_pos[2] - 0.05, 0.0]

        if g_timer_done is None:
            g_timer_done = True
            print("Path planing took " + str(np.round(g_timer, 1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[g_index_current_setpoint]
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
        g_index_current_setpoint += 1
        # Hover at the final setpoint
        if g_index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, g_height_desired, 0.0]
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
