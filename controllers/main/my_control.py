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

MAP_X_MIN, MAP_X_MAX = 0.0, 5.0  # [m]
MAP_Y_MIN, MAP_Y_MAX = 0.0, 3.0  # [m]
MAP_RESOLUTION = 0.05  # [m]
SENSOR_RANGE_MAX = 2.0  # [m]
MAP_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / MAP_RESOLUTION)
MAP_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RESOLUTION)

IMG_RESOLUTION = 0.01  # [m]
IMG_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / IMG_RESOLUTION)
IMG_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / IMG_RESOLUTION)

OBSTACLE_CONFIDENCE = 0.1
OBSTACLE_RADIUS = 0.3  # [m]
KERNEL_RADIUS = int(OBSTACLE_RADIUS / MAP_RESOLUTION)

# Global variables
on_ground = True
height_desired = 1.0
timer = None
start_pos = None
timer_done = None
t = 0

# 0 = free, 0.5 = unknown, 1 = occupied
occupancy_map = np.zeros((MAP_SIZE_Y, MAP_SIZE_X), dtype=np.float32)
occupancy_map[:] = 0.5
drone_positions = []
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
            [start_pos[0], start_pos[1], 1.0, 0.0],
        ]
    )
    path = np.array([[2.0, 1.0, 1.0, 0.0]])
    target = path_to_setpoint(path, sensor_data, dt)
    occupancy_map = update_occupancy_map(sensor_data)
    potential_field = build_potential_field(occupancy_map)
    control_command = get_control_command(
        sensor_data=sensor_data, target=target, occupancy_map=occupancy_map
    )

    if t % 2 == 0:
        map_image = create_image(
            sensor_data=sensor_data,
            occupancy_map=occupancy_map,
            potential_field=potential_field,
        )
        cv2.imshow("map", map_image)
        cv2.waitKey(1)

    t += 1

    # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]
    return control_command


def build_potential_field(occupancy_map: np.ndarray) -> np.ndarray:
    kernel = get_quadratic_kernel(KERNEL_RADIUS)
    potential = cv2.filter2D(occupancy_map, -1, kernel)
    return potential


def get_linear_kernel(kernel_radius: int) -> np.ndarray:
    kernel_size = 2 * kernel_radius + 1
    x = np.linspace(-kernel_radius, kernel_radius, kernel_size)
    y = np.linspace(-kernel_radius, kernel_radius, kernel_size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(np.square(xv) + np.square(yv))
    kernel = np.maximum(kernel_radius - r, 0.0)
    return kernel / kernel_radius  # np.sum(kernel)


def get_quadratic_kernel(kernel_radius: int) -> np.ndarray:
    kernel_size = 2 * kernel_radius + 1
    x = np.linspace(-kernel_radius, kernel_radius, kernel_size)
    y = np.linspace(-kernel_radius, kernel_radius, kernel_size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(np.square(xv) + np.square(yv))
    kernel = np.square(np.maximum(kernel_radius - r, 0.0))
    return kernel / np.sum(kernel)


def get_repulsion(
    pos: np.ndarray,
    potential_field: np.ndarray,
) -> np.ndarray:
    grad = get_gradient(pos, potential_field)
    return -grad


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

    p0 = potential_field[i, j]
    p1 = potential_field[i + 1, j - 1]
    p2 = potential_field[i + 1, j]
    p3 = potential_field[i + 1, j + 1]
    p4 = potential_field[i, j - 1]
    p5 = potential_field[i, j + 1]
    p6 = potential_field[i - 1, j - 1]
    p7 = potential_field[i - 1, j]
    p8 = potential_field[i - 1, j + 1]
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
    print(grad_00, grad_10, grad_01, grad_11, grad)
    return grad


def create_image(
    sensor_data: dict,
    occupancy_map: np.ndarray,
    potential_field: np.ndarray,
) -> np.ndarray:
    map_grayscale = np.clip((1.0 - occupancy_map) * 255.0, 0.0, 255.0).astype(np.uint8)
    img = cv2.cvtColor(
        cv2.resize(
            map_grayscale,
            dsize=(IMG_SIZE_X, IMG_SIZE_Y),
            interpolation=cv2.INTER_NEAREST,
        ),
        cv2.COLOR_GRAY2BGR,
    )

    pot = np.clip(
        cv2.resize(
            potential_field,
            dsize=(IMG_SIZE_X, IMG_SIZE_Y),
            interpolation=cv2.INTER_NEAREST,
        )
        * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    pot = cv2.cvtColor(pot, cv2.COLOR_GRAY2BGR)

    # pot[..., :2] = 0
    # img = cv2.addWeighted(img, 0.5, pot, 0.5, 0.0)
    img[:] = pot

    # Draw drone
    pos_global = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    yaw = sensor_data["yaw"]
    DRONE_SIZE = 0.1  # [m]
    pos_to_tip = np.array([np.cos(yaw), np.sin(yaw)]) * DRONE_SIZE * 0.5
    left = np.array([-pos_to_tip[1], pos_to_tip[0]]) * 0.5
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

    offset = np.array([MAP_X_MIN, MAP_Y_MIN])
    mouse_global = (np.array([g_mouse_x, g_mouse_y]) + 0.5) * IMG_RESOLUTION + offset
    rep = get_repulsion(mouse_global, potential_field)
    tip = mouse_global + rep * 100
    tip = global_to_img(tip)
    cv2.arrowedLine(img, (g_mouse_x, g_mouse_y), tip, color=255)

    # img[:] = np.flip(img, axis=0)
    return img

    # TEST
    img = np.zeros_like(occupancy_map)
    img[10, 20] = 1.0
    img[11, 20] = 1.0
    img[12, 20] = 1.0
    img[14, 20] = 1.0
    img[12, 23] = 1.0
    print(img.min(), img.max(), end=" ")
    kernel = get_quadratic_kernel(KERNEL_RADIUS)
    img = cv2.filter2D(img, -1, kernel)
    if img.max() > 0.0:
        img /= img.max()
    print(img.min(), img.max())
    img = np.clip(
        cv2.resize(
            1.0 - img,
            dsize=(IMG_SIZE_X, IMG_SIZE_Y),
            interpolation=cv2.INTER_NEAREST,
        )
        * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return np.flip(img, axis=0)


def get_control_command(
    sensor_data: dict, target: np.ndarray, occupancy_map: np.ndarray
) -> np.ndarray:
    CORRECTION_FACTOR = 1.0

    pos = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    yaw = sensor_data["yaw"]

    vel_attractive = attraction_to_target(
        pos, target[:2], min_radius=0.01, max_radius=0.2, max_value=0.2
    )

    obstacles = map_to_global(np.argwhere(occupancy_map <= -0.2))
    vel_repulsive = repulsion(pos, obstacles)

    norm_attractive = np.linalg.norm(vel_attractive)
    norm_repulsive = np.linalg.norm(vel_repulsive)
    if norm_repulsive > 0.001 and norm_attractive > 0.001:
        cos_angle = np.dot(vel_attractive, vel_repulsive) / (
            norm_attractive * norm_repulsive
        )
        perpendicular = np.array([-vel_repulsive[1], vel_repulsive[0]])
        vel_corrective = CORRECTION_FACTOR * np.abs(cos_angle) * perpendicular
    else:
        vel_corrective = np.zeros(2)

    vel = clip_norm(
        (vel_attractive + vel_repulsive + vel_corrective).squeeze(),
        max_norm=0.3,
        epsilon=0.001,
    )
    # print(
    #    f"{np.linalg.norm(vel_attractive):7.4f}",
    #    f"{np.linalg.norm(vel_repulsive):7.4f}",
    #    f"{np.linalg.norm(vel_corrective):7.4f}",
    #    f"{sensor_data["range_down"]:7.4f}",
    # )

    vel = rotate(vel, -yaw)
    control_command = [vel[0], vel[1], 1.0, 2.0]
    return control_command


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


def get_prefered_repulsion_sign(pos: np.ndarray, dir: np.ndarray) -> float:
    if distance_to_wall(pos, dir) > distance_to_wall(pos, -dir):
        return 1.0
    else:
        return -1.0


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
                    occupancy_map[index[0], index[1]] -= OBSTACLE_CONFIDENCE
            else:
                if is_index_valid_map(row=index[0], col=index[1]):
                    occupancy_map[index[0], index[1]] += OBSTACLE_CONFIDENCE
                break

    occupancy_map = np.clip(occupancy_map, 0.0, 1.0)

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
