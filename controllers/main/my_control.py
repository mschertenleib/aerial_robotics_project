import cv2
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py lines 296-323.
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "range_down": Downward range finder distance (Used instead of Global Z distance)
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)


MAP_X_MIN, MAP_X_MAX = 0.0, 5.0  # [m]
MAP_Y_MIN, MAP_Y_MAX = 0.0, 3.0  # [m]
MAP_RESOLUTION = 0.05  # [m]
SENSOR_RANGE_MAX = 2.0  # [m]
LANDING_ZONE_X = 3.5  # [m]
MAP_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / MAP_RESOLUTION)
MAP_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RESOLUTION)
LANDING_ZONE_IDX = int((LANDING_ZONE_X - MAP_X_MIN) / MAP_RESOLUTION)
IMG_RESOLUTION = 0.005  # [m]
IMG_SIZE_X = int((MAP_X_MAX - MAP_X_MIN) / IMG_RESOLUTION)
IMG_SIZE_Y = int((MAP_Y_MAX - MAP_Y_MIN) / IMG_RESOLUTION)
OBSTACLE_CONFIDENCE = 0.1
KERNEL_RADIUS_M = 0.4  # [m]
KERNEL_RADIUS = int(KERNEL_RADIUS_M / MAP_RESOLUTION)
EXPLORATION_RADIUS_M = 0.2  # [m]
EXPLORATION_RADIUS = int(EXPLORATION_RADIUS_M / MAP_RESOLUTION)
LANDING_PAD_DETECT_RANGE = 0.94


class State(Enum):
    STARTUP = 0
    FIND_PINK_SQUARE = 1
    MOVE_TO_PINK_SQUARE = 2
    MOVE_TO_LANDING_ZONE = 3
    FIND_LANDING_PAD = 4
    CONFIRM_LANDING_PAD = 5
    LAND_ON_LANDING_PAD = 6
    TAKE_OFF_FROM_LANDING_PAD = 7
    PASS_BY_PINK_SQUARE = 8
    BACK_TO_TAKE_OFF_PAD = 9
    FINAL_LANDING = 10


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
g_to_explore = np.zeros((MAP_SIZE_Y, MAP_SIZE_X), dtype=bool)
g_explored = np.zeros((MAP_SIZE_Y, MAP_SIZE_X), dtype=bool)
g_start_time = None
g_target = None
g_pink_square_pos = None

# Visualization
g_enable_visualization = True
g_use_potential_field = False
g_pink_square_detection = False
g_drone_positions = []
g_mouse_x, g_mouse_y = 0, 0

if g_enable_visualization:

    def mouse_callback(event, x, y, flags, param):
        global g_mouse_x, g_mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            g_mouse_x, g_mouse_y = x, y

    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("map", mouse_callback)
    cv2.resizeWindow("map", IMG_SIZE_X, IMG_SIZE_Y)


# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global g_on_ground, g_start_pos, g_t, g_drone_positions, g_first_map_update
    global g_state, g_height_desired, g_start_time, g_enable_visualization
    global g_target, g_pink_square_pos

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
    update_exploration(pos, potential_field)

    if g_state == State.STARTUP:
        do_rotate = True
        g_target = [pos[0], pos[1], 1.0, 0.0]
        if g_pink_square_detection:
            if np.abs(yaw) > 2.0 * np.pi / 3.0:
                g_state = State.FIND_PINK_SQUARE
        else:
            g_state = State.MOVE_TO_LANDING_ZONE

    elif g_state == State.FIND_PINK_SQUARE:
        pink_y, pink_x = np.nonzero(is_pink(camera_data))
        if len(pink_x) > 0:
            do_rotate = False
            cam_y_error = np.mean(pink_y) - camera_data.shape[0] // 2
            if cam_y_error > 10:
                target_alt = g_height_desired - 0.05
            elif cam_y_error < -10:
                target_alt = g_height_desired + 0.05
            else:
                do_rotate = True
                target_alt = g_height_desired
            g_target = [pos[0], pos[1], target_alt, 0.0]

            if np.abs(np.mean(pink_x) - camera_data.shape[1] // 2) < 5:
                g_pink_square_pos = [
                    2.5,
                    pos[1] + (2.5 - pos[0]) * np.tan(yaw),
                    target_alt,
                ]
                if target_alt == g_height_desired:
                    g_state = State.MOVE_TO_PINK_SQUARE

        else:
            do_rotate = True
            g_target = [1.5, 1.5 + np.sin(g_t / 1000.0), g_height_desired, 0.0]

    elif g_state == State.MOVE_TO_PINK_SQUARE:
        do_rotate = True
        g_target = g_pink_square_pos + [0.0]
        if pos[0] > 2.4:
            g_state = State.MOVE_TO_LANDING_ZONE

    elif g_state == State.MOVE_TO_LANDING_ZONE:
        do_rotate = True
        g_target = [4.5, 1.5, 1.0, 0.0]
        if pos[0] > 3.5:
            g_state = State.FIND_LANDING_PAD

    elif g_state == State.FIND_LANDING_PAD:
        do_rotate = True
        g_target = [pos[0], pos[1], 1.0, 0.0]
        g_target[:2] = get_exploration_target(pos)
        if sensor_data["range_down"] < LANDING_PAD_DETECT_RANGE:
            g_target = [pos[0], pos[1], 1.0, 0.0]
            g_start_time = g_t
            g_state = State.CONFIRM_LANDING_PAD

    elif g_state == State.CONFIRM_LANDING_PAD:
        do_rotate = False
        g_target = [pos[0], pos[1], 1.0, 0.0]
        if g_t >= g_start_time + 1.0 / dt:
            g_start_time = None
            g_state = State.LAND_ON_LANDING_PAD

    elif g_state == State.LAND_ON_LANDING_PAD:
        do_rotate = False
        g_target = [pos[0], pos[1], 0.0, 0.0]
        if g_start_time is None and sensor_data["range_down"] < 0.03:
            g_start_time = g_t
        elif g_start_time is not None and g_t >= g_start_time + 4.0 / dt:
            g_start_time = g_t
            g_state = State.TAKE_OFF_FROM_LANDING_PAD

    elif g_state == State.TAKE_OFF_FROM_LANDING_PAD:
        do_rotate = False
        g_target = [pos[0], pos[1], 1.0, 0.0]
        if sensor_data["range_down"] > 0.9:
            if g_pink_square_detection:
                g_state = State.PASS_BY_PINK_SQUARE
            else:
                g_state = State.BACK_TO_TAKE_OFF_PAD

    elif g_state == State.PASS_BY_PINK_SQUARE:
        do_rotate = True
        g_target = g_pink_square_pos + [0.0]
        if pos[0] < 2.6:
            g_state = State.BACK_TO_TAKE_OFF_PAD

    elif g_state == State.BACK_TO_TAKE_OFF_PAD:
        do_rotate = True
        g_target = [g_start_pos[0], g_start_pos[1], 1.0, 0.0]
        if np.linalg.norm(g_target[:2] - pos) < 0.1:
            g_state = State.FINAL_LANDING

    elif g_state == State.FINAL_LANDING:
        do_rotate = False
        g_target = [g_start_pos[0], g_start_pos[1], 0.0, 0.0]

    else:  # This should never happen
        do_rotate = False
        g_target = [pos[0], pos[1], g_height_desired, 0.0]

    if g_use_potential_field:
        control_command = get_potential_field_control_command(
            pos=pos,
            yaw=yaw,
            target=np.array(g_target),
            dt=dt,
            do_rotate=do_rotate,
            potential_field=potential_field,
        )
    else:
        control_command = get_control_command(
            sensor_data=sensor_data,
            target=np.array(g_target),
            dt=dt,
            do_rotate=False,
        )

    if g_enable_visualization and g_t % 5 == 0:
        g_drone_positions += pos.tolist()
        map_image = create_image(
            pos=pos,
            yaw=yaw,
            target=np.array(g_target),
            sensor_data=sensor_data,
            occupancy_map=occupancy_map,
            potential_field=potential_field,
            to_explore=g_to_explore,
            explored=g_explored,
            pink_square=(
                np.array(g_pink_square_pos) if g_pink_square_pos is not None else None
            ),
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


def get_potential_field_control_command(
    pos: np.ndarray,
    yaw: float,
    target: np.ndarray,
    dt: float,
    do_rotate: bool,
    potential_field: np.ndarray,
) -> np.ndarray:
    global g_height_desired

    vel_cmd, _, _, _ = get_world_potential_field_velocity_command(
        pos=pos, target=target[:2], potential_field=potential_field
    )
    vel_cmd = rotate(vel_cmd, -yaw)
    VERTICAL_SPEED = 0.15  # [m/s]
    if target[2] > g_height_desired:
        g_height_desired = target[2]
    elif target[2] < g_height_desired:
        g_height_desired = max(g_height_desired - VERTICAL_SPEED * dt, target[2])
    yaw_rate_cmd = 2.0 if do_rotate else 0.0
    control_command = [vel_cmd[0], vel_cmd[1], g_height_desired, yaw_rate_cmd]
    return control_command


def get_world_potential_field_velocity_command(
    pos: np.ndarray, target: np.ndarray, potential_field: np.ndarray
):
    vel_attractive = get_attraction(pos=pos, target=target, radius=0.1, max_value=0.2)
    vel_repulsive = get_potential_field_repulsion(
        pos=pos, potential_field=potential_field
    )
    vel_corrective = get_correction(
        pos=pos, attraction=vel_attractive, repulsion=vel_repulsive
    )
    vel = clip_norm(
        vel_attractive + vel_repulsive + vel_corrective,
        max_norm=0.3,
        epsilon=0.001,
    )
    return vel, vel_attractive, vel_repulsive, vel_corrective


def get_control_command(
    sensor_data: np.ndarray,
    target: np.ndarray,
    dt: float,
    do_rotate: bool,
) -> np.ndarray:
    global g_height_desired

    vel_cmd, _, _, _= get_world_velocity_command(
        sensor_data=sensor_data, target=target[:2]
    )
    vel_cmd = rotate(vel_cmd, -sensor_data["yaw"])
    VERTICAL_SPEED = 0.15  # [m/s]
    if target[2] > g_height_desired:
        g_height_desired = target[2]
    elif target[2] < g_height_desired:
        g_height_desired = max(g_height_desired - VERTICAL_SPEED * dt, target[2])
    yaw_rate_cmd = 2.0 if do_rotate else 0.0
    control_command = [vel_cmd[0], vel_cmd[1], g_height_desired, yaw_rate_cmd]
    return control_command


def get_world_velocity_command(sensor_data: np.ndarray, target: np.ndarray):
    pos = np.array([sensor_data["x_global"], sensor_data["y_global"]])
    yaw = sensor_data["yaw"]
    range_front = sensor_data["range_front"]
    range_left = sensor_data["range_left"]
    range_back = sensor_data["range_back"]
    range_right = sensor_data["range_right"]

    vel_attractive = get_attraction(pos=pos, target=target, radius=0.1, max_value=0.2)
    vel_repulsive = get_repulsion(
        pos=pos,
        yaw=yaw,
        range_front=range_front,
        range_left=range_left,
        range_back=range_back,
        range_right=range_right,
    )
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
    radius: float,
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
    return clip_norm(
        max_value / radius * (target - pos),
        max_norm=max_value,
        epsilon=0.001,
    )


def get_repulsion(
    pos: np.ndarray,
    yaw: float,
    range_front: float,
    range_left: float,
    range_back: float,
    range_right: float,
) -> np.ndarray:
    MAX_REPULSION = 0.6
    REPULSION_RANGE = 0.4

    def rep(range: float) -> float:
        return max(MAX_REPULSION / REPULSION_RANGE * (REPULSION_RANGE - range), 0.0)

    rep_local = np.array(
        [rep(range_back) - rep(range_front), rep(range_right) - rep(range_left)]
    )
    rep_obstacles = rotate(rep_local, yaw)

    rep_walls = np.zeros(2)
    return rep_obstacles + rep_walls


def get_potential_field_repulsion(
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
    pos_safe = np.clip(
        pos, [MAP_X_MIN, MAP_Y_MIN], [MAP_X_MAX - 0.001, MAP_Y_MAX - 0.001]
    )
    map_pos = (pos_safe - np.array([MAP_X_MIN, MAP_Y_MIN])) / MAP_RESOLUTION
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
    if cos_angle >= 0.0:
        return np.zeros(2)

    tangent = np.array([-repulsion[1], repulsion[0]])
    if np.cross(attraction, repulsion) >= 0.0:
        tangent *= -1.0

    return CORRECTION_FACTOR * np.abs(cos_angle) * tangent


def update_exploration(pos: np.ndarray, potential_field: np.ndarray) -> None:
    global g_to_explore, g_explored
    g_to_explore = potential_field < 0.12
    g_to_explore[:, :LANDING_ZONE_IDX] = False
    g_explored = cv2.circle(
        g_explored.astype(np.uint8),
        center=global_to_map(pos)[::-1],
        radius=EXPLORATION_RADIUS,
        color=1,
        thickness=-1,
    ).astype(bool)


def get_exploration_target(pos: np.ndarray) -> np.ndarray:
    global g_to_explore, g_explored

    unexplored = g_to_explore & ~g_explored
    idx_unexplored = np.argwhere(unexplored)
    if len(idx_unexplored) == 0:
        g_explored = np.zeros_like(g_explored)
        return pos

    deltas = idx_unexplored - global_to_map(pos).reshape((1, 2))
    distances_sq = np.sum(np.square(deltas), axis=-1)
    min_dist_idx = np.argmin(distances_sq)
    return map_to_global(idx_unexplored[min_dist_idx, :])


def is_pink(arr: np.ndarray) -> np.ndarray:
    return (arr[..., 0] // 2 > arr[..., 1]) & (arr[..., 2] // 2 > arr[..., 1])


def create_image(
    pos: np.ndarray,
    yaw: float,
    target: np.ndarray,
    sensor_data: np.ndarray,
    occupancy_map: np.ndarray,
    potential_field: np.ndarray,
    to_explore: np.ndarray,
    explored: np.ndarray,
    pink_square: np.ndarray | None,
) -> np.ndarray:

    img = 1.0 - occupancy_map
    # img = 1.0 - potential_field
    img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[to_explore & ~explored] = (64, 192, 192)
    img[to_explore & explored] = (192, 192, 64)
    img = cv2.resize(
        img,
        dsize=(IMG_SIZE_X, IMG_SIZE_Y),
        interpolation=cv2.INTER_NEAREST,
    )

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

    # Draw target
    cv2.drawMarker(
        img,
        global_to_img(target[:2]),
        color=(0, 0, 255),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=20,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    # Draw pink square position
    if pink_square is not None:
        cv2.drawMarker(
            img,
            global_to_img(pink_square[:2]),
            color=(255, 0, 255),
            markerType=cv2.MARKER_SQUARE,
            markerSize=20,
            thickness=2,
            line_type=cv2.LINE_AA,
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
        if g_use_potential_field:
            vel, vel_attractive, vel_repulsive, vel_corrective = (
                get_world_potential_field_velocity_command(
                    pos=pos, target=target[:2], potential_field=potential_field
                )
            )
        else:
            vel, vel_attractive, vel_repulsive, vel_corrective = get_world_velocity_command(
                sensor_data=sensor_data, target=target[:2]
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
    if g_use_potential_field:
        draw_arrows(pos=mouse_global, target=target)

    return np.flip(img, axis=0)


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
    Clips the norm of the input vector to a maximum value.
    If epsilon is greater than zero, sets the input vector to zero
    if its norm is smaller than epsilon.
    """
    norm = np.linalg.norm(vec)
    if norm > max_norm:
        return vec * (max_norm / norm)
    elif epsilon > 0.0 and norm < epsilon:
        return np.zeros(2)
    else:
        return vec
