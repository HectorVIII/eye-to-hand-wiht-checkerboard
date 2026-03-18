#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zed_wrist_move_once_single_person.py

Workflow:
    1. Detect and track the user's right wrist with ZED2
    2. Convert wrist point from camera frame to robot base frame using T_cam2base
    3. Filter the wrist position
    4. If the filtered wrist stays sufficiently still for a while,
       generate a target near the wrist in base frame
    5. Move xArm ONLY ONCE
    6. After the robot reaches the target, end the program directly

Current behavior:
    - Single-person testing style
    - No locked body ID
    - No "choose nearest body" logic
    - In each frame, whichever body FIRST satisfies the wrist condition is used
    - After one successful detection + move, the task ends
"""

import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl
from xarm.wrapper import XArmAPI


# =========================================================
# Hard-coded hand-eye result (camera -> base)
# =========================================================
T_cam2base = np.array([
    [-0.483071955,  0.520801933, -0.703851428,  0.835195325],
    [ 0.875557739,  0.281511210, -0.392619515,  0.717600193],
    [-0.006334936, -0.805926042, -0.591982334,  0.530602896],
    [ 0.0,          0.0,          0.0,          1.0        ],
], dtype=np.float64)

# Running method: PARK
#   camera_in_base = [0.835195325, 0.717600193, 0.530602896]
#   trans_mean = 0.200780 m
#   trans_max  = 0.430585 m
#   rot_mean   = 25.4708 deg
#   rot_max    = 52.1918 deg
#   score      = 0.455488


# =========================================================
# User settings
# =========================================================
ROBOT_IP = "192.168.1.225"
USE_RADIANS = True

WINDOW_NAME = "ZED Wrist Detect -> xArm Move Once"

# BODY_18 right wrist index
RIGHT_WRIST_IDX = 17
USE_KEYPOINT_NAME = "RIGHT_WRIST"

# ZED settings
USE_HD720 = True
CAMERA_FPS = 60
COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.IMAGE
DEPTH_MODE = sl.DEPTH_MODE.NEURAL
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 2.00

# Body tracking
BODY_MODEL = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
BODY_FORMAT = sl.BODY_FORMAT.BODY_34
DETECTION_CONF_THRESH = 40

# Filtering
MEDIAN_WINDOW = 7
EMA_ALPHA = 0.20
MAX_JUMP_M = 0.18

# Static detection
STATIC_TIME_S = 1.5
STATIC_RADIUS_M = 0.04

# Target in robot base frame
# Move to wrist directly: [0, 0, 0]
# Move above wrist by 20 cm: [0, 0, 0.20]
WRIST_TARGET_OFFSET_M = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Robot motion
MOVE_SPEED_MM_S = 60
MOVE_ACC_MM_S2 = 120
WAIT_AFTER_MOVE_S = 0.5

# Safety workspace in base frame (meters)
X_MIN, X_MAX = 0.20, 0.80
Y_MIN, Y_MAX = -0.10, 0.80
Z_MIN, Z_MAX = 0.15, 0.70

# Visual scale
DISPLAY_SCALE = 1.4
PRINT_INTERVAL_S = 0.3


# =========================================================
# Math helpers
# =========================================================
def cam_point_to_base(T_cam2base, p_cam):
    p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    p_base_h = T_cam2base @ p_cam_h
    return p_base_h[:3]


def median_filter_point(buffer):
    if len(buffer) == 0:
        return None
    arr = np.array(buffer, dtype=np.float64)
    return np.median(arr, axis=0)


def ema_update(prev, current, alpha):
    if current is None:
        return prev
    if prev is None:
        return current.copy()
    return alpha * current + (1.0 - alpha) * prev


def to_bgr(frame):
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def is_valid_point(p):
    if p is None:
        return False
    if not np.all(np.isfinite(p)):
        return False
    z = float(p[2])
    return DEPTH_MIN_M <= z <= DEPTH_MAX_M


def draw_text_block(img, lines, org=(20, 35), dy=28, color=(0, 255, 0)):
    x, y = org
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        y += dy


def workspace_ok(p_base):
    x, y, z = p_base
    return (X_MIN <= x <= X_MAX) and (Y_MIN <= y <= Y_MAX) and (Z_MIN <= z <= Z_MAX)


def clamp_workspace(p_base):
    x = min(max(p_base[0], X_MIN), X_MAX)
    y = min(max(p_base[1], Y_MIN), Y_MAX)
    z = min(max(p_base[2], Z_MIN), Z_MAX)
    return np.array([x, y, z], dtype=np.float64)


# =========================================================
# xArm helpers
# =========================================================
def connect_arm(robot_ip):
    print(f"Connecting to xArm at {robot_ip} ...")
    arm = XArmAPI(robot_ip)
    arm.connect()

    arm.motion_enable(enable=True)
    arm.clean_warn()
    arm.clean_error()
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    return arm


def get_robot_pose_aa(arm):
    code, pose_aa = arm.get_position_aa(is_radian=USE_RADIANS)
    if code != 0:
        raise RuntimeError(f"get_position_aa failed, code={code}")
    return np.array(pose_aa, dtype=np.float64)


def move_pose_aa(arm, pose_aa):
    code = arm.set_position_aa(
        axis_angle_pose=pose_aa.tolist(),
        speed=MOVE_SPEED_MM_S,
        mvacc=MOVE_ACC_MM_S2,
        is_radian=USE_RADIANS,
        wait=True,
    )
    if code != 0:
        raise RuntimeError(f"set_position_aa failed, code={code}")


def move_target_pose(arm, target_base_xyz_m, fixed_orientation_rvec):
    target_mm = target_base_xyz_m * 1000.0

    pose_aa = np.array([
        target_mm[0],
        target_mm[1],
        target_mm[2],
        fixed_orientation_rvec[0],
        fixed_orientation_rvec[1],
        fixed_orientation_rvec[2],
    ], dtype=np.float64)

    move_pose_aa(arm, pose_aa)
    time.sleep(WAIT_AFTER_MOVE_S)


# =========================================================
# ZED helpers
# =========================================================
def create_zed():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 if USE_HD720 else sl.RESOLUTION.HD1080
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = DEPTH_MODE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = COORDINATE_SYSTEM

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {status}")

    return zed


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 78)
    print("ZED Wrist Detect -> xArm Move Once")
    print("=" * 78)
    print("Controls:")
    print("  q -> quit")
    print("  r -> reset tracking / filters / static detector")
    print()

    zed = None
    arm = None

    wrist_buffer = deque(maxlen=MEDIAN_WINDOW)
    filtered_cam = None
    filtered_base = None

    static_anchor = None
    static_start_time = None
    target_ready = False
    target_base = None

    last_print_time = 0.0
    task_finished = False

    try:
        arm = connect_arm(ROBOT_IP)

        startup_pose_aa = get_robot_pose_aa(arm)
        startup_orientation_rvec = startup_pose_aa[3:6].copy()

        print("Startup TCP axis-angle pose:")
        print(startup_pose_aa)
        print("Using fixed orientation rvec (rad):", startup_orientation_rvec)

        zed = create_zed()

        positional_tracking_params = sl.PositionalTrackingParameters()
        err = zed.enable_positional_tracking(positional_tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable positional tracking: {err}")

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        body_params.detection_model = BODY_MODEL
        body_params.body_format = BODY_FORMAT

        err = zed.enable_body_tracking(body_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable body tracking: {err}")

        body_runtime = sl.BodyTrackingRuntimeParameters()
        body_runtime.detection_confidence_threshold = DETECTION_CONF_THRESH

        runtime = sl.RuntimeParameters()
        image_zed = sl.Mat()
        bodies = sl.Bodies()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)

        while True:
            if task_finished:
                print("[Task Finished] One detection + one move completed. Exiting program.")
                break

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_bodies(bodies, body_runtime)

            frame = image_zed.get_data()
            vis = to_bgr(frame)
            if vis is None:
                continue

            lines = []
            wrist_found = False

            # -------------------------------------------------
            # Single-person test logic:
            # use the FIRST body whose right wrist is valid
            # -------------------------------------------------
            body = None
            p2d = None
            p3d = None

            for candidate in bodies.body_list:
                keypoints_2d = candidate.keypoint_2d
                keypoints_3d = candidate.keypoint

                if len(keypoints_2d) <= RIGHT_WRIST_IDX or len(keypoints_3d) <= RIGHT_WRIST_IDX:
                    continue

                candidate_p3d = np.array(keypoints_3d[RIGHT_WRIST_IDX], dtype=np.float64)
                if not is_valid_point(candidate_p3d):
                    continue

                candidate_p2d = keypoints_2d[RIGHT_WRIST_IDX]

                # Whoever first satisfies the condition is used
                body = candidate
                p2d = candidate_p2d
                p3d = candidate_p3d
                break

            if body is not None:
                if filtered_cam is not None:
                    jump = np.linalg.norm(p3d - filtered_cam)
                    if jump > MAX_JUMP_M:
                        p3d = None

                if p3d is not None and is_valid_point(p3d):
                    wrist_found = True

                    x_px, y_px = int(p2d[0]), int(p2d[1])
                    cv2.circle(vis, (x_px, y_px), 8, (0, 255, 0), -1)
                    cv2.putText(
                        vis,
                        f"{USE_KEYPOINT_NAME} (ID {body.id})",
                        (x_px + 10, y_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    wrist_buffer.append(p3d)
                    median_cam = median_filter_point(wrist_buffer)
                    filtered_cam = ema_update(filtered_cam, median_cam, EMA_ALPHA)

                    filtered_base = cam_point_to_base(T_cam2base, filtered_cam)

                    now = time.time()

                    # Static detector
                    if static_anchor is None:
                        static_anchor = filtered_base.copy()
                        static_start_time = now
                        target_ready = False
                        target_base = None
                    else:
                        dist = np.linalg.norm(filtered_base - static_anchor)
                        if dist <= STATIC_RADIUS_M:
                            if (now - static_start_time) >= STATIC_TIME_S:
                                target_ready = True
                                target_base = filtered_base + WRIST_TARGET_OFFSET_M
                                target_base = clamp_workspace(target_base)
                        else:
                            static_anchor = filtered_base.copy()
                            static_start_time = now
                            target_ready = False
                            target_base = None

                    static_elapsed = 0.0 if static_start_time is None else (now - static_start_time)

                    # Once target is ready -> move once -> finish task
                    if target_ready and target_base is not None:
                        print("[Target Locked] Stable wrist detected.")
                        print("[Target Base m]:", target_base)

                        if workspace_ok(target_base):
                            print("[Move] Moving xArm to target...")
                            move_target_pose(
                                arm=arm,
                                target_base_xyz_m=target_base,
                                fixed_orientation_rvec=startup_orientation_rvec,
                            )
                            print("[Done] Robot reached target.")
                            task_finished = True
                        else:
                            print("[Warning] Target out of workspace even after clamp.")
                            task_finished = True

                    status_text = "Detecting steady wrist..."
                    if target_ready:
                        status_text = "TARGET LOCKED -> MOVING"

                    lines = [
                        f"Tracked body id   : {body.id}",
                        f"Camera filt (m)   : x={filtered_cam[0]: .3f}, y={filtered_cam[1]: .3f}, z={filtered_cam[2]: .3f}",
                        f"Base filt (m)     : x={filtered_base[0]: .3f}, y={filtered_base[1]: .3f}, z={filtered_base[2]: .3f}",
                        f"Static elapsed    : {static_elapsed: .2f} / {STATIC_TIME_S:.2f} s",
                        f"Static radius     : {STATIC_RADIUS_M:.3f} m",
                        f"Status            : {status_text}",
                    ]

                    if target_base is not None:
                        lines.append(
                            f"Target (m)        : x={target_base[0]: .3f}, y={target_base[1]: .3f}, z={target_base[2]: .3f}"
                        )
                        lines.append(
                            f"Workspace check   : {'OK' if workspace_ok(target_base) else 'CLAMPED'}"
                        )

                    if now - last_print_time > PRINT_INTERVAL_S:
                        print("-" * 60)
                        print(f"Tracked body id    : {body.id}")
                        print(f"Camera filt (m)    : {filtered_cam}")
                        print(f"Base filt (m)      : {filtered_base}")
                        if target_base is not None:
                            print(f"Target (m)         : {target_base}")
                            print(f"Target ready       : {target_ready}")
                        last_print_time = now

            if not wrist_found:
                lines = [
                    "Right wrist not detected / invalid / rejected",
                    "Tracked body id   : None",
                    "Tips:",
                    "- face camera more directly",
                    "- keep wrist inside image",
                    "- keep wrist at moderate depth",
                    "- press 'r' to reset filters",
                ]

                static_anchor = None
                static_start_time = None
                target_ready = False
                target_base = None

            draw_text_block(vis, lines, org=(20, 35), dy=28, color=(0, 255, 0))

            if DISPLAY_SCALE != 1.0:
                vis_show = cv2.resize(
                    vis,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                vis_show = vis

            cv2.imshow(WINDOW_NAME, vis_show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                wrist_buffer.clear()
                filtered_cam = None
                filtered_base = None
                static_anchor = None
                static_start_time = None
                target_ready = False
                target_base = None
                print("Tracking, filters, and static detector reset.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
    finally:
        try:
            if zed is not None:
                zed.disable_body_tracking()
        except Exception:
            pass

        try:
            if zed is not None:
                zed.disable_positional_tracking()
        except Exception:
            pass

        try:
            if zed is not None:
                zed.close()
        except Exception:
            pass

        try:
            if arm is not None:
                arm.disconnect()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()