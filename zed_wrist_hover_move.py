#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zed_wrist_hover_move.py

Workflow:
    1. Detect and track the user's right wrist with ZED2
    2. Convert wrist point from camera frame to robot base frame using T_cam2base
    3. Filter the wrist position
    4. If the filtered wrist stays sufficiently still for 2 seconds,
       generate a hover target above the wrist in base frame
    5. Only when user presses 'g', move xArm slowly to that hover target

Notes:
    - This script moves to a hover point above the wrist, not to the wrist itself
    - It preserves a fixed tool orientation captured at startup
    - Motion is limited by workspace checks and low speed
"""

import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl
from xarm.wrapper import XArmAPI


# =========================================================
# Hard-coded hand-eye result (camera -> base)
# from your current best_result = HORAUD
# =========================================================
T_cam2base = np.array([
    [-0.691532644,  0.326676329, -0.644255523,  0.853619759],
    [ 0.722295719,  0.302279846, -0.622025553,  0.707727934],
    [-0.008455565, -0.895493981, -0.444993295,  0.586156444],
    [ 0.0,          0.0,          0.0,          1.0        ],
], dtype=np.float64)

# =========================================================
# User settings
# =========================================================
ROBOT_IP = "192.168.1.225"
USE_RADIANS = True

WINDOW_NAME = "ZED Wrist Hover Target + Confirmed xArm Move"

# BODY_34 right wrist index
RIGHT_WRIST_IDX = 14
USE_KEYPOINT_NAME = "RIGHT_WRIST"

# ZED settings
USE_HD720 = True
CAMERA_FPS = 30
COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.IMAGE
DEPTH_MODE = sl.DEPTH_MODE.ULTRA
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 2.00

# Body tracking
BODY_MODEL = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
BODY_FORMAT = sl.BODY_FORMAT.BODY_34
DETECTION_CONF_THRESH = 50

# Filtering
MEDIAN_WINDOW = 5
EMA_ALPHA = 0.35
MAX_JUMP_M = 0.25

# Static detection
STATIC_TIME_S = 2.0
STATIC_RADIUS_M = 0.02   # wrist filtered point must stay within 2 cm ball

# Hover target in robot base frame
WRIST_HOVER_OFFSET_M = np.array([0.0, 0.0, 0.20], dtype=np.float64)

# Robot motion
MOVE_SPEED_MM_S = 40
MOVE_ACC_MM_S2 = 100
WAIT_AFTER_MOVE_S = 0.5

# Fixed orientation captured at startup from current robot TCP axis-angle
# We will keep this orientation during all hover moves
USE_STARTUP_ORIENTATION = True

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


def choose_body_to_track(body_list, locked_id):
    if not body_list:
        return None, locked_id

    if locked_id is not None:
        for body in body_list:
            if body.id == locked_id:
                return body, locked_id

    best_body = None
    best_depth = 1e9

    for body in body_list:
        keypoints_3d = body.keypoint
        if len(keypoints_3d) <= RIGHT_WRIST_IDX:
            continue

        p3d = keypoints_3d[RIGHT_WRIST_IDX]
        if not is_valid_point(p3d):
            continue

        z = float(p3d[2])
        if z < best_depth:
            best_depth = z
            best_body = body

    if best_body is None:
        return None, None

    return best_body, best_body.id


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


def move_hover_pose(arm, target_base_xyz_m, fixed_orientation_rvec):
    """
    Move TCP to hover point in base frame.
    Position is in meters; xArm expects mm.
    Orientation is axis-angle rotation vector in radians.
    """
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


def tune_camera(zed):
    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)
    except Exception:
        pass


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 78)
    print("ZED Wrist Hover Target + Confirmed xArm Move")
    print("=" * 78)
    print("Controls:")
    print("  q -> quit")
    print("  r -> reset tracking / filters / static detector")
    print("  g -> move to current READY hover target")
    print()

    zed = None
    arm = None

    locked_body_id = None
    wrist_buffer = deque(maxlen=MEDIAN_WINDOW)
    filtered_cam = None
    filtered_base = None

    static_anchor = None
    static_start_time = None
    target_ready = False
    hover_target_base = None

    last_print_time = 0.0

    try:
        arm = connect_arm(ROBOT_IP)

        startup_pose_aa = get_robot_pose_aa(arm)
        startup_orientation_rvec = startup_pose_aa[3:6].copy()

        print("Startup TCP axis-angle pose:")
        print(startup_pose_aa)
        print("Using fixed orientation rvec (rad):", startup_orientation_rvec)

        zed = create_zed()
        tune_camera(zed)

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

            body, locked_body_id = choose_body_to_track(bodies.body_list, locked_body_id)

            if body is not None:
                keypoints_2d = body.keypoint_2d
                keypoints_3d = body.keypoint

                if len(keypoints_2d) > RIGHT_WRIST_IDX and len(keypoints_3d) > RIGHT_WRIST_IDX:
                    p2d = keypoints_2d[RIGHT_WRIST_IDX]
                    p3d = np.array(keypoints_3d[RIGHT_WRIST_IDX], dtype=np.float64)

                    if is_valid_point(p3d):
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

                            # Filter in camera frame
                            wrist_buffer.append(p3d)
                            median_cam = median_filter_point(wrist_buffer)
                            filtered_cam = ema_update(filtered_cam, median_cam, EMA_ALPHA)

                            # Convert to base frame after filtering
                            filtered_base = cam_point_to_base(T_cam2base, filtered_cam)

                            # Static detector in base frame
                            now = time.time()
                            if static_anchor is None:
                                static_anchor = filtered_base.copy()
                                static_start_time = now
                                target_ready = False
                                hover_target_base = None
                            else:
                                dist = np.linalg.norm(filtered_base - static_anchor)
                                if dist <= STATIC_RADIUS_M:
                                    if (now - static_start_time) >= STATIC_TIME_S:
                                        target_ready = True
                                        hover_target_base = filtered_base + WRIST_HOVER_OFFSET_M
                                        hover_target_base = clamp_workspace(hover_target_base)
                                else:
                                    static_anchor = filtered_base.copy()
                                    static_start_time = now
                                    target_ready = False
                                    hover_target_base = None

                            static_elapsed = 0.0 if static_start_time is None else (now - static_start_time)

                            if target_ready and hover_target_base is not None:
                                cv2.circle(vis, (x_px, y_px), 14, (0, 255, 255), 2)
                                ready_text = "READY: press 'g' to move to hover target"
                            else:
                                ready_text = "Waiting for wrist to stay still 2.0 s"

                            lines = [
                                f"Tracked body id   : {body.id}",
                                f"Camera filt (m)   : x={filtered_cam[0]: .3f}, y={filtered_cam[1]: .3f}, z={filtered_cam[2]: .3f}",
                                f"Base filt (m)     : x={filtered_base[0]: .3f}, y={filtered_base[1]: .3f}, z={filtered_base[2]: .3f}",
                                f"Static elapsed    : {static_elapsed: .2f} / {STATIC_TIME_S:.2f} s",
                                f"Static radius     : {STATIC_RADIUS_M:.3f} m",
                                ready_text,
                            ]

                            if hover_target_base is not None:
                                ok = workspace_ok(hover_target_base)
                                lines.append(
                                    f"Hover target (m)  : x={hover_target_base[0]: .3f}, y={hover_target_base[1]: .3f}, z={hover_target_base[2]: .3f}"
                                )
                                lines.append(f"Workspace check    : {'OK' if ok else 'CLAMPED'}")

                            if now - last_print_time > PRINT_INTERVAL_S:
                                print("-" * 60)
                                print(f"Tracked body id    : {body.id}")
                                print(f"Camera filt (m)    : {filtered_cam}")
                                print(f"Base filt (m)      : {filtered_base}")
                                if hover_target_base is not None:
                                    print(f"Hover target (m)   : {hover_target_base}")
                                    print(f"Target ready       : {target_ready}")
                                last_print_time = now

            if not wrist_found:
                lines = [
                    "Right wrist not detected / invalid / rejected",
                    f"Tracked body id   : {locked_body_id}",
                    "Tips:",
                    "- face camera more directly",
                    "- keep wrist inside image",
                    "- keep wrist at moderate depth",
                    "- press 'r' if tracking locked wrong",
                ]

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
                locked_body_id = None
                wrist_buffer.clear()
                filtered_cam = None
                filtered_base = None
                static_anchor = None
                static_start_time = None
                target_ready = False
                hover_target_base = None
                print("Tracking, filters, and static detector reset.")

            elif key == ord("g"):
                if not target_ready or hover_target_base is None:
                    print("[Skip] Hover target is not ready yet.")
                    continue

                if not workspace_ok(hover_target_base):
                    print("[Skip] Hover target outside workspace.")
                    continue

                print("[Move] Moving to hover target (m):", hover_target_base)
                move_hover_pose(
                    arm=arm,
                    target_base_xyz_m=hover_target_base,
                    fixed_orientation_rvec=startup_orientation_rvec,
                )
                print("[Done] Reached hover target.")

                # Force reacquire after move
                static_anchor = None
                static_start_time = None
                target_ready = False
                hover_target_base = None

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
