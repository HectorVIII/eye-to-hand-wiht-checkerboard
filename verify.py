#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zed_wrist_verify_cam2base.py

Use ZED body tracking wrist point to roughly verify T_cam2base.

Important:
    - This is only a rough verification tool.
    - Wrist keypoints are noisier than checkerboard corners.
    - Use this to check whether the transform is roughly reasonable,
      not as a final millimeter-level validation.
"""

import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl


# =========================================================
# Hard-coded hand-eye result
# camera -> base
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
WINDOW_NAME = "ZED2 Wrist -> Robot Base Verification"

# BODY_34 right-side hand indices
# From Stereolabs BODY_34:
# RIGHT_WRIST = 14, RIGHT_HAND = 15, RIGHT_HANDTIP = 16
USE_KEYPOINT_NAME = "RIGHT_WRIST"
RIGHT_WRIST_IDX = 14

# Camera / tracking
USE_HD720 = True
CAMERA_FPS = 30

# Prefer IMAGE coordinate system to stay aligned with computer vision / OpenCV conventions
COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.IMAGE

# Depth settings
DEPTH_MODE = sl.DEPTH_MODE.ULTRA
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 2.00

# Body tracking model
# HUMAN_BODY_ACCURATE is highest accuracy; if too slow, change to HUMAN_BODY_MEDIUM
BODY_MODEL = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
BODY_FORMAT = sl.BODY_FORMAT.BODY_34
DETECTION_CONF_THRESH = 50

# Filtering
MEDIAN_WINDOW = 5
EMA_ALPHA = 0.35          # larger -> follows faster, smaller -> smoother
MAX_JUMP_M = 0.25         # reject sudden unrealistic jump

# UI / printing
DISPLAY_SCALE = 1.5
PRINT_INTERVAL_S = 0.3


# =========================================================
# Math helpers
# =========================================================
def cam_point_to_base(T_cam2base, p_cam):
    """Convert [x, y, z] in camera frame to base frame."""
    p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    p_base_h = T_cam2base @ p_cam_h
    return p_base_h[:3]


def draw_text_block(img, lines, org=(20, 30), dy=30, color=(0, 255, 0)):
    x, y = org
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        y += dy


def to_bgr(frame):
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def is_valid_point(p):
    """Basic validity checks for 3D point."""
    if p is None:
        return False
    if not np.all(np.isfinite(p)):
        return False

    z = float(p[2])
    if z < DEPTH_MIN_M or z > DEPTH_MAX_M:
        return False

    return True


def median_filter_point(buffer):
    """Return component-wise median of a deque of 3D points."""
    if len(buffer) == 0:
        return None
    arr = np.array(buffer, dtype=np.float64)
    return np.median(arr, axis=0)


def ema_update(prev, current, alpha):
    """Exponential moving average."""
    if prev is None:
        return current.copy()
    return alpha * current + (1.0 - alpha) * prev


def choose_body_to_track(body_list, locked_id):
    """
    Keep tracking the same person if possible.
    If no locked_id yet, choose the closest valid body.
    """
    if not body_list:
        return None, locked_id

    # First try to keep the same ID
    if locked_id is not None:
        for body in body_list:
            if body.id == locked_id:
                return body, locked_id

    # Otherwise choose the closest body with a valid wrist point
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
    """
    Try to reduce exposure/white balance fluctuations.
    Safe best-effort only.
    """
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
    print("=" * 72)
    print("ZED Wrist -> Robot Base Verification")
    print("=" * 72)
    print("Loaded T_cam2base:")
    print(T_cam2base)
    print()
    print("Controls:")
    print("  q -> quit")
    print("  r -> reset tracked body and filters")
    print()

    zed = None
    locked_body_id = None
    wrist_buffer = deque(maxlen=MEDIAN_WINDOW)
    filtered_cam = None
    last_print_time = 0.0

    try:
        zed = create_zed()
        tune_camera(zed)

        # Enable positional tracking because body tracking with tracking uses it
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

            # Body selection
            body, locked_body_id = choose_body_to_track(bodies.body_list, locked_body_id)

            if body is not None:
                keypoints_2d = body.keypoint_2d
                keypoints_3d = body.keypoint

                if len(keypoints_2d) > RIGHT_WRIST_IDX and len(keypoints_3d) > RIGHT_WRIST_IDX:
                    p2d = keypoints_2d[RIGHT_WRIST_IDX]
                    p3d = np.array(keypoints_3d[RIGHT_WRIST_IDX], dtype=np.float64)

                    if is_valid_point(p3d):
                        # Reject sudden jump against previous filtered point
                        if filtered_cam is not None:
                            jump = np.linalg.norm(p3d - filtered_cam)
                            if jump > MAX_JUMP_M:
                                p3d = None

                        if p3d is not None and is_valid_point(p3d):
                            wrist_found = True

                            # 2D draw
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

                            # Filtering: median then EMA
                            wrist_buffer.append(p3d)
                            median_cam = median_filter_point(wrist_buffer)
                            filtered_cam = ema_update(filtered_cam, median_cam, EMA_ALPHA)

                            p_cam_raw = p3d
                            p_cam_flt = filtered_cam
                            p_base_raw = cam_point_to_base(T_cam2base, p_cam_raw)
                            p_base_flt = cam_point_to_base(T_cam2base, p_cam_flt)

                            lines = [
                                f"Tracked body id : {body.id}",
                                f"Keypoint         : {USE_KEYPOINT_NAME}",
                                f"Camera raw (m)   : x={p_cam_raw[0]: .3f}, y={p_cam_raw[1]: .3f}, z={p_cam_raw[2]: .3f}",
                                f"Camera filt (m)  : x={p_cam_flt[0]: .3f}, y={p_cam_flt[1]: .3f}, z={p_cam_flt[2]: .3f}",
                                f"Base raw (m)     : x={p_base_raw[0]: .3f}, y={p_base_raw[1]: .3f}, z={p_base_raw[2]: .3f}",
                                f"Base filt (m)    : x={p_base_flt[0]: .3f}, y={p_base_flt[1]: .3f}, z={p_base_flt[2]: .3f}",
                                f"Median win       : {len(wrist_buffer)} / {MEDIAN_WINDOW}",
                                f"EMA alpha        : {EMA_ALPHA:.2f}",
                            ]

                            now = time.time()
                            if now - last_print_time > PRINT_INTERVAL_S:
                                print("-" * 60)
                                print(f"Tracked body id        : {body.id}")
                                print(f"{USE_KEYPOINT_NAME} raw cam  : {p_cam_raw}")
                                print(f"{USE_KEYPOINT_NAME} filt cam : {p_cam_flt}")
                                print(f"{USE_KEYPOINT_NAME} raw base : {p_base_raw}")
                                print(f"{USE_KEYPOINT_NAME} filt base: {p_base_flt}")
                                last_print_time = now

            if not wrist_found:
                lines = [
                    "Right wrist not detected / invalid / rejected",
                    f"Tracked body id : {locked_body_id}",
                    "Tips:",
                    "- face camera more directly",
                    "- keep wrist inside image",
                    "- keep distance moderate",
                    "- press 'r' if tracking locked to wrong body",
                ]

            draw_text_block(vis, lines, org=(20, 35), dy=30, color=(0, 255, 0))

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
                print("Tracking and filters reset.")

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

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()