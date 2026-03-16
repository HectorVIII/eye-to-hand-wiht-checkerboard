#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_robot_poses.py

Manual xArm pose collection with robust live ZED2 checkerboard detection.

Workflow:
    1. Put xArm into manual / drag mode using your normal workflow.
    2. Move robot by hand until the checkerboard is fully visible and stably detected.
    3. Press 's' to save the current robot pose.
    4. Repeat until enough poses are collected.
    5. Press 'q' to quit.

This script saves robot poses only.
It does not yet save full calibration samples.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyzed.sl as sl
from xarm.wrapper import XArmAPI


# =========================================================
# User settings
# =========================================================
ROBOT_IP = "192.168.1.225"
SAVE_FILE = Path("robot_poses_20.json")

USE_RADIANS = True
TARGET_NUM_POSES = 20

# Your current board setting
# Keep this as you requested
PATTERNS_TO_TEST = [(6, 7)]

USE_HD720 = True

WINDOW_NAME = "ZED2 Checkerboard Preview"

# How many consecutive detected frames are required to call it stable
STABLE_DETECTION_FRAMES = 5

# Display scale for a larger window image
DISPLAY_SCALE = 1.4

# Read pose a little after key press
READ_STABLE_DELAY_S = 0.15


# =========================================================
# General helpers
# =========================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def round_list(values, digits=6):
    return [round(float(v), digits) for v in values]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {path.resolve()}")


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

    code_state, state = arm.get_state()
    code_err, err_warn = arm.get_err_warn_code()

    print("Connected.")
    if code_state == 0:
        print(f"Robot state: {state}")
    if code_err == 0:
        print(f"Error/Warning: {err_warn}")

    return arm


def get_robot_state(arm, use_radians=True):
    code_rpy, pose_rpy = arm.get_position(is_radian=use_radians)
    code_aa, pose_aa = arm.get_position_aa(is_radian=use_radians)
    code_j, joints = arm.get_servo_angle(is_radian=use_radians)

    if code_rpy != 0:
        raise RuntimeError(f"Failed to read TCP RPY pose, code={code_rpy}")
    if code_aa != 0:
        raise RuntimeError(f"Failed to read TCP axis-angle pose, code={code_aa}")
    if code_j != 0:
        raise RuntimeError(f"Failed to read joint angles, code={code_j}")

    return pose_rpy, pose_aa, joints


def build_pose_record(idx, pose_rpy, pose_aa, joints):
    return {
        "id": idx,
        "timestamp": now_str(),
        "tcp_pose_rpy": round_list(pose_rpy, 6),
        "tcp_pose_aa": round_list(pose_aa, 6),
        "joint_angles": round_list(joints, 6),
        "angle_unit": "radian" if USE_RADIANS else "degree",
        "position_unit": "mm",
    }


# =========================================================
# ZED helpers
# =========================================================
def create_zed():
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720 if USE_HD720 else sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED2: {status}")

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()

    return zed, runtime, image_mat


def set_zed_camera_params(zed):
    """
    Optional camera tuning for more stable checkerboard detection.
    If any setting fails on your SDK version, the exception is ignored.
    """
    try:
        # Turn off auto exposure / auto white balance if supported
        zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
    except Exception:
        pass

    # You can tune these if needed
    # Safe defaults; if unsupported, they will be ignored
    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 4)
    except Exception:
        pass


# =========================================================
# Checkerboard detection helpers
# =========================================================
def to_bgr(frame):
    if frame is None:
        return None

    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def preprocess_variants(bgr):
    """
    Generate several image variants for robust checkerboard detection.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    variants = []

    # Original gray
    variants.append(("gray", gray))

    # Histogram equalization
    eq = cv2.equalizeHist(gray)
    variants.append(("equalizeHist", eq))

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(("clahe", clahe_img))

    # Slight blur after CLAHE
    blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    variants.append(("clahe_blur", blur))

    # Upscaled version for small / far board
    up = cv2.resize(clahe_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    variants.append(("clahe_up1.5", up))

    return variants


def try_detect_sb(gray_img, pattern_size):
    """
    Robust checkerboard detection using findChessboardCornersSB.
    """
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    found, corners = cv2.findChessboardCornersSB(gray_img, pattern_size, flags=flags)
    return found, corners


def scale_corners_back(corners, scale):
    """
    If detection was run on an upscaled image, map corners back to original image scale.
    """
    if scale == 1.0:
        return corners

    corners_scaled = corners.copy()
    corners_scaled[:, 0, 0] /= scale
    corners_scaled[:, 0, 1] /= scale
    return corners_scaled


def detect_checkerboard_robust(bgr, patterns):
    """
    Try robust detection on multiple preprocessed image variants.

    Returns:
        found: bool
        pattern_size: tuple or None
        corners: ndarray or None
        vis: image for visualization
        method_name: str
    """
    vis = bgr.copy()
    variants = preprocess_variants(bgr)

    for pattern_size in patterns:
        for method_name, img_variant in variants:
            scale = 1.0
            if method_name == "clahe_up1.5":
                scale = 1.5

            found, corners = try_detect_sb(img_variant, pattern_size)
            if found:
                corners = scale_corners_back(corners, scale)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                return True, pattern_size, corners, vis, method_name

    return False, None, None, vis, "none"


def draw_overlay(vis, found, pattern_size, method_name, stable_count, num_saved, target_num):
    h, w = vis.shape[:2]

    stable = stable_count >= STABLE_DETECTION_FRAMES

    if found and stable:
        status_text = "CHECKERBOARD: STABLE"
        status_color = (0, 220, 0)
    elif found:
        status_text = "CHECKERBOARD: DETECTED (not yet stable)"
        status_color = (0, 255, 255)
    else:
        status_text = "CHECKERBOARD: NOT DETECTED"
        status_color = (0, 0, 255)

    cv2.putText(vis, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)

    if pattern_size is not None:
        cv2.putText(
            vis,
            f"Pattern: {pattern_size[0]} x {pattern_size[1]} inner corners",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"Detection method: {method_name}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        f"Stable frames: {stable_count} / {STABLE_DETECTION_FRAMES}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        f"Saved poses: {num_saved} / {target_num}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        "Move robot by hand. Press 's' to save current robot pose.",
        (20, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        "Tip: save only when checkerboard is fully visible and stable.",
        (20, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 72)
    print("xArm Manual Pose Collector with Robust ZED2 Checkerboard Detection")
    print("=" * 72)
    print(f"Robot IP          : {ROBOT_IP}")
    print(f"Save file         : {SAVE_FILE}")
    print(f"Angle unit        : {'radian' if USE_RADIANS else 'degree'}")
    print(f"Target pose count : {TARGET_NUM_POSES}")
    print(f"Patterns to test  : {PATTERNS_TO_TEST}")
    print()

    print("Instructions:")
    print("1. Put robot into manual / drag mode.")
    print("2. Move robot until checkerboard is fully visible and stably detected.")
    print("3. Press 's' to save the current robot pose.")
    print("4. Press 'q' to quit.")
    print()

    arm = None
    zed = None
    records = []
    stable_count = 0

    try:
        arm = connect_arm(ROBOT_IP)

        zed, runtime, image_mat = create_zed()
        set_zed_camera_params(zed)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)

        while True:
            grab_status = zed.grab(runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                print(f"[Warning] ZED grab failed: {grab_status}")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            frame = image_mat.get_data()
            frame_bgr = to_bgr(frame)

            if frame_bgr is None:
                continue

            found, pattern_size, corners, vis, method_name = detect_checkerboard_robust(
                frame_bgr, PATTERNS_TO_TEST
            )

            if found:
                stable_count += 1
            else:
                stable_count = 0

            vis = draw_overlay(
                vis=vis,
                found=found,
                pattern_size=pattern_size,
                method_name=method_name,
                stable_count=stable_count,
                num_saved=len(records),
                target_num=TARGET_NUM_POSES,
            )

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

            elif key == ord("s"):
                if not found:
                    print("[Skip] Checkerboard not detected.")
                    continue

                if stable_count < STABLE_DETECTION_FRAMES:
                    print(f"[Skip] Detection not stable yet: {stable_count}/{STABLE_DETECTION_FRAMES}")
                    continue

                time.sleep(READ_STABLE_DELAY_S)

                try:
                    pose_rpy, pose_aa, joints = get_robot_state(
                        arm,
                        use_radians=USE_RADIANS
                    )
                except Exception as e:
                    print(f"[Error] Failed to read robot state: {e}")
                    continue

                idx = len(records)
                record = build_pose_record(idx, pose_rpy, pose_aa, joints)
                records.append(record)

                print("-" * 60)
                print(f"[Saved pose #{idx}]")
                print(f"  tcp_pose_rpy : {record['tcp_pose_rpy']}")
                print(f"  tcp_pose_aa  : {record['tcp_pose_aa']}")
                print(f"  joint_angles : {record['joint_angles']}")
                print(f"  detected pattern: {pattern_size}")
                print(f"  detection method: {method_name}")

                output = {
                    "meta": {
                        "created_at": now_str(),
                        "robot_ip": ROBOT_IP,
                        "robot_model": "xArm7",
                        "use_radians": USE_RADIANS,
                        "angle_unit": "radian" if USE_RADIANS else "degree",
                        "position_unit": "mm",
                        "description": "Manually collected robot poses with robust live ZED2 checkerboard preview",
                        "num_poses": len(records),
                    },
                    "poses": records,
                }
                save_json(SAVE_FILE, output)

        if len(records) == 0:
            print("\nNo poses collected.")
            return

        output = {
            "meta": {
                "created_at": now_str(),
                "robot_ip": ROBOT_IP,
                "robot_model": "xArm7",
                "use_radians": USE_RADIANS,
                "angle_unit": "radian" if USE_RADIANS else "degree",
                "position_unit": "mm",
                "description": "Manually collected robot poses with robust live ZED2 checkerboard preview",
                "num_poses": len(records),
            },
            "poses": records,
        }
        save_json(SAVE_FILE, output)

        print("\nCollection finished successfully.")
        print(f"Total poses saved: {len(records)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        sys.exit(1)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if zed is not None:
            try:
                zed.close()
                print("ZED closed.")
            except Exception:
                pass

        if arm is not None:
            try:
                arm.disconnect()
                print("Robot disconnected.")
            except Exception:
                pass


if __name__ == "__main__":
    main()