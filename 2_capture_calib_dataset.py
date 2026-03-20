#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
capture_calib_dataset.py

Read pre-collected robot poses from robot_poses_30.json, move the xArm
automatically by JOINT ANGLES, detect the checkerboard with ZED2,
estimate target pose in camera frame, and save a full hand-eye dataset.

Output:
    calib_dataset/
        images/
            sample_000.png
            sample_001.png
            ...
        samples.json

Each saved sample contains:
    - robot pose (RPY, axis-angle, joints)
    - R_gripper2base, t_gripper2base
    - R_target2cam, t_target2cam
    - image path
    - reprojection error
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

POSE_FILE = Path("robot_poses_30.json")
SAVE_DIR = Path("calib_dataset")
IMAGE_DIR = SAVE_DIR / "images"
SAMPLES_FILE = SAVE_DIR / "samples.json"

USE_RADIANS = True

# Zivid 7x8 squares -> 6x7 inner corners
PATTERN_SIZE = (6, 7)
SQUARE_SIZE_M = 0.03

# ZED
USE_HD720 = True
WINDOW_NAME = "Capture Calibration Dataset"

# =========================================================
# Stable joint-space motion parameters
# =========================================================
JOINT_SPEED_RAD_S = 0.25       # slower and safer
JOINT_ACC_RAD_S2 = 0.5         # slower and safer
WAIT_AFTER_MOVE_S = 2.5

# Optional safety waypoint:
# if enabled, robot first goes to a safe joint pose, then to target joint pose
USE_SAFE_INTERMEDIATE_JOINT = False

# Example placeholder safe joint pose (edit if you want to use it)
# Must match your robot and your use_radians setting
SAFE_INTERMEDIATE_JOINT = [0.0, -0.35, -0.35, 0.0, 0.70, 0.0, 0.0]

# If any one joint jump is too large, print warning
MAX_SINGLE_JOINT_STEP_RAD = 1.2

# Detection / capture behavior
STABLE_DETECTION_FRAMES = 5
AUTO_CAPTURE_WHEN_STABLE = True
AUTO_CAPTURE_DELAY_S = 0.5
MAX_WAIT_PER_POSE_S = 20.0

# Keyboard
# q: quit
# n: skip pose
# s: save immediately if checkerboard currently detected

# Visualization
DISPLAY_SCALE = 1.3


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
    print(f"Saved JSON: {path.resolve()}")


def load_pose_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "poses" not in data:
        raise ValueError("Pose file does not contain key 'poses'")

    poses = data["poses"]
    if not isinstance(poses, list) or len(poses) == 0:
        raise ValueError("Pose file contains no poses")

    for i, p in enumerate(poses):
        if "joint_angles" not in p:
            raise ValueError(f"Pose #{i} missing 'joint_angles'")
        joints = p["joint_angles"]
        if not isinstance(joints, list) or len(joints) != 7:
            raise ValueError(f"Pose #{i} has invalid 'joint_angles' (must be length 7)")

    return data


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


def ensure_arm_ready(arm):
    try:
        arm.motion_enable(enable=True)
    except Exception:
        pass

    try:
        arm.set_mode(0)
    except Exception:
        pass

    try:
        arm.set_state(0)
    except Exception:
        pass

    time.sleep(0.2)


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


def move_joints(arm, joints, speed=JOINT_SPEED_RAD_S, mvacc=JOINT_ACC_RAD_S2, wait=True):
    """
    Move robot by joint angles instead of TCP pose.
    This is much more stable for replaying manually collected calibration poses.
    """
    ensure_arm_ready(arm)

    code = arm.set_servo_angle(
        angle=joints,
        speed=speed,
        mvacc=mvacc,
        is_radian=USE_RADIANS,
        wait=wait,
    )
    if code != 0:
        raise RuntimeError(f"set_servo_angle failed, code={code}")


def joint_diff_abs(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.abs(a - b)


def print_joint_move_info(current_joints, target_joints):
    diff = joint_diff_abs(current_joints, target_joints)
    max_idx = int(np.argmax(diff))
    max_val = float(diff[max_idx])

    print("Current joints:", [round(float(v), 4) for v in current_joints])
    print("Target  joints:", [round(float(v), 4) for v in target_joints])
    print("Joint abs diff:", [round(float(v), 4) for v in diff])
    print(f"Max joint step : J{max_idx + 1} = {max_val:.4f} rad")

    if max_val > MAX_SINGLE_JOINT_STEP_RAD:
        print("[Warning] Large single-joint jump detected. Consider reordering poses or inserting a safe middle pose.")


def move_joint_stable(arm, target_joints):
    """
    Stable replay:
      1) read current joint state
      2) optionally go to a safe intermediate joint pose
      3) move to target joints
      4) wait a bit after arrival
    """
    _, _, current_joints = get_robot_state(arm, use_radians=USE_RADIANS)
    print_joint_move_info(current_joints, target_joints)

    if USE_SAFE_INTERMEDIATE_JOINT:
        print("Moving to safe intermediate joint pose ...")
        move_joints(arm, SAFE_INTERMEDIATE_JOINT, wait=True)
        time.sleep(0.6)

    print("Moving to target joint pose ...")
    move_joints(arm, target_joints, wait=True)
    time.sleep(WAIT_AFTER_MOVE_S)


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


def get_left_camera_intrinsics(zed):
    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam

    fx = float(calib.fx)
    fy = float(calib.fy)
    cx = float(calib.cx)
    cy = float(calib.cy)

    dist = np.array(calib.disto, dtype=np.float64).reshape(-1, 1)

    camera_matrix = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64
    )

    return camera_matrix, dist


def to_bgr(frame):
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


# =========================================================
# Vision helpers
# =========================================================
def build_object_points(pattern_size, square_size_m):
    """
    Build checkerboard 3D points in target(board) frame.
    pattern_size = (cols, rows) of inner corners.
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def preprocess_variants(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    variants = []
    variants.append(("gray", gray))
    variants.append(("equalizeHist", cv2.equalizeHist(gray)))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(("clahe", clahe_img))

    blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    variants.append(("clahe_blur", blur))

    up = cv2.resize(clahe_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    variants.append(("clahe_up1.5", up))

    return variants


def try_detect_sb(gray_img, pattern_size):
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCornersSB(gray_img, pattern_size, flags=flags)
    return found, corners


def scale_corners_back(corners, scale):
    if scale == 1.0:
        return corners
    out = corners.copy()
    out[:, 0, 0] /= scale
    out[:, 0, 1] /= scale
    return out


def detect_checkerboard_robust(bgr, pattern_size):
    vis = bgr.copy()
    variants = preprocess_variants(bgr)

    for method_name, img_variant in variants:
        scale = 1.5 if method_name == "clahe_up1.5" else 1.0
        found, corners = try_detect_sb(img_variant, pattern_size)
        if found:
            corners = scale_corners_back(corners, scale)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            return True, corners, vis, method_name

    return False, None, vis, "none"


def estimate_target_pose(object_points, image_points, camera_matrix, dist_coeffs):
    """
    Solve target(board) pose in camera frame:
      X_cam = R_target2cam * X_target + t_target2cam
    """
    obj = np.ascontiguousarray(object_points.reshape(-1, 1, 3), dtype=np.float32)
    img = np.ascontiguousarray(image_points.reshape(-1, 1, 2), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")

    R_target2cam, _ = cv2.Rodrigues(rvec)
    t_target2cam = tvec.reshape(3)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)
    img2 = img.reshape(-1, 2)
    reproj_error = float(np.mean(np.linalg.norm(proj - img2, axis=1)))

    return R_target2cam, t_target2cam, reproj_error


def pose_aa_to_transform(pose_aa):
    """
    Convert xArm axis-angle pose [x_mm, y_mm, z_mm, rx, ry, rz]
    into gripper->base transform:
      R_gripper2base, t_gripper2base (meters)
    """
    x_mm, y_mm, z_mm, rx, ry, rz = pose_aa
    rvec = np.array([rx, ry, rz], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return R, t


def draw_overlay(vis, pose_idx, total_poses, found, stable_count, method_name, reproj_error=None):
    h, _ = vis.shape[:2]
    stable = stable_count >= STABLE_DETECTION_FRAMES

    if found and stable:
        text = "CHECKERBOARD: STABLE"
        color = (0, 220, 0)
    elif found:
        text = "CHECKERBOARD: DETECTED"
        color = (0, 255, 255)
    else:
        text = "CHECKERBOARD: NOT DETECTED"
        color = (0, 0, 255)

    cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Pose: {pose_idx + 1} / {total_poses}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Stable frames: {stable_count} / {STABLE_DETECTION_FRAMES}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Method: {method_name}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    if reproj_error is not None:
        cv2.putText(vis, f"Reproj error: {reproj_error:.3f}px", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(vis, "q: quit   n: skip pose   s: save now", (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


# =========================================================
# Sample handling
# =========================================================
def make_sample_record(sample_id, pose_record, pose_rpy, pose_aa, joints,
                       R_gripper2base, t_gripper2base,
                       R_target2cam, t_target2cam,
                       image_path, reproj_error):
    return {
        "id": sample_id,
        "timestamp": now_str(),
        "image_path": str(image_path),
        "pattern_size": list(PATTERN_SIZE),
        "square_size_m": float(SQUARE_SIZE_M),

        "tcp_pose_rpy": round_list(pose_rpy, 6),
        "tcp_pose_aa": round_list(pose_aa, 6),
        "joint_angles": round_list(joints, 6),

        "R_gripper2base": np.asarray(R_gripper2base, dtype=float).round(9).tolist(),
        "t_gripper2base": np.asarray(t_gripper2base, dtype=float).round(9).tolist(),

        "R_target2cam": np.asarray(R_target2cam, dtype=float).round(9).tolist(),
        "t_target2cam": np.asarray(t_target2cam, dtype=float).round(9).tolist(),

        "reproj_error_px": round(float(reproj_error), 6),

        "angle_unit": "radian" if USE_RADIANS else "degree",
        "position_unit": "mm",
    }


def save_samples_json(samples):
    output = {
        "meta": {
            "created_at": now_str(),
            "robot_ip": ROBOT_IP,
            "robot_model": "xArm7",
            "pose_file": str(POSE_FILE),
            "pattern_size": list(PATTERN_SIZE),
            "square_size_m": SQUARE_SIZE_M,
            "use_radians": USE_RADIANS,
            "angle_unit": "radian" if USE_RADIANS else "degree",
            "position_unit": "mm",
            "num_samples": len(samples),
            "motion_type": "joint_space_replay",
        },
        "samples": samples,
    }
    save_json(SAMPLES_FILE, output)


# =========================================================
# Main capture routine
# =========================================================
def main():
    print("=" * 72)
    print("Capture Calibration Dataset (JOINT REPLAY VERSION)")
    print("=" * 72)
    print(f"Pose file     : {POSE_FILE}")
    print(f"Save dir      : {SAVE_DIR}")
    print(f"Pattern size  : {PATTERN_SIZE}")
    print(f"Square size   : {SQUARE_SIZE_M} m")
    print(f"Joint speed   : {JOINT_SPEED_RAD_S} rad/s")
    print(f"Joint acc     : {JOINT_ACC_RAD_S2} rad/s^2")
    print()

    pose_file_data = load_pose_file(POSE_FILE)
    pose_records = pose_file_data["poses"]
    print(f"Loaded {len(pose_records)} poses")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    arm = None
    zed = None
    samples = []

    try:
        arm = connect_arm(ROBOT_IP)

        zed, runtime, image_mat = create_zed()
        set_zed_camera_params(zed)
        camera_matrix, dist_coeffs = get_left_camera_intrinsics(zed)

        print("Camera matrix:")
        print(camera_matrix.ravel())

        object_points = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)

        for pose_idx, pose_record in enumerate(pose_records):
            print("-" * 72)
            print(f"Moving to pose {pose_idx + 1} / {len(pose_records)}")

            target_joints = pose_record["joint_angles"]
            move_joint_stable(arm, target_joints)

            stable_count = 0
            last_found = False
            last_corners = None
            last_method = "none"
            last_reproj_error = None
            capture_start = time.time()
            auto_captured = False

            while True:
                if time.time() - capture_start > MAX_WAIT_PER_POSE_S:
                    print(f"[Skip] Timeout at pose {pose_idx}")
                    break

                grab_status = zed.grab(runtime)
                if grab_status != sl.ERROR_CODE.SUCCESS:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        raise KeyboardInterrupt
                    continue

                zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                frame = image_mat.get_data()
                frame_bgr = to_bgr(frame)
                if frame_bgr is None:
                    continue

                found, corners, vis, method_name = detect_checkerboard_robust(frame_bgr, PATTERN_SIZE)

                if found:
                    try:
                        img_pts = corners.reshape(-1, 2)
                        _, _, reproj_error = estimate_target_pose(
                            object_points=object_points,
                            image_points=img_pts,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                        )
                        stable_count += 1
                        last_found = True
                        last_corners = corners
                        last_method = method_name
                        last_reproj_error = reproj_error
                    except Exception:
                        stable_count = 0
                        last_found = False
                        last_corners = None
                        last_method = "solvePnP_failed"
                        last_reproj_error = None
                else:
                    stable_count = 0
                    last_found = False
                    last_corners = None
                    last_method = method_name
                    last_reproj_error = None

                vis = draw_overlay(
                    vis=vis,
                    pose_idx=pose_idx,
                    total_poses=len(pose_records),
                    found=last_found,
                    stable_count=stable_count,
                    method_name=last_method,
                    reproj_error=last_reproj_error,
                )

                if DISPLAY_SCALE != 1.0:
                    vis_show = cv2.resize(
                        vis, None,
                        fx=DISPLAY_SCALE,
                        fy=DISPLAY_SCALE,
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    vis_show = vis

                cv2.imshow(WINDOW_NAME, vis_show)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    raise KeyboardInterrupt

                if key == ord("n"):
                    print(f"[Skip] User skipped pose {pose_idx}")
                    break

                manual_save = (key == ord("s"))

                if AUTO_CAPTURE_WHEN_STABLE and stable_count >= STABLE_DETECTION_FRAMES and not auto_captured:
                    time.sleep(AUTO_CAPTURE_DELAY_S)
                    auto_captured = True
                    do_save = True
                else:
                    do_save = manual_save

                if do_save:
                    if not last_found or last_corners is None:
                        print("[Skip] Checkerboard not ready.")
                        continue

                    pose_rpy, pose_aa, joints = get_robot_state(arm, use_radians=USE_RADIANS)

                    # To test the unit is mm
                    print(f"[DEBUG] pose_aa raw = {pose_aa}")
                    print(f"[DEBUG] XYZ (should be ~hundreds of mm): x={pose_aa[0]:.2f}, y={pose_aa[1]:.2f}, z={pose_aa[2]:.2f}")

                    R_gripper2base, t_gripper2base = pose_aa_to_transform(pose_aa)

                    # To test the unit is mm
                    print(f"[DEBUG] t_gripper2base (m) = {t_gripper2base}")
    
                    img_pts = last_corners.reshape(-1, 2)
                    R_target2cam, t_target2cam, reproj_error = estimate_target_pose(
                        object_points=object_points,
                        image_points=img_pts,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                    )

                    sample_id = len(samples)
                    image_path = IMAGE_DIR / f"sample_{sample_id:03d}.png"
                    cv2.imwrite(str(image_path), frame_bgr)

                    sample = make_sample_record(
                        sample_id=sample_id,
                        pose_record=pose_record,
                        pose_rpy=pose_rpy,
                        pose_aa=pose_aa,
                        joints=joints,
                        R_gripper2base=R_gripper2base,
                        t_gripper2base=t_gripper2base,
                        R_target2cam=R_target2cam,
                        t_target2cam=t_target2cam,
                        image_path=image_path,
                        reproj_error=reproj_error,
                    )
                    samples.append(sample)
                    save_samples_json(samples)

                    print(f"[Saved] sample_{sample_id:03d}.png")
                    print(f"        reproj_error = {reproj_error:.4f} px")
                    break

        print("=" * 72)
        print("Done.")
        print(f"Saved {len(samples)} samples to {SAMPLES_FILE}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if len(samples) > 0:
            save_samples_json(samples)
            print(f"Partial samples saved: {len(samples)}")
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