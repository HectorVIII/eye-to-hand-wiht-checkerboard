#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
capture_calib_dataset.py

Purpose:
- Read robot poses from robot_poses_20.json
- Move xArm7 to each saved pose
- Open ZED2 camera stream
- Detect checkerboard in real time
- Save successful hand-eye calibration samples

Workflow for each pose:
1. Move robot to target pose
2. Wait for robot to stabilize
3. Show live ZED2 left image
4. Detect checkerboard continuously
5. Press ENTER to save sample when detection looks good
6. Press S to skip current pose
7. Press R to re-open capture loop for current pose
8. Press Q to quit

Saved output:
- calib_dataset/images/sample_XXX.png
- calib_dataset/samples.json
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from xarm.wrapper import XArmAPI


# =========================================================
# User settings
# =========================================================
ROBOT_IP = "192.168.1.225"

POSE_FILE = Path("robot_poses_20.json")
SAVE_DIR = Path("calib_dataset")
IMAGE_DIR = SAVE_DIR / "images"
SAMPLES_FILE = SAVE_DIR / "samples.json"

USE_RADIANS = True

# Motion parameters
MOVE_SPEED_MM_S = 80
MOVE_ACC_MM_S2 = 300
WAIT_AFTER_MOVE = 1.5

# If True, use tcp_pose_aa for robot motion.
# If False, use tcp_pose_rpy.
PREFER_AXIS_ANGLE_MOVE = True

# Checkerboard settings
# Zivid 7x8 squares board -> inner corners = (6, 7)
PATTERN_SIZE = (6, 7)
SQUARE_SIZE_M = 0.03

# ZED settings
USE_HD720 = True
USE_LEFT_IMAGE = True

# Real-time display
WINDOW_NAME = "ZED2 Checkerboard Capture"

# Detection flags
USE_SB_DETECTOR = True   # True: findChessboardCornersSB, False: classic findChessboardCorners
ENABLE_SUBPIX = True

# If True, only allow saving when checkerboard is detected
REQUIRE_VALID_DETECTION_TO_SAVE = True


# =========================================================
# Utility helpers
# =========================================================
def ensure_dirs():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_robot_poses(pose_file: Path):
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")

    with open(pose_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        poses = data
    elif isinstance(data, dict):
        # Support either raw list or dict-wrapped list
        if "poses" in data:
            poses = data["poses"]
        elif "samples" in data:
            poses = data["samples"]
        else:
            raise ValueError("Unsupported JSON structure: expected list or dict with 'poses'/'samples'")
    else:
        raise ValueError("Unsupported pose file format")

    if len(poses) == 0:
        raise ValueError("No poses found in pose file")

    return poses


def load_existing_samples(samples_file: Path):
    if not samples_file.exists():
        return {"samples": []}

    with open(samples_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "samples" not in data:
        data = {"samples": []}

    return data


def save_samples(samples_file: Path, data: dict):
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def rvec_tvec_to_rt(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R, t


def draw_text_block(img, lines, org=(20, 30), line_h=28):
    x, y = org
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def pose_to_rt_from_xarm_tcp_rpy(tcp_pose_rpy, use_radians=True):
    """
    Convert xArm TCP pose [x, y, z, roll, pitch, yaw] to:
    - R_gripper2base
    - t_gripper2base (meters)

    IMPORTANT:
    The Euler order here is assumed to be XYZ intrinsic equivalent to:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    If your final calibration is bad, this Euler convention may need to be changed.
    """
    x_mm, y_mm, z_mm, roll, pitch, yaw = tcp_pose_rpy

    if not use_radians:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    R = Rz @ Ry @ Rx
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return R, t


def pose_to_rt_from_xarm_tcp_aa(tcp_pose_aa, use_radians=True):
    """
    Convert xArm axis-angle TCP pose [x, y, z, rx, ry, rz] to:
    - R_gripper2base
    - t_gripper2base (meters)

    rx, ry, rz is the axis-angle rotation vector.
    """
    x_mm, y_mm, z_mm, rx, ry, rz = tcp_pose_aa

    if not use_radians:
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)

    rvec = np.array([rx, ry, rz], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return R, t


def build_object_points(pattern_size, square_size_m):
    """
    Build checkerboard 3D corner coordinates in board frame.

    pattern_size = (cols, rows) = number of inner corners
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def create_zed():
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720 if USE_HD720 else sl.RESOLUTION.HD1080
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED2: {status}")

    runtime = sl.RuntimeParameters()
    runtime.enable_fill_mode = False
    return zed, runtime


def get_zed_intrinsics(zed):
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters

    if USE_LEFT_IMAGE:
        fx = calib.left_cam.fx
        fy = calib.left_cam.fy
        cx = calib.left_cam.cx
        cy = calib.left_cam.cy
        dist = np.array(calib.left_cam.disto, dtype=np.float64)
    else:
        fx = calib.right_cam.fx
        fy = calib.right_cam.fy
        cx = calib.right_cam.cx
        cy = calib.right_cam.cy
        dist = np.array(calib.right_cam.disto, dtype=np.float64)

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    return K, dist


def grab_bgr_image(zed, runtime):
    mat = sl.Mat()
    view = sl.VIEW.LEFT if USE_LEFT_IMAGE else sl.VIEW.RIGHT

    err = zed.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        return None

    zed.retrieve_image(mat, view)
    img_rgba = mat.get_data()
    if img_rgba is None:
        return None

    # ZED returns BGRA in many Python setups
    if img_rgba.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img_rgba.copy()

    return img_bgr


def detect_checkerboard_pose(img_bgr, pattern_size, square_size_m, K, dist):
    """
    Detect checkerboard and estimate board pose relative to camera:
    returns:
        found: bool
        corners: Nx1x2
        rvec: 3x1
        tvec: 3x1
        reproj_error: float or None
        vis_img: visualization image
    """
    vis = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    found = False
    corners = None

    if USE_SB_DETECTOR:
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
        if found and ENABLE_SUBPIX:
            term = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                50,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    rvec, tvec, reproj_error = None, None, None

    if found:
        objp = build_object_points(pattern_size, square_size_m)

        ok, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if ok:
            # Compute reprojection error
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = np.linalg.norm(corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
            reproj_error = float(np.mean(err))

            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.08, 2)

    return found, corners, rvec, tvec, reproj_error, vis


def connect_xarm(ip: str):
    arm = XArmAPI(ip)
    arm.connect()

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    # Optional but usually helpful
    arm.clean_error()
    arm.clean_warn()

    return arm


def move_robot_to_pose(arm, pose: dict):
    """
    Move robot to a target pose from robot_poses_20.json.

    Supported fields:
    - tcp_pose_aa: [x, y, z, rx, ry, rz]
    - tcp_pose_rpy: [x, y, z, roll, pitch, yaw]
    """
    if PREFER_AXIS_ANGLE_MOVE and "tcp_pose_aa" in pose:
        aa = pose["tcp_pose_aa"]
        if len(aa) != 6:
            raise ValueError(f"Invalid tcp_pose_aa in pose {pose.get('pose_id', 'unknown')}")
        code = arm.set_position_aa(
            aa,
            speed=MOVE_SPEED_MM_S,
            mvacc=MOVE_ACC_MM_S2,
            is_radian=USE_RADIANS,
            wait=True
        )
        if code != 0:
            raise RuntimeError(f"Failed to move using tcp_pose_aa, code={code}")
        return

    if "tcp_pose_rpy" in pose:
        rpy = pose["tcp_pose_rpy"]
        if len(rpy) != 6:
            raise ValueError(f"Invalid tcp_pose_rpy in pose {pose.get('pose_id', 'unknown')}")
        x, y, z, roll, pitch, yaw = rpy
        code = arm.set_position(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=MOVE_SPEED_MM_S,
            mvacc=MOVE_ACC_MM_S2,
            is_radian=USE_RADIANS,
            wait=True
        )
        if code != 0:
            raise RuntimeError(f"Failed to move using tcp_pose_rpy, code={code}")
        return

    raise ValueError(f"Pose {pose.get('pose_id', 'unknown')} has neither tcp_pose_aa nor tcp_pose_rpy")


def read_current_robot_state(pose_record: dict):
    """
    Convert pose record into R_gripper2base / t_gripper2base for later hand-eye.
    Prefer tcp_pose_aa because it is less ambiguous than Euler order.
    """
    if "tcp_pose_aa" in pose_record and len(pose_record["tcp_pose_aa"]) == 6:
        Rg2b, tg2b = pose_to_rt_from_xarm_tcp_aa(pose_record["tcp_pose_aa"], use_radians=USE_RADIANS)
    elif "tcp_pose_rpy" in pose_record and len(pose_record["tcp_pose_rpy"]) == 6:
        Rg2b, tg2b = pose_to_rt_from_xarm_tcp_rpy(pose_record["tcp_pose_rpy"], use_radians=USE_RADIANS)
    else:
        raise ValueError(f"Pose record {pose_record.get('pose_id', 'unknown')} missing valid tcp pose")

    return Rg2b, tg2b


def make_sample_dict(
    sample_id,
    pose_record,
    image_path,
    R_gripper2base,
    t_gripper2base,
    R_target2cam,
    t_target2cam,
    reproj_error
):
    return {
        "sample_id": sample_id,
        "pose_id": pose_record.get("pose_id", sample_id),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

        "pattern_size": list(PATTERN_SIZE),
        "square_size_m": SQUARE_SIZE_M,

        "image_path": str(image_path),

        "tcp_pose_rpy": pose_record.get("tcp_pose_rpy"),
        "tcp_pose_aa": pose_record.get("tcp_pose_aa"),
        "joint_angles": pose_record.get("joint_angles"),

        "R_gripper2base": np.asarray(R_gripper2base, dtype=float).tolist(),
        "t_gripper2base": np.asarray(t_gripper2base, dtype=float).reshape(3).tolist(),

        "R_target2cam": np.asarray(R_target2cam, dtype=float).tolist(),
        "t_target2cam": np.asarray(t_target2cam, dtype=float).reshape(3).tolist(),

        "reproj_error_px": None if reproj_error is None else float(reproj_error),
    }


# =========================================================
# Main
# =========================================================
def main():
    ensure_dirs()

    print("Loading robot poses...")
    robot_poses = load_robot_poses(POSE_FILE)
    samples_data = load_existing_samples(SAMPLES_FILE)
    existing_count = len(samples_data["samples"])

    print(f"Loaded {len(robot_poses)} robot poses from: {POSE_FILE}")
    print(f"Existing saved samples: {existing_count}")

    print("Connecting xArm...")
    arm = connect_xarm(ROBOT_IP)

    print("Opening ZED2...")
    zed, runtime = create_zed()
    K, dist = get_zed_intrinsics(zed)

    print("Camera intrinsics:")
    print(K)
    print("Distortion:", dist)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for idx, pose in enumerate(robot_poses):
            pose_id = pose.get("pose_id", idx)
            print("\n" + "=" * 70)
            print(f"[Pose {idx+1}/{len(robot_poses)}] pose_id = {pose_id}")
            print("=" * 70)

            # Move robot
            print("Moving robot...")
            move_robot_to_pose(arm, pose)
            print(f"Waiting {WAIT_AFTER_MOVE:.1f}s for stabilization...")
            time.sleep(WAIT_AFTER_MOVE)

            while True:
                img_bgr = grab_bgr_image(zed, runtime)
                if img_bgr is None:
                    print("Failed to grab image from ZED2, retrying...")
                    continue

                found, corners, rvec, tvec, reproj_error, vis = detect_checkerboard_pose(
                    img_bgr, PATTERN_SIZE, SQUARE_SIZE_M, K, dist
                )

                # Overlay status
                lines = [
                    f"Pose {idx+1}/{len(robot_poses)}  |  pose_id={pose_id}",
                    f"Board detected: {'YES' if found else 'NO'}",
                    f"Reproj error: {reproj_error:.3f}px" if reproj_error is not None else "Reproj error: N/A",
                    "ENTER=save   S=skip   R=retry   Q=quit"
                ]

                if found and tvec is not None:
                    tx, ty, tz = tvec.reshape(3)
                    lines.append(f"t_target2cam [m]: {tx:.3f}, {ty:.3f}, {tz:.3f}")

                draw_text_block(vis, lines)

                # Add red warning if not detected
                if not found:
                    cv2.putText(
                        vis,
                        "Checkerboard NOT detected",
                        (20, vis.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_NAME, vis)
                key = cv2.waitKey(10) & 0xFF

                # ENTER key (10 on Linux, sometimes 13)
                if key in (10, 13):
                    if REQUIRE_VALID_DETECTION_TO_SAVE and not found:
                        print("Cannot save: checkerboard not detected.")
                        continue

                    # Compute robot transform
                    R_gripper2base, t_gripper2base = read_current_robot_state(pose)

                    # Compute board pose
                    R_target2cam, t_target2cam = rvec_tvec_to_rt(rvec, tvec)

                    # Save image
                    sample_id = len(samples_data["samples"])
                    image_path = IMAGE_DIR / f"sample_{sample_id:03d}.png"
                    cv2.imwrite(str(image_path), img_bgr)

                    # Save sample record
                    sample = make_sample_dict(
                        sample_id=sample_id,
                        pose_record=pose,
                        image_path=image_path,
                        R_gripper2base=R_gripper2base,
                        t_gripper2base=t_gripper2base,
                        R_target2cam=R_target2cam,
                        t_target2cam=t_target2cam,
                        reproj_error=reproj_error,
                    )

                    samples_data["samples"].append(sample)
                    save_samples(SAMPLES_FILE, samples_data)

                    print(f"Saved sample #{sample_id} from pose_id={pose_id}")
                    print(f"Image: {image_path}")
                    print(f"Samples JSON updated: {SAMPLES_FILE}")
                    break

                elif key in (ord("s"), ord("S")):
                    print(f"Skipped pose_id={pose_id}")
                    break

                elif key in (ord("r"), ord("R")):
                    print(f"Retry current pose_id={pose_id}")
                    continue

                elif key in (ord("q"), ord("Q")):
                    print("Quit requested by user.")
                    return

        print("\nAll poses processed.")
        print(f"Final samples saved in: {SAMPLES_FILE}")

    finally:
        print("Closing camera and robot connection...")
        try:
            zed.close()
        except Exception:
            pass

        try:
            arm.disconnect()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
