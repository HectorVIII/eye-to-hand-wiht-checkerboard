#!/usr/bin/env python3

"""
manual_capture_calib_dataset.py

Manual hand-eye calibration dataset capture for xArm7 + ZED2.

Workflow:
    1. Put xArm into manual / drag mode.
    2. Move robot by hand until checkerboard is fully visible and stably detected.
    3. Press 's' to save one calibration sample.
    4. Repeat until enough samples are collected.
    5. Press 'q' to quit.

Each saved sample contains:
    - current robot TCP pose (RPY, axis-angle, joints)
    - R_gripper2base, t_gripper2base
    - R_target2cam, t_target2cam
    - image path
    - reprojection error

Output:
    calib_dataset/
        images/
            sample_000.png
            sample_001.png
            ...
        samples.json
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
# Parameter settings
# =========================================================
ROBOT_IP = "192.168.1.225"

SAVE_DIR = Path("calib_dataset")
RUN_TIMESTAMP = datetime.now().strftime("%m%d_%H%M%S")
IMAGE_DIR = SAVE_DIR / f"images_{RUN_TIMESTAMP}"
SAMPLES_FILE = SAVE_DIR / f"manual_samples_{RUN_TIMESTAMP}.json"

USE_RADIANS = True  # Set to False if you prefer degrees in saved samples.json
TARGET_NUM_SAMPLES = 40

# Zivid 7x8 squares -> 6x7 inner corners
PATTERN_SIZE = (6, 7)
SQUARE_SIZE_M = 0.03    # 3cm squares, adjust if using different checkerboard

WINDOW_NAME = "Manual Capture Calibration Dataset"

# Detection settings
STABLE_DETECTION_FRAMES = 5 # Number of consecutive frames with successful checkerboard detection required to consider it "stable"
DISPLAY_SCALE = 1.3
READ_STABLE_DELAY_S = 0.15  # Delay after stable detection before reading robot state and saving sample, to reduce motion blur and ensure stable pose

# Optional image quality settings for ZED
SET_MANUAL_CAMERA_PARAMS = False
EXPOSURE_VALUE = 50
GAIN_VALUE = 50


# =========================================================
# General helpers
# =========================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def round_list(values, digits=6):
    """Round a list of values to specified number of decimal places for cleaner JSON output."""
    return [round(float(v), digits) for v in values]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {path.resolve()}")


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
    arm.set_mode(0)     # 0=position control
    arm.set_state(0)    # 0=motion state, 3=pause state, 4=stop state， 6=deceleration stop state

    time.sleep(0.5)

    code_state, state = arm.get_state() # state: 1=in motion, 2=sleeping, 3=suspended, 4=stopping
    code_err, err_warn = arm.get_err_warn_code()

    print("Connected.")
    if code_state == 0:
        print(f"Robot state: {state}")
    if code_err == 0:
        print(f"Error/Warning: {err_warn}")

    return arm


def get_robot_state(arm, use_radians=True):
    code_rpy, pose_rpy = arm.get_position(is_radian=use_radians)    #let the value(roll/pitch/yaw) to return is an radian unit
    code_aa, pose_aa = arm.get_position_aa(is_radian=use_radians)
    code_j, joints = arm.get_servo_angle(is_radian=use_radians)

    if code_rpy != 0:
        raise RuntimeError(f"Failed to read TCP RPY pose, code={code_rpy}")
    if code_aa != 0:
        raise RuntimeError(f"Failed to read TCP axis-angle pose, code={code_aa}")
    if code_j != 0:
        raise RuntimeError(f"Failed to read joint angles, code={code_j}")

    return pose_rpy, pose_aa, joints


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


# =========================================================
# ZED helpers
# =========================================================
def create_zed():
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED2: {status}")

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    return zed, runtime, image_mat


def set_zed_camera_params(zed):
    if not SET_MANUAL_CAMERA_PARAMS:
        return

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, EXPOSURE_VALUE)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, GAIN_VALUE)
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
    """
    Convert an input image(BGRA) to 3-channel BGR format for OpenCV processing.
    """
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


# =========================================================
# Checkerboard detection and pose estimation
# =========================================================
def build_object_points(pattern_size, square_size_m):
    """
    Build checkerboard corner coordinates in the target frame.
    All points lie on the z=0 plane, with adjacent corners spaced
    by square_size_m in meters.
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def preprocess_variants(bgr):
    """
    Generate several grayscale preprocessing variants of the same image
    to improve checkerboard detection under different lighting and scale conditions.
    """
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
    """
    Detect checkerboard corners with the sector-based OpenCV detector
    using robust flags for normalization, exhaustive search, and accuracy.
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
    Map checkerboard corner coordinates from a resized image
    back to the original image scale.
    """
    if scale == 1.0:
        return corners
    out = corners.copy()
    out[:, 0, 0] /= scale
    out[:, 0, 1] /= scale
    return out


def detect_checkerboard_robust(bgr, pattern_size):
    """
    Try checkerboard detection on serveral preprocessed image variants
    and return the first successful result together with a visualiyation image.
    """
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
    Estimate the checkerboard  pose in the camera frame from 3D to 2D
    corner correspondences using solvePnP. and compute the mean reprojection error to test the precision.
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

    return R_target2cam, t_target2cam, reproj_error, rvec, tvec


def draw_axes_on_board(vis, camera_matrix, dist_coeffs, rvec, tvec, axis_len=0.10):
    try:
        cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, axis_len, 3)
    except Exception:
        pass
    return vis


def draw_overlay(vis, found, stable_count, method_name, num_saved, target_num, reproj_error=None, board_pos=None):
    """Draw text overlay on the image to show detection status, method, reprojection error, and saved samples info."""
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
    cv2.putText(vis, f"Stable frames: {stable_count} / {STABLE_DETECTION_FRAMES}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Method: {method_name}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Saved samples: {num_saved} / {target_num}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    if reproj_error is not None:
        cv2.putText(vis, f"Reproj error: {reproj_error:.3f}px", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    if board_pos is not None:
        bx, by, bz = board_pos
        cv2.putText(vis, f"Board in cam (m): x={bx:.3f}, y={by:.3f}, z={bz:.3f}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(vis, "Move robot by hand. Press 's' to save current sample.", (20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Press 'q' to quit. Save only when board is fully visible and stable.", (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


# =========================================================
# Sample handling
# =========================================================
def make_sample_record(sample_id, pose_rpy, pose_aa, joints,
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


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 72)
    print("Manual Capture Calibration Dataset")
    print("=" * 72)
    print(f"Save dir          : {SAVE_DIR}")
    print(f"Target samples    : {TARGET_NUM_SAMPLES}")
    print()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    arm = None
    zed = None
    samples = []
    stable_count = 0

    try:
        arm = connect_arm(ROBOT_IP)

        zed, runtime, image_mat = create_zed()
        set_zed_camera_params(zed)
        camera_matrix, dist_coeffs = get_left_camera_intrinsics(zed)

        object_points = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)

        latest_corners = None
        latest_method = "none"
        latest_reproj_error = None
        latest_R_target2cam = None
        latest_t_target2cam = None
        latest_rvec = None
        latest_tvec = None

        while True:
            grab_status = zed.grab(runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            frame = image_mat.get_data()
            frame_bgr = to_bgr(frame)
            if frame_bgr is None:
                continue

            found, corners, vis, method_name = detect_checkerboard_robust(frame_bgr, PATTERN_SIZE)

            latest_corners = None
            latest_method = method_name
            latest_reproj_error = None
            latest_R_target2cam = None
            latest_t_target2cam = None
            latest_rvec = None
            latest_tvec = None

            if found:
                try:
                    img_pts = corners.reshape(-1, 2)
                    R_target2cam, t_target2cam, reproj_error, rvec, tvec = estimate_target_pose(
                        object_points=object_points,
                        image_points=img_pts,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                    )
                    stable_count += 1
                    latest_corners = corners
                    latest_method = method_name
                    latest_reproj_error = reproj_error
                    latest_R_target2cam = R_target2cam
                    latest_t_target2cam = t_target2cam
                    latest_rvec = rvec
                    latest_tvec = tvec
                    vis = draw_axes_on_board(vis, camera_matrix, dist_coeffs, rvec, tvec)
                except Exception:
                    stable_count = 0
            else:
                stable_count = 0

            board_pos = latest_t_target2cam.tolist() if latest_t_target2cam is not None else None

            vis = draw_overlay(
                vis=vis,
                found=(latest_corners is not None),
                stable_count=stable_count,
                method_name=latest_method,
                num_saved=len(samples),
                target_num=TARGET_NUM_SAMPLES,
                reproj_error=latest_reproj_error,
                board_pos=board_pos,
            )

            if DISPLAY_SCALE != 1.0:
                vis_show = cv2.resize(
                    vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                vis_show = vis

            cv2.imshow(WINDOW_NAME, vis_show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                if latest_corners is None:
                    print("[Skip] Checkerboard not detected.")
                    continue

                if stable_count < STABLE_DETECTION_FRAMES:
                    print(f"[Skip] Detection not stable yet: {stable_count}/{STABLE_DETECTION_FRAMES}")
                    continue

                time.sleep(READ_STABLE_DELAY_S)

                try:
                    pose_rpy, pose_aa, joints = get_robot_state(arm, use_radians=USE_RADIANS)
                except Exception as e:
                    print(f"[Error] Failed to read robot state: {e}")
                    continue

                try:
                    R_gripper2base, t_gripper2base = pose_aa_to_transform(pose_aa)

                    img_pts = latest_corners.reshape(-1, 2)
                    R_target2cam, t_target2cam, reproj_error, _, _ = estimate_target_pose(
                        object_points=object_points,
                        image_points=img_pts,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                    )
                except Exception as e:
                    print(f"[Error] Failed to build sample: {e}")
                    continue

                sample_id = len(samples)
                image_path = IMAGE_DIR / f"sample_{sample_id:03d}.png"

                ok_img = cv2.imwrite(str(image_path), frame_bgr)
                if not ok_img:
                    print(f"[Warning] Failed to save image: {image_path}")
                    continue

                sample = make_sample_record(
                    sample_id=sample_id,
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

                output = {
                    "meta": {
                        "created_at": now_str(),
                        "robot_ip": ROBOT_IP,
                        "robot_model": "xArm7",
                        "pattern_size": list(PATTERN_SIZE),
                        "square_size_m": SQUARE_SIZE_M,
                        "use_radians": USE_RADIANS,
                        "angle_unit": "radian" if USE_RADIANS else "degree",
                        "position_unit": "mm",
                        "description": "Manually captured hand-eye calibration dataset with live ZED2 checkerboard preview",
                        "num_samples": len(samples),
                    },
                    "samples": samples,
                }
                save_json(SAMPLES_FILE, output)

                x, y, z, roll, pitch, yaw = pose_rpy
                bx, by, bz = t_target2cam.tolist()

                print("-" * 60)
                print(f"[Saved sample #{sample_id}]")
                print(f"image_path      : {image_path}")
                print(f"tcp_pose_rpy    : {[round(v, 6) for v in pose_rpy]}")
                print(f"tcp_pose_aa     : {[round(v, 6) for v in pose_aa]}")
                print(f"joint_angles    : {[round(v, 6) for v in joints]}")
                print(f"board_in_cam(m) : x={bx:.6f}, y={by:.6f}, z={bz:.6f}")
                print(f"reproj_error_px : {reproj_error:.6f}")
                print(f"method          : {latest_method}")

                print("\n[Robot TCP xyzrpy]")
                print(f"x={x:.6f}, y={y:.6f}, z={z:.6f}")
                print(f"roll={roll:.6f}, pitch={pitch:.6f}, yaw={yaw:.6f}")

                if len(samples) >= TARGET_NUM_SAMPLES:
                    print("\nTarget number of samples reached.")
                    break

        if len(samples) == 0:
            print("\nNo samples saved.")
            return

        output = {
            "meta": {
                "created_at": now_str(),
                "robot_ip": ROBOT_IP,
                "robot_model": "xArm7",
                "pattern_size": list(PATTERN_SIZE),
                "square_size_m": SQUARE_SIZE_M,
                "use_radians": USE_RADIANS,
                "angle_unit": "radian" if USE_RADIANS else "degree",
                "position_unit": "mm",
                "description": "Manually captured hand-eye calibration dataset with live ZED2 checkerboard preview",
                "num_samples": len(samples),
            },
            "samples": samples,
        }
        save_json(SAMPLES_FILE, output)

        print("\nCollection finished successfully.")
        print(f"Total samples saved: {len(samples)}")

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