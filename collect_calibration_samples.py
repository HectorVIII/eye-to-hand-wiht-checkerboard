import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from xarm.wrapper import XArmAPI

# =========================
# User settings
# =========================
ROBOT_IP = "192.168.1.XXX"
POSE_FILE = "robot_poses.json"
SAVE_DIR = "calib_dataset"

USE_RADIANS = True
CONFIRM_EACH_POSE = True

# Motion parameters
SPEED_MM_S = 50
ACC_MM_S2 = 200
WAIT_AFTER_MOVE = 1.5

# Safe motion parameters
SAFE_Z_MM = 450
Z_LIFT_MARGIN_MM = 80
MIN_Z_MM = 120
MAX_Z_MM = 650

# Checkerboard settings
# !!! You must verify this !!!
PATTERN_SIZE = (6, 7)      # try (6,7) first; if fails, test (7,8)
SQUARE_SIZE_M = 0.03       # 30 mm

# ZED settings
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# =========================
# Helpers
# =========================
def clamp(value, low, high):
    return max(low, min(high, value))

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def pose_to_dict(pose):
    return {
        "x": pose[0],
        "y": pose[1],
        "z": pose[2],
        "roll": pose[3],
        "pitch": pose[4],
        "yaw": pose[5]
    }

def move_pose(arm, x, y, z, roll, pitch, yaw, label=""):
    print(f"  -> Moving to {label}: "
          f"x={x:.2f}, y={y:.2f}, z={z:.2f}, "
          f"roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")

    code = arm.set_position(
        x=x, y=y, z=z,
        roll=roll, pitch=pitch, yaw=yaw,
        speed=SPEED_MM_S,
        mvacc=ACC_MM_S2,
        wait=True,
        is_radian=USE_RADIANS
    )
    return code

def move_two_stage(arm, target_tcp):
    code_pose, current_pose = arm.get_position(is_radian=USE_RADIANS)
    if code_pose != 0:
        print(f"Failed to read current pose, code={code_pose}")
        return code_pose

    cx, cy, cz, croll, cpitch, cyaw = current_pose
    tx = target_tcp["x"]
    ty = target_tcp["y"]
    tz = clamp(target_tcp["z"], MIN_Z_MM, MAX_Z_MM)
    troll = target_tcp["roll"]
    tpitch = target_tcp["pitch"]
    tyaw = target_tcp["yaw"]

    safe_z = max(cz, tz, SAFE_Z_MM) + Z_LIFT_MARGIN_MM
    safe_z = clamp(safe_z, MIN_Z_MM, MAX_Z_MM)

    print(f"  Current z: {cz:.2f}, Target z: {tz:.2f}, Safe z: {safe_z:.2f}")

    code = move_pose(arm, cx, cy, safe_z, croll, cpitch, cyaw, label="raise-to-safe-z")
    if code != 0:
        return code
    time.sleep(0.2)

    code = move_pose(arm, tx, ty, safe_z, troll, tpitch, tyaw, label="lateral-at-safe-z")
    if code != 0:
        return code
    time.sleep(0.2)

    code = move_pose(arm, tx, ty, tz, troll, tpitch, tyaw, label="descend-to-target")
    return code

def init_zed():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {status}")

    runtime_params = sl.RuntimeParameters()
    return zed, runtime_params

def get_left_image_bgr(zed, runtime_params):
    image = sl.Mat()
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        return None

    zed.retrieve_image(image, sl.VIEW.LEFT)
    frame = image.get_data()  # RGBA
    if frame is None:
        return None

    if frame.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        frame_bgr = frame.copy()
    return frame_bgr

def get_camera_matrix_and_dist(zed):
    calib = zed.get_camera_information().camera_configuration.calibration_parameters
    left_cam = calib.left_cam

    fx = left_cam.fx
    fy = left_cam.fy
    cx = left_cam.cx
    cy = left_cam.cy

    # distortion order may depend on SDK version
    # Often [k1, k2, p1, p2, k3]
    dist = np.array(left_cam.disto, dtype=np.float64)

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ], dtype=np.float64)

    return K, dist

def build_object_points(pattern_size, square_size_m):
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp

def detect_checkerboard_pose(image_bgr, K, dist, pattern_size, square_size_m):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    vis = image_bgr.copy()

    if not found:
        return False, None, None, None, vis

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )
    corners_refined = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    objp = build_object_points(pattern_size, square_size_m)

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners_refined,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)

    if not ok:
        return False, None, None, corners_refined, vis

    return True, rvec, tvec, corners_refined, vis

# =========================
# Main
# =========================
def main():
    pose_path = Path(POSE_FILE)
    if not pose_path.exists():
        print(f"Pose file not found: {POSE_FILE}")
        return

    poses = json.loads(pose_path.read_text(encoding="utf-8"))
    if not poses:
        print("No poses in pose file.")
        return

    save_dir = Path(SAVE_DIR)
    image_dir = save_dir / "images"
    debug_dir = save_dir / "debug"
    ensure_dir(image_dir)
    ensure_dir(debug_dir)

    # init robot
    arm = XArmAPI(ROBOT_IP)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    # init camera
    zed, runtime_params = init_zed()
    K, dist = get_camera_matrix_and_dist(zed)

    print("Camera matrix:")
    print(K)
    print("Dist coeffs:")
    print(dist)

    samples = []
    failed_ids = []

    start = input("Press ENTER to start sample collection, or type 'q' to quit: ").strip().lower()
    if start == "q":
        zed.close()
        arm.disconnect()
        return

    for i, item in enumerate(poses, start=1):
        tcp = item["tcp_pose_mm_rad"]

        print(f"\n=== Pose {i}/{len(poses)} ===")
        print(f"Target TCP: {tcp}")

        if CONFIRM_EACH_POSE:
            cmd = input("Press ENTER to move and capture, or type 'q' to stop: ").strip().lower()
            if cmd == "q":
                break

        code = move_two_stage(arm, tcp)
        if code != 0:
            print(f"Move failed at pose {i}, code={code}")
            failed_ids.append(i)
            continue

        print(f"Waiting {WAIT_AFTER_MOVE:.1f}s for robot/image to stabilize...")
        time.sleep(WAIT_AFTER_MOVE)

        frame = get_left_image_bgr(zed, runtime_params)
        if frame is None:
            print("Failed to grab image from ZED.")
            failed_ids.append(i)
            continue

        found, rvec, tvec, corners, vis = detect_checkerboard_pose(
            frame, K, dist, PATTERN_SIZE, SQUARE_SIZE_M
        )

        raw_img_name = f"pose_{i:03d}.png"
        dbg_img_name = f"pose_{i:03d}_debug.png"

        raw_img_path = image_dir / raw_img_name
        dbg_img_path = debug_dir / dbg_img_name

        cv2.imwrite(str(raw_img_path), frame)
        cv2.imwrite(str(dbg_img_path), vis)

        if not found:
            print(f"Checkerboard NOT found at pose {i}.")
            failed_ids.append(i)
            continue

        code_pose, actual_pose = arm.get_position(is_radian=USE_RADIANS)
        if code_pose != 0:
            print(f"Warning: failed to read actual TCP pose, code={code_pose}")
            actual_pose = [None] * 6

        sample = {
            "id": i,
            "image_path": str(raw_img_path),
            "debug_image_path": str(dbg_img_path),
            "tcp_pose_mm_rad": pose_to_dict(actual_pose),
            "camera_to_board": {
                "rvec": rvec.reshape(-1).tolist(),
                "tvec_m": tvec.reshape(-1).tolist()
            },
            "meta": {
                "pattern_size": list(PATTERN_SIZE),
                "square_size_m": SQUARE_SIZE_M,
                "num_corners": int(len(corners))
            }
        }

        samples.append(sample)
        print(f"Saved valid sample {i}. tvec = {tvec.reshape(-1)}")

        samples_path = save_dir / "samples.json"
        samples_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    summary = {
        "num_total_poses": len(poses),
        "num_valid_samples": len(samples),
        "failed_ids": failed_ids
    }

    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    zed.close()
    arm.disconnect()

    print("\n=== Done ===")
    print(f"Valid samples: {len(samples)} / {len(poses)}")
    print(f"Failed poses : {failed_ids}")
    print(f"Saved to: {save_dir}")

if __name__ == "__main__":
    main()