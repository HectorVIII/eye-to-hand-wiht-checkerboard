import json
from pathlib import Path

import cv2
import numpy as np

SAMPLES_FILE = "calib_dataset/samples.json"
OUTPUT_FILE = "calib_dataset/handeye_result.json"

# 选择方法
METHOD_NAME = "TSAI"
METHOD_MAP = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def euler_xyz_to_rotmat(roll, pitch, yaw):
    """
    Build rotation matrix from roll, pitch, yaw.
    Assumes the robot pose angles are given in radians.

    IMPORTANT:
    xArm's exact convention must match this assumption.
    If final calibration looks wrong, this convention is one of the first things to verify.
    """
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=np.float64)

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=np.float64)

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,   0,  1]
    ], dtype=np.float64)

    # Common convention guess: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def make_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def pose_dict_to_transform_mm_rad(pose_dict):
    x = pose_dict["x"]
    y = pose_dict["y"]
    z = pose_dict["z"]
    roll = pose_dict["roll"]
    pitch = pose_dict["pitch"]
    yaw = pose_dict["yaw"]

    R = euler_xyz_to_rotmat(roll, pitch, yaw)
    t_m = np.array([x, y, z], dtype=np.float64) / 1000.0  # mm -> m
    return make_transform(R, t_m)


def rvec_tvec_to_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return make_transform(R, t)


def pretty_matrix(T, name):
    print(f"\n{name} =")
    with np.printoptions(precision=6, suppress=True):
        print(T)


def transform_to_json(T):
    return {
        "matrix_4x4": T.tolist(),
        "R": T[:3, :3].tolist(),
        "t_m": T[:3, 3].tolist()
    }


def main():
    samples_path = Path(SAMPLES_FILE)
    if not samples_path.exists():
        print(f"Samples file not found: {SAMPLES_FILE}")
        return

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if len(samples) < 5:
        print("Too few samples. Try at least 10, preferably 15-20 valid samples.")
        return

    method = METHOD_MAP[METHOD_NAME]

    # OpenCV inputs
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    valid_count = 0

    for s in samples:
        tcp = s["tcp_pose_mm_rad"]
        cam_board = s["camera_to_board"]

        # T_base_flange from robot pose
        T_base_flange = pose_dict_to_transform_mm_rad(tcp)

        # OpenCV calibrateHandEye expects gripper->base, so invert base->flange
        T_flange_base = invert_transform(T_base_flange)

        # T_cam_board from solvePnP
        T_cam_board = rvec_tvec_to_transform(
            cam_board["rvec"],
            cam_board["tvec_m"]
        )

        R_gripper2base.append(T_flange_base[:3, :3])
        t_gripper2base.append(T_flange_base[:3, 3].reshape(3, 1))

        # OpenCV expects target->cam
        # We currently have cam->board, so invert to board->cam? Wait:
        # solvePnP returns object(board) -> camera
        # which is exactly target->cam.
        # So T_cam_board name in previous script is actually misleading.
        # It is mathematically T_target_to_cam (board->camera).
        # We'll use it directly.
        T_board_cam = T_cam_board

        R_target2cam.append(T_board_cam[:3, :3])
        t_target2cam.append(T_board_cam[:3, 3].reshape(3, 1))

        valid_count += 1

    print(f"Using {valid_count} samples")
    print(f"Method: {METHOD_NAME}")

    # Result is camera->gripper for the eye-in-hand convention in docs,
    # but for eye-to-hand with the chosen inputs, many people use the same API
    # with gripper2base and target2cam.
    # We will compute it, then explicitly derive and print both possibilities.
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method
    )

    T_cam_gripper = make_transform(R_cam2gripper, t_cam2gripper.reshape(3))
    T_gripper_cam = invert_transform(T_cam_gripper)

    pretty_matrix(T_cam_gripper, "T_cam_gripper")
    pretty_matrix(T_gripper_cam, "T_gripper_cam")

    # For eye-to-hand, what you usually need in practice is T_base_cam.
    # A simple way to get a per-sample estimate is:
    # T_base_cam_i = T_base_flange_i @ T_flange_cam
    # if flange->cam is fixed.
    T_base_cam_list = []
    for s in samples:
        T_base_flange = pose_dict_to_transform_mm_rad(s["tcp_pose_mm_rad"])
        T_base_cam_i = T_base_flange @ T_gripper_cam
        T_base_cam_list.append(T_base_cam_i)

    # Average translation
    t_all = np.array([T[:3, 3] for T in T_base_cam_list])
    t_mean = np.mean(t_all, axis=0)

    # Average rotation using SVD projection
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for T in T_base_cam_list:
        R_sum += T[:3, :3]
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt

    T_base_cam = make_transform(R_mean, t_mean)
    T_cam_base = invert_transform(T_base_cam)

    pretty_matrix(T_base_cam, "T_base_cam")
    pretty_matrix(T_cam_base, "T_cam_base")

    # Consistency check across samples
    trans_errors_mm = []
    rot_errors_deg = []

    for T_i in T_base_cam_list:
        dT = invert_transform(T_base_cam) @ T_i
        dt = np.linalg.norm(dT[:3, 3]) * 1000.0
        dR = dT[:3, :3]
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2.0, -1.0, 1.0))
        rot_deg = np.degrees(angle)

        trans_errors_mm.append(float(dt))
        rot_errors_deg.append(float(rot_deg))

    result = {
        "method": METHOD_NAME,
        "num_samples": valid_count,
        "T_cam_gripper": transform_to_json(T_cam_gripper),
        "T_gripper_cam": transform_to_json(T_gripper_cam),
        "T_base_cam": transform_to_json(T_base_cam),
        "T_cam_base": transform_to_json(T_cam_base),
        "consistency": {
            "translation_error_mm": {
                "mean": float(np.mean(trans_errors_mm)),
                "max": float(np.max(trans_errors_mm)),
                "all": trans_errors_mm
            },
            "rotation_error_deg": {
                "mean": float(np.mean(rot_errors_deg)),
                "max": float(np.max(rot_errors_deg)),
                "all": rot_errors_deg
            }
        },
        "notes": [
            "Robot Euler angle convention must match the conversion in this script.",
            "solvePnP returns board(object) to camera transform.",
            "If results look unreasonable, first verify xArm roll/pitch/yaw convention."
        ]
    }

    output_path = Path(OUTPUT_FILE)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved result to: {output_path}")

    print("\nConsistency summary:")
    print(f"Translation mean error: {np.mean(trans_errors_mm):.3f} mm")
    print(f"Translation max  error: {np.max(trans_errors_mm):.3f} mm")
    print(f"Rotation mean error   : {np.mean(rot_errors_deg):.3f} deg")
    print(f"Rotation max  error   : {np.max(rot_errors_deg):.3f} deg")


if __name__ == "__main__":
    main()