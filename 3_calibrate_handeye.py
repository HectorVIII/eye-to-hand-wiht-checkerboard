#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate_handeye.py

Compute eye-to-hand calibration from saved samples.json.

Input:
    calib_dataset/samples.json

Output:
    calib_dataset/handeye_result.json

This script assumes:
    - camera is fixed in the environment
    - checkerboard is mounted on robot flange / gripper
    - each sample contains:
        R_gripper2base, t_gripper2base
        R_target2cam,  t_target2cam

For eye-to-hand with OpenCV calibrateHandEye():
    we convert gripper->base into base->gripper first,
    then call cv2.calibrateHandEye(...)

Result:
    R_cam2base, t_cam2base
"""

import json
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# User settings
# =========================================================
SAMPLES_FILE = Path("calib_dataset/manual_samples_01.json")
OUTPUT_FILE = Path("calib_dataset/handeye_result_0319.json")

METHOD_MAP = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# =========================================================
# Helper functions
# =========================================================
def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "samples" not in data:
        raise ValueError("samples.json does not contain key 'samples'")

    samples = data["samples"]
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("Need at least 3 samples for hand-eye calibration")

    return data, samples


def to_numpy_rotation(R):
    R = np.array(R, dtype=np.float64).reshape(3, 3)
    return R


def to_numpy_translation(t):
    t = np.array(t, dtype=np.float64).reshape(3, 1)
    return t


def invert_transform(R_ab, t_ab):
    """
    Invert transform:
        T_ab = [R_ab, t_ab]
    return:
        T_ba = [R_ba, t_ba]
    """
    R_ba = R_ab.T
    t_ba = -R_ba @ t_ab
    return R_ba, t_ba


def make_homogeneous(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def rotation_matrix_to_rvec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def rotation_error_deg(R1, R2):
    """
    Geodesic rotation error in degrees.
    """
    R = R1.T @ R2
    trace_val = np.trace(R)
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def evaluate_cam2base_consistency(R_cam2base, t_cam2base, samples):
    """
    Evaluate consistency by estimating target pose in base frame for each sample:

        T_target2base = T_cam2base @ T_target2cam

    If calibration is good, all estimated target poses in base should be
    mutually more consistent.

    This is not a perfect physical metric, but it is a useful sanity check.
    """
    t_list = []
    R_list = []

    T_cam2base = make_homogeneous(R_cam2base, t_cam2base)

    for s in samples:
        R_target2cam = to_numpy_rotation(s["R_target2cam"])
        t_target2cam = to_numpy_translation(s["t_target2cam"])
        T_target2cam = make_homogeneous(R_target2cam, t_target2cam)

        T_target2base = T_cam2base @ T_target2cam
        R_target2base = T_target2base[:3, :3]
        t_target2base = T_target2base[:3, 3]

        R_list.append(R_target2base)
        t_list.append(t_target2base)

    t_stack = np.stack(t_list, axis=0)
    t_mean = np.mean(t_stack, axis=0)

    trans_errors = np.linalg.norm(t_stack - t_mean, axis=1)
    trans_mean = float(np.mean(trans_errors))
    trans_max = float(np.max(trans_errors))

    # rotation consistency against mean-like reference (use first as anchor after averaging is nontrivial)
    R_ref = R_list[0]
    rot_errors = [rotation_error_deg(R_ref, R_i) for R_i in R_list]
    rot_mean = float(np.mean(rot_errors))
    rot_max = float(np.max(rot_errors))

    # smaller is better
    score = trans_mean + 0.01 * rot_mean

    return {
        "trans_mean_m": trans_mean,
        "trans_max_m": trans_max,
        "rot_mean_deg": rot_mean,
        "rot_max_deg": rot_max,
        "score": float(score),
    }


def collect_eye_to_hand_inputs(samples):
    """
    For eye-to-hand:
        OpenCV calibrateHandEye is commonly used with base->gripper
        plus target->cam, then interpreted as cam->base.

    So here we convert:
        gripper->base  -->  base->gripper
    """
    R_base2gripper = []
    t_base2gripper = []
    R_target2cam = []
    t_target2cam = []

    for s in samples:
        R_g2b = to_numpy_rotation(s["R_gripper2base"])
        t_g2b = to_numpy_translation(s["t_gripper2base"])

        R_b2g, t_b2g = invert_transform(R_g2b, t_g2b)

        R_t2c = to_numpy_rotation(s["R_target2cam"])
        t_t2c = to_numpy_translation(s["t_target2cam"])

        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)
        R_target2cam.append(R_t2c)
        t_target2cam.append(t_t2c)

    return R_base2gripper, t_base2gripper, R_target2cam, t_target2cam


# =========================================================
# Main
# =========================================================
def main():
    data, samples = load_samples(SAMPLES_FILE)

    print("=" * 72)
    print("Eye-to-Hand Calibration")
    print("=" * 72)
    print(f"Input file : {SAMPLES_FILE}")
    print(f"Samples    : {len(samples)}")
    print()

    R_base2gripper, t_base2gripper, R_target2cam, t_target2cam = collect_eye_to_hand_inputs(samples)

    results = []

    for method_name, method_flag in METHOD_MAP.items():
        print(f"Running method: {method_name}")

        try:
            # For eye-to-hand:
            # pass base->gripper in the gripper2base slot after inversion
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base=R_base2gripper,
                t_gripper2base=t_base2gripper,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=method_flag,
            )

            R_cam2base = np.array(R_cam2base, dtype=np.float64).reshape(3, 3)
            t_cam2base = np.array(t_cam2base, dtype=np.float64).reshape(3, 1)

            eval_info = evaluate_cam2base_consistency(R_cam2base, t_cam2base, samples)

            result = {
                "method": method_name,
                "R_cam2base": R_cam2base.round(9).tolist(),
                "t_cam2base_m": t_cam2base.reshape(3).round(9).tolist(),
                "rvec_cam2base": rotation_matrix_to_rvec(R_cam2base).round(9).tolist(),
                "camera_in_base_m": t_cam2base.reshape(3).round(9).tolist(),
                "evaluation": eval_info,
            }
            results.append(result)

            print(f"  camera_in_base = {result['camera_in_base_m']}")
            print(f"  trans_mean = {eval_info['trans_mean_m']:.6f} m")
            print(f"  trans_max  = {eval_info['trans_max_m']:.6f} m")
            print(f"  rot_mean   = {eval_info['rot_mean_deg']:.4f} deg")
            print(f"  rot_max    = {eval_info['rot_max_deg']:.4f} deg")
            print(f"  score      = {eval_info['score']:.6f}")
            print()

        except Exception as e:
            print(f"  [Failed] {method_name}: {e}")
            print()

    if len(results) == 0:
        raise RuntimeError("All hand-eye methods failed")

    results.sort(key=lambda x: x["evaluation"]["score"])
    best = results[0]

    output = {
        "meta": {
            "input_file": str(SAMPLES_FILE),
            "num_samples": len(samples),
            "configuration": "eye-to-hand",
            "notes": [
                "Input samples store gripper2base and target2cam.",
                "For eye-to-hand, script inverts gripper2base to base2gripper before calibrateHandEye.",
                "Returned result is interpreted as cam2base.",
            ],
        },
        "best_result": best,
        "all_results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Finished.")
    print(f"Best method: {best['method']}")
    print(f"Best camera_in_base_m: {best['camera_in_base_m']}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
