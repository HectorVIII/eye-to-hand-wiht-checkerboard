import json
import cv2
import numpy as np
from itertools import product


# ======================================================
# Basic transform helpers
# ======================================================
def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_RT(R, t):
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    R_inv, t_inv = invert_RT(R, t)
    return make_T(R_inv, t_inv)


def rot_err_deg(R):
    v = (np.trace(R) - 1.0) / 2.0
    v = np.clip(v, -1.0, 1.0)
    return np.degrees(np.arccos(v))


# ======================================================
# Input conversions
# ======================================================
def tcp_pose_aa_to_T_base_gripper(pose_aa):
    x, y, z, ax, ay, az = pose_aa

    rvec = np.array([ax, ay, az], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)

    t = np.array([x, y, z], dtype=np.float64).reshape(3, 1) / 1000.0
    return make_T(R, t)


def pnp_to_T_target_cam(rvec, tvec):
    rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


# ======================================================
# Consistency score:
# board rigidly mounted on gripper => T_gripper_target constant
# ======================================================
def evaluate_solution(T_cam2base, T_base_gripper_all, T_target_cam_all):
    T_base_cam = invert_T(T_cam2base)

    gt_list = []
    for T_bg, T_tc in zip(T_base_gripper_all, T_target_cam_all):
        T_cam_target = invert_T(T_tc)
        T_gt = invert_T(T_bg) @ T_base_cam @ T_cam_target
        gt_list.append(T_gt)

    translations = np.array([T[:3, 3] for T in gt_list])
    mean_t = translations.mean(axis=0)
    trans_scatter = np.linalg.norm(translations - mean_t, axis=1)

    R_ref = gt_list[0][:3, :3]
    rot_scatter = []
    for T in gt_list:
        R_rel = R_ref.T @ T[:3, :3]
        rot_scatter.append(rot_err_deg(R_rel))
    rot_scatter = np.array(rot_scatter)

    return {
        "trans_mean": float(trans_scatter.mean()),
        "trans_max": float(trans_scatter.max()),
        "rot_mean_deg": float(rot_scatter.mean()),
        "rot_max_deg": float(rot_scatter.max()),
        "mean_gripper_target_t": mean_t,
    }


# ======================================================
# Load data
# ======================================================
with open("samples.json", "r") as f:
    data = json.load(f)

print("samples:", len(data))

for i, s in enumerate(data):
    for k in ["tcp_pose_aa", "rvec", "tvec_m"]:
        if k not in s:
            raise KeyError(
                f"Sample {i} missing '{k}'. "
                "This script requires new samples with tcp_pose_aa."
            )

T_base_gripper_all = [tcp_pose_aa_to_T_base_gripper(s["tcp_pose_aa"]) for s in data]
T_target_cam_all = [pnp_to_T_target_cam(s["rvec"], s["tvec_m"]) for s in data]

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

INPUT_MODES = ["base2gripper", "gripper2base"]

results = []

for input_mode, (method_name, method_id) in product(INPUT_MODES, METHODS.items()):
    try:
        R_list = []
        t_list = []

        for T_bg in T_base_gripper_all:
            if input_mode == "base2gripper":
                T_in = T_bg
            else:
                T_in = invert_T(T_bg)

            R_list.append(T_in[:3, :3])
            t_list.append(T_in[:3, 3].reshape(3, 1))

        R_target2cam = [T[:3, :3] for T in T_target_cam_all]
        t_target2cam = [T[:3, 3].reshape(3, 1) for T in T_target_cam_all]

        R_out, t_out = cv2.calibrateHandEye(
            R_list,
            t_list,
            R_target2cam,
            t_target2cam,
            method=method_id
        )

        T_out = make_T(R_out, t_out)

        # Interpretation A: output is cam->base
        eval_A = evaluate_solution(T_out, T_base_gripper_all, T_target_cam_all)
        score_A = eval_A["trans_mean"] + 0.01 * eval_A["rot_mean_deg"]

        # Interpretation B: output is base->cam, invert it
        T_inv = invert_T(T_out)
        eval_B = evaluate_solution(T_inv, T_base_gripper_all, T_target_cam_all)
        score_B = eval_B["trans_mean"] + 0.01 * eval_B["rot_mean_deg"]

        if score_A <= score_B:
            final_T = T_out
            final_eval = eval_A
            output_mode = "cam2base"
            score = score_A
        else:
            final_T = T_inv
            final_eval = eval_B
            output_mode = "base2cam_inverted_to_cam2base"
            score = score_B

        results.append({
            "input_mode": input_mode,
            "method": method_name,
            "output_mode": output_mode,
            "score": score,
            "eval": final_eval,
            "T_cam2base": final_T,
        })

    except Exception as e:
        print(f"FAILED: input={input_mode}, method={method_name}, err={e}")

results.sort(key=lambda x: x["score"])

print("\n================ TOP RESULTS ================\n")
for i, r in enumerate(results[:10], start=1):
    tx, ty, tz = r["T_cam2base"][:3, 3]
    ev = r["eval"]
    print(f"[{i}] input={r['input_mode']}, method={r['method']}, out={r['output_mode']}")
    print(f"    camera_in_base = [{tx:.4f}, {ty:.4f}, {tz:.4f}] m")
    print(f"    trans_mean={ev['trans_mean']:.6f} m, trans_max={ev['trans_max']:.6f} m")
    print(f"    rot_mean={ev['rot_mean_deg']:.4f} deg, rot_max={ev['rot_max_deg']:.4f} deg")
    print(f"    score={r['score']:.6f}\n")

best = results[0]
T_best = best["T_cam2base"]

print("\n================ BEST RESULT ================\n")
print(f"input_mode  : {best['input_mode']}")
print(f"method      : {best['method']}")
print(f"output_mode : {best['output_mode']}")
print(f"camera pos  : {T_best[:3, 3]}")

print("\nT_cam2base = np.array([")
for row in T_best:
    print(f"    [{row[0]: .8f}, {row[1]: .8f}, {row[2]: .8f}, {row[3]: .8f}],")
print("], dtype=np.float64)")