import json
import time
from pathlib import Path
from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.225"
SAVE_PATH = "robot_poses.json"
NUM_POSES = 20

def main():
    arm = XArmAPI(ROBOT_IP)
    arm.connect()

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    poses = []

    print("\n=== xArm pose recorder ===")
    print("Move the robot manually to a pose, then press ENTER to save.")
    print("Type 'q' then ENTER to quit early.\n")

    for i in range(NUM_POSES):
        cmd = input(f"[{i+1}/{NUM_POSES}] Move robot to desired pose, then press ENTER to save: ")
        if cmd.strip().lower() == "q":
            break

        code_pose, pose = arm.get_position(is_radian=True)
        
        code_joint, joints = arm.get_servo_angle(is_radian=True)

        if code_pose != 0:
            print(f"Failed to read TCP pose, code={code_pose}")
            continue

        if code_joint != 0:
            print(f"Warning: failed to read joint angles, code={code_joint}")
            joints = None

        record = {
            "id": len(poses) + 1,
            "timestamp": time.time(),
            "tcp_pose_mm_rad": {
                "x": pose[0],
                "y": pose[1],
                "z": pose[2],
                "roll": pose[3],
                "pitch": pose[4],
                "yaw": pose[5]
            },
            "joints_rad": joints
        }

        poses.append(record)

        print(f"Saved pose {record['id']}:")
        print(record["tcp_pose_mm_rad"])
        print()

    Path(SAVE_PATH).write_text(json.dumps(poses, indent=2), encoding="utf-8")
    print(f"Saved {len(poses)} poses to {SAVE_PATH}")

    arm.disconnect()

if __name__ == "__main__":
    main()