import json
import time
from pathlib import Path
from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.225"   
POSE_FILE = "robot_poses.json"

# Motion parameters
SPEED_MM_S = 50             #linear speed mm/s
ACC_MM_S2 = 200             #accelerate speed
WAIT_AFTER_MOVE = 1.5       #After reaching each point, wait 1.5 seconds
USE_RADIANS = True          #Angle units are in radians
CONFIRM_EACH_POSE = True    #manually press Enter to confirm before each pose.

# Safe motion parameters
SAFE_Z_MM = 450           # safety height, Before moving sideways, the robot should first raise itself as close as possible to this height
Z_LIFT_MARGIN_MM = 80     # Raise an additional 80 mm
MIN_Z_MM = 120           
MAX_Z_MM = 650           

# Limit the z-height to prevent it from being too low or too high
def clamp(value, low, high):
    return max(low, min(high, value))

def move_pose(arm, x, y, z, roll, pitch, yaw, label=""):
    print(f"  -> Moving to {label}: "
          f"x={x:.2f}, y={y:.2f}, z={z:.2f}, "
          f"roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")

    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=SPEED_MM_S,
        mvacc=ACC_MM_S2,
        wait=True,
        is_radian=USE_RADIANS
    )
    return code

def move_two_stage(arm, target_tcp):
    """
    Safer move:
    1) raise current pose to safe Z
    2) move laterally at safe Z to target x/y with target orientation
    3) descend vertically to target z
    """
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

    print(f"  Current pose: x={cx:.2f}, y={cy:.2f}, z={cz:.2f}")
    print(f"  Target pose : x={tx:.2f}, y={ty:.2f}, z={tz:.2f}")
    print(f"  Computed safe_z = {safe_z:.2f}")

    # Step 1: raise vertically from current position
    code = move_pose(
        arm,
        cx, cy, safe_z,
        croll, cpitch, cyaw,
        label="raise-to-safe-z"
    )
    if code != 0:
        print(f"Failed at step 1 (raise), code={code}")
        return code

    time.sleep(0.2)

    # Step 2: move laterally at safe Z with target orientation
    code = move_pose(
        arm,
        tx, ty, safe_z,
        troll, tpitch, tyaw,
        label="lateral-at-safe-z"
    )
    if code != 0:
        print(f"Failed at step 2 (lateral move), code={code}")
        return code

    time.sleep(0.2)

    # Step 3: descend vertically to target Z
    code = move_pose(
        arm,
        tx, ty, tz,
        troll, tpitch, tyaw,
        label="descend-to-target"
    )
    if code != 0:
        print(f"Failed at step 3 (descend), code={code}")
        return code

    return 0

def main():
    pose_path = Path(POSE_FILE)
    if not pose_path.exists():
        print(f"Pose file not found: {POSE_FILE}")
        return

    poses = json.loads(pose_path.read_text(encoding="utf-8"))
    if not poses:
        print("No poses found in file.")
        return

    arm = XArmAPI(ROBOT_IP)
    arm.connect()

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    print(f"Loaded {len(poses)} poses from {POSE_FILE}")
    print("Safer replay mode: raise -> lateral move -> descend")
    print("Make sure the workspace is clear.\n")

    cmd = input("Press ENTER to start replay, or type 'q' to quit: ").strip().lower()
    if cmd == "q":
        arm.disconnect()
        return

    for i, item in enumerate(poses, start=1):
        tcp = item["tcp_pose_mm_rad"]

        print(f"\n=== Pose {i}/{len(poses)} ===")
        print(f"Target: x={tcp['x']:.2f}, y={tcp['y']:.2f}, z={tcp['z']:.2f}, "
              f"roll={tcp['roll']:.4f}, pitch={tcp['pitch']:.4f}, yaw={tcp['yaw']:.4f}")

        if CONFIRM_EACH_POSE:
            cmd = input("Press ENTER to move, or type 'q' to stop: ").strip().lower()
            if cmd == "q":
                print("Replay stopped by user.")
                break

        code = move_two_stage(arm, tcp)
        if code != 0:
            print(f"Failed to reach pose {i}, code={code}")
            break

        print(f"Arrived at pose {i}. Waiting {WAIT_AFTER_MOVE:.1f}s...")
        time.sleep(WAIT_AFTER_MOVE)

    print("\nReplay finished.")
    arm.disconnect()

if __name__ == "__main__":
    main()