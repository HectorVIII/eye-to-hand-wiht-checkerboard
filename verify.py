import cv2
import numpy as np
import pyzed.sl as sl
import time
from xarm.wrapper import XArmAPI

# ================= 配置参数 (替换为你的实际参数) =================
XARM_IP = "192.168.1.225"  # xArm7 的实际 IP 地址

CONF_THR = 40              # 关键点置信度阈值
EMA_ALPHA = 0.2            # EMA 平滑系数 (0-1)
POS_TOL = 0.015            # 稳定性容差，单位：米 (1.5厘米)
FPS = 60                   # 相机帧率
STABLE_SECONDS = 2.0       # 需要保持稳定的时间 (秒)
STABLE_FRAMES_REQUIRED = int(FPS * STABLE_SECONDS)  # 60 * 2 = 120 帧

# 手眼标定矩阵 (相机坐标系到机械臂基坐标系的转换)
# 这里使用单位矩阵和零平移作为占位符，请务必替换为真实标定数据！
# ================= 手眼标定矩阵 =================
T_cam2base = np.array([
    [-0.691532644,  0.326676329, -0.644255523,  0.853619759],
    [ 0.722295719,  0.302279846, -0.622025553,  0.707727934],
    [-0.008455565, -0.895493981, -0.444993295,  0.586156444],
    [ 0.0,          0.0,          0.0,          1.0        ],
], dtype=np.float64)

# 自动提取 3x3 旋转矩阵和 3x1 平移向量 (米)
R_cb = T_cam2base[:3, :3]  
t_cb = T_cam2base[:3, 3]   
# ===============================================

SAFE_Z_MIN = 100.0         # 机械臂安全高度下限 (mm)
SAFE_Z_MAX = 600.0         # 机械臂安全高度上限 (mm)

# 机械臂到达目标点时的固定姿态 (Roll, Pitch, Yaw)，单位: mm/度 或 mm/弧度 (取决于你的xarm设置)
P2_ORI = dict(roll=180.0, pitch=0.0, yaw=0.0) 
# ===============================================================

def detect_right_hand_target():
    """
    使用 ZED2 检测右手，稳定 2 秒后映射到机械臂基坐标系 (mm)。
    返回: dict(x=..., y=..., z=..., roll=..., pitch=..., yaw=...)
    """
    zed = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720
    ip.camera_fps = FPS
    ip.depth_mode = sl.DEPTH_MODE.NEURAL
    ip.coordinate_units = sl.UNIT.METER
    ip.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    err = zed.open(ip)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED 相机打开失败: {err}")

    ptp = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(ptp)

    btp = sl.BodyTrackingParameters()
    btp.enable_tracking = True
    btp.enable_body_fitting = False
    btp.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    btp.body_format = sl.BODY_FORMAT.BODY_34
    zed.enable_body_tracking(btp)

    bodies = sl.Bodies()
    brt = sl.BodyTrackingRuntimeParameters()
    brt.detection_confidence_threshold = 20
    rtp = sl.RuntimeParameters()
    image = sl.Mat()

    RH_IDX = 14  # BODY_34 格式中，右手的索引为 15
    ema = None
    last_ema = None
    stable_frames = 0

    print(f"\n[视觉系统] 请伸出右手并保持稳定 {STABLE_SECONDS} 秒...")

    try:
        while True:
            if zed.grab(rtp) != sl.ERROR_CODE.SUCCESS:
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            zed.retrieve_bodies(bodies, brt)
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            if bodies.is_new:
                for body in bodies.body_list:
                    kc = body.keypoint_confidence
                    if len(kc) > RH_IDX and kc[RH_IDX] > CONF_THR:
                        rh = np.array(body.keypoint[RH_IDX], dtype=float)  # 单位: 米
                        if np.any(np.isnan(rh)):
                            continue

                        # EMA 滤波平滑
                        ema = rh if ema is None else EMA_ALPHA * rh + (1 - EMA_ALPHA) * ema

                        if last_ema is None:
                            last_ema = ema.copy()
                            stable_frames = 1
                        else:
                            diff = float(np.linalg.norm(ema - last_ema))
                            last_ema = ema.copy()
                            if diff <= POS_TOL:
                                stable_frames += 1
                            else:
                                stable_frames = 1  # 移动超限，重置计数器

                        # 在画面上显示进度信息
                        progress = min(100, int((stable_frames / STABLE_FRAMES_REQUIRED) * 100))
                        cv2.putText(
                            frame,
                            f"Right Hand Stable: {progress}%",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if progress == 100 else (0, 165, 255), 2
                        )
                        
                        # 画出右手的位置
                        px, py = int(body.keypoint_2d[RH_IDX][0]), int(body.keypoint_2d[RH_IDX][1])
                        cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)

                        cv2.imshow("ZED Right Hand Detection", frame)
                        cv2.waitKey(1)

                        # 当稳定帧数达到要求时触发
                        if stable_frames >= STABLE_FRAMES_REQUIRED:
                            print(f"[视觉系统] 右手已稳定 2 秒！当前坐标：{ema}")
                            
                            # 坐标系转换: 相机坐标(m) -> 机械臂基坐标系(m) -> 毫米(mm)
                            p_base_m = R_cb @ ema + t_cb
                            x_mm, y_mm, z_mm = 1000 * p_base_m[0], 1000 * p_base_m[1], 1000 * p_base_m[2]
                            
                            # 安全限位防护
                            z_mm = max(min(z_mm, SAFE_Z_MAX), SAFE_Z_MIN)
                            
                            target_pose = dict(x=x_mm, y=y_mm, z=z_mm, **P2_ORI)
                            print(f"[视觉系统] -> 转换为机械臂目标点: {target_pose}")
                            
                            return target_pose

            # 保持画面刷新
            cv2.imshow("ZED Right Hand Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                print("[视觉系统] 手动退出。")
                return None

    finally:
        try:
            zed.disable_body_tracking()
            zed.disable_positional_tracking()
        except Exception:
            pass
        zed.close()
        cv2.destroyAllWindows()


def main():
    # 1. 初始化并连接 xArm7
    print(f"[机械臂] 正在连接 xArm7 ({XARM_IP})...")
    arm = XArmAPI(XARM_IP)
    arm.motion_enable(enable=True)
    arm.set_mode(0)  # 位置控制模式
    arm.set_state(state=0)
    print("[机械臂] 连接成功，等待视觉系统指令。")

    # 2. 启动视觉检测
    target_pose = detect_right_hand_target()

    # 3. 机械臂执行移动
    if target_pose is not None:
        print("\n[机械臂] 接收到目标点，开始移动...")
        # 记录开始时间以防止阻塞死锁
        start_time = time.time()
        
        # 调用 set_position 移动机械臂 (确保与你的 SDK 设定的速度单位一致，通常是 mm/s)
        code = arm.set_position(
            x=target_pose['x'], 
            y=target_pose['y'], 
            z=target_pose['z'],
            roll=target_pose['roll'], 
            pitch=target_pose['pitch'], 
            yaw=target_pose['yaw'],
            speed=200,   # 移动速度
            mvacc=1000,  # 移动加速度
            wait=True    # 等待运动完成
        )
        
        if code == 0:
            print(f"[机械臂] 移动成功！耗时: {time.time() - start_time:.2f}秒")
        else:
            print(f"[机械臂] 移动失败，错误码: {code}")
            
    # 4. 清理资源
    arm.disconnect()
    print("[系统] 程序结束。")

if __name__ == "__main__":
    main()