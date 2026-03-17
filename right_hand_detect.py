import pyzed.sl as sl
import cv2
import numpy as np
import time
import math

def main():
    # 1. 初始化 ZED 相机
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  
    init_params.coordinate_units = sl.UNIT.METER         
    # 🌟 修复警告：将 ULTRA 改为 NEURAL (新版 SDK 推荐，精度更高)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL         

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("无法打开 ZED 相机，请检查连接！")
        return

    # 🌟 核心修复：在启用人体追踪前，必须先启用位置追踪 (Positional Tracking)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(positional_tracking_parameters) != sl.ERROR_CODE.SUCCESS:
        print("启用位置追踪失败！")
        zed.close()
        return

    # 2. 配置并启用人体骨骼追踪 (BODY_34)
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.body_format = sl.BODY_FORMAT.BODY_34
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    
    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("启用人体追踪失败！")
        zed.close()
        return

    # 运行时参数及数据容器
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    bodies = sl.Bodies()

    # 3. 稳定性判断相关的状态变量
    RIGHT_HAND_INDEX = 15          # BODY_34 模型中 RIGHT_HAND 的索引
    STABILITY_THRESHOLD = 0.03     # 稳定阈值：3 厘米 (0.03米) 内认为没有移动
    TARGET_DURATION = 1.5          # 目标稳定时间：1.5 秒

    last_hand_pos = None           # 上一帧的右手 3D 坐标
    stable_start_time = 0          # 开始保持稳定的时间戳
    already_triggered = False      # 防止 1.5s 后被疯狂重复触发的标志位

    print("开始实时检测... 请按 'Esc' 键退出。")

    while True:
        # 获取最新一帧
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT) # 获取左眼彩色图像用于显示
            zed.retrieve_bodies(bodies)             # 获取骨骼数据
            
            # 将 ZED 图像格式转换为 OpenCV 可用的格式 (BGRA)
            cv_image = image.get_data()

            # 如果检测到新的人体数据
            if bodies.is_new and len(bodies.body_list) > 0:
                # 为了简单起见，我们追踪画面中的第一个人
                body = bodies.body_list[0]
                
                # 获取 3D 和 2D 坐标
                hand_3d = body.keypoint[RIGHT_HAND_INDEX]
                hand_2d = body.keypoint_2d[RIGHT_HAND_INDEX]

                # 确保关键点没有被遮挡（有效坐标）
                if not math.isnan(hand_3d[0]) and not math.isnan(hand_2d[0]):
                    
                    # 当前位置
                    current_pos = np.array([hand_3d[0], hand_3d[1], hand_3d[2]])
                    
                    # 状态文本与圆圈颜色（默认为红色：不稳定）
                    status_text = "Moving..."
                    color = (0, 0, 255) 

                    if last_hand_pos is not None:
                        # 计算当前帧与上一帧手部位置的欧氏距离
                        distance = np.linalg.norm(current_pos - last_hand_pos)

                        if distance < STABILITY_THRESHOLD:
                            # 如果移动距离小于阈值，说明手是稳定的
                            if stable_start_time == 0:
                                stable_start_time = time.time()
                            
                            elapsed_time = time.time() - stable_start_time
                            status_text = f"Stable: {elapsed_time:.1f}s"
                            color = (0, 255, 255) # 黄色：正在稳定中

                            # 满足 1.5 秒条件
                            if elapsed_time >= TARGET_DURATION:
                                color = (0, 255, 0) # 绿色：目标达成
                                status_text = "TRIGGERED!"
                                
                                if not already_triggered:
                                    # 触发输出！
                                    print(f"✅ 目标稳定 1.5s! 右手 3D 坐标 -> X: {current_pos[0]:.3f}, Y: {current_pos[1]:.3f}, Z: {current_pos[2]:.3f} (米)")
                                    already_triggered = True
                        else:
                            # 移动幅度过大，打断稳定状态，重置计时器
                            stable_start_time = 0
                            already_triggered = False

                    # 更新上一帧的位置
                    last_hand_pos = current_pos

                    # --- OpenCV 画面绘制 ---
                    # 画出右手的点
                    cv2.circle(cv_image, (int(hand_2d[0]), int(hand_2d[1])), 8, color, -1)
                    # 在手部旁边显示状态文本
                    cv2.putText(cv_image, status_text, (int(hand_2d[0]) + 15, int(hand_2d[1]) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # 如果手被遮挡（失去坐标），重置状态
                    last_hand_pos = None
                    stable_start_time = 0
                    already_triggered = False

            # 显示实时画面
            cv2.imshow("ZED BODY_34 Right Hand Tracker", cv_image)

            # 按下 'Esc' 键退出
            if cv2.waitKey(10) & 0xFF == 27:
                break

    # 释放资源
    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()