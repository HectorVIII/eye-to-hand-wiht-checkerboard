import pyzed.sl as sl
import cv2
import numpy as np
import time
import math

def main():
    print("正在初始化 ZED 相机 (高精度模式，加载可能稍慢)...")
    # 1. 初始化 ZED 相机 (画质与深度拉满)
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # 🌟 提升至 1080p，边缘更精准
    init_params.coordinate_units = sl.UNIT.METER         
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL         # 🌟 必须使用 NEURAL 模式

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("无法打开 ZED 相机，请检查连接！")
        return

    # 启用位置追踪 (原点将被钉死在启动时左眼的光心位置)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(positional_tracking_parameters) != sl.ERROR_CODE.SUCCESS:
        print("启用位置追踪失败！")
        zed.close()
        return

    # 2. 配置人体骨骼追踪 (开启最高精度 AI 模型)
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.body_format = sl.BODY_FORMAT.BODY_34
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE # 🌟 开启高精度骨骼模型
    
    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("启用人体追踪失败！")
        zed.close()
        return

    # 运行时参数及数据容器
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    bodies = sl.Bodies()

    # 3. 稳定性判断与滤波变量
    RIGHT_HAND_INDEX = 15          
    STABILITY_THRESHOLD = 0.03     # 容差范围：3 厘米
    TARGET_DURATION = 1.5          # 目标时间：1.5 秒
    CONFIDENCE_THRESHOLD = 60.0    # 🌟 置信度阈值 (0-100)，低于此值的数据被视为噪点丢弃

    last_hand_pos = None           
    stable_start_time = 0          
    already_triggered = False      
    stable_positions_buffer = []   # 🌟 滑动平均滤波器的数据池

    print("\n✅ 初始化完成！开始实时高精度检测...")
    print("⚠️ 切记：请以相机【左眼镜头中心】作为 (0,0,0) 原点进行测量验证！\n")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT) 
            zed.retrieve_bodies(bodies)             
            
            cv_image = image.get_data()

            if bodies.is_new and len(bodies.body_list) > 0:
                body = bodies.body_list[0]
                
                hand_3d = body.keypoint[RIGHT_HAND_INDEX]
                hand_2d = body.keypoint_2d[RIGHT_HAND_INDEX]
                hand_confidence = body.keypoint_confidence[RIGHT_HAND_INDEX] # 获取关键点置信度

                # 🌟 数据清洗：排除遮挡 (NaN) 且 置信度必须大于 60
                if not math.isnan(hand_3d[0]) and not math.isnan(hand_2d[0]) and hand_confidence > CONFIDENCE_THRESHOLD:
                    
                    current_pos = np.array([hand_3d[0], hand_3d[1], hand_3d[2]])
                    
                    status_text = f"Moving (Conf:{hand_confidence:.0f}%)"
                    color = (0, 0, 255) 

                    if last_hand_pos is not None:
                        distance = np.linalg.norm(current_pos - last_hand_pos)

                        if distance < STABILITY_THRESHOLD:
                            # 如果刚开始稳定，记录时间并清空缓冲池
                            if stable_start_time == 0:
                                stable_start_time = time.time()
                                stable_positions_buffer = [] 
                            
                            # 🌟 将当前有效坐标加入缓冲池
                            stable_positions_buffer.append(current_pos)
                            
                            elapsed_time = time.time() - stable_start_time
                            status_text = f"Stable: {elapsed_time:.1f}s"
                            color = (0, 255, 255) 

                            if elapsed_time >= TARGET_DURATION:
                                color = (0, 255, 0) 
                                status_text = "TRIGGERED!"
                                
                                if not already_triggered:
                                    # 🌟 核心：计算这 1.5 秒内所有收集到的坐标的【平均值】
                                    avg_pos = np.mean(stable_positions_buffer, axis=0)
                                    buffer_size = len(stable_positions_buffer)
                                    
                                    print(f"✅ 目标稳定 1.5s! (基于 {buffer_size} 帧数据平均计算)")
                                    print(f"👉 最终高精度 3D 坐标 -> X: {avg_pos[0]:.4f}, Y: {avg_pos[1]:.4f}, Z: {avg_pos[2]:.4f} (米)\n")
                                    
                                    already_triggered = True
                        else:
                            # 移动幅度过大，打断稳定状态，重置所有状态
                            stable_start_time = 0
                            already_triggered = False
                            stable_positions_buffer = []

                    last_hand_pos = current_pos

                    # OpenCV 画面绘制
                    cv2.circle(cv_image, (int(hand_2d[0]), int(hand_2d[1])), 8, color, -1)
                    cv2.putText(cv_image, status_text, (int(hand_2d[0]) + 15, int(hand_2d[1]) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # 如果手被遮挡或置信度太低，视为无效，重置状态
                    last_hand_pos = None
                    stable_start_time = 0
                    already_triggered = False
                    stable_positions_buffer = []

            # 绘制屏幕中心十字星，辅助对齐左眼光心位置
            h, w = cv_image.shape[:2]
            cv2.line(cv_image, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 1)
            cv2.line(cv_image, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 1)
            cv2.putText(cv_image, "Left Eye Center (0,0)", (w//2 + 10, h//2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("ZED BODY_34 High-Precision Tracker", cv_image)

            if cv2.waitKey(10) & 0xFF == 27:
                break

    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()