import pyzed.sl as sl
import cv2
import mediapipe as mp
import numpy as np
import math

def main():
    # ==========================================
    # 1. 初始化 MediaPipe Hands
    # ==========================================
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # 设定只检测一只手，可根据需求调整
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # ==========================================
    # 2. 初始化 ZED 相机
    # ==========================================
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 使用 720p 以保证帧率
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式设为 ULTRA 以获取最高精度
    init_params.coordinate_units = sl.UNIT.MILLIMETER    # 单位：毫米
    init_params.depth_minimum_distance = 200             # 最小深度距离设为 20cm

    # 打开相机
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"相机打开失败: {err}")
        exit(1)

    # 创建用于存放图像和点云的容器
    image = sl.Mat()
    point_cloud = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    print("相机已启动，按 'q' 键退出...")

    # ==========================================
    # 3. 主循环
    # ==========================================
    while True:
        # 抓取新的一帧
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # 获取左眼图像和点云数据
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # 将 ZED 图像格式转换为 OpenCV 格式 (BGRA -> BGR)
            image_cv = image.get_data()
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGB) # MediaPipe 需要 RGB

            # 图像的高和宽
            h, w, _ = image_rgb.shape

            # 处理图像，进行手部检测
            results = hands.process(image_rgb)

            # 切回 BGR 以便使用 OpenCV 显示
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 判断是否为右手 (注意：前置/后置摄像头可能会导致左右手镜像，可根据实际测试调整)
                    label = handedness.classification[0].label
                    
                    if label == "Right": # 或者 "Left"，取决于你的物理视角
                        # 获取食指指尖 (INDEX_FINGER_TIP, 节点索引为 8) 的 2D 归一化坐标
                        landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        
                        # 将归一化坐标转换为实际像素坐标 (u, v)
                        u = int(landmark.x * w)
                        v = int(landmark.y * h)

                        # 边界保护
                        if 0 <= u < w and 0 <= v < h:
                            # 从点云中获取该像素点的 3D 数据
                            # point3D 是一个包含 [X, Y, Z, 颜色] 的数组
                            err, point3D = point_cloud.get_value(u, v)
                            
                            if err == sl.ERROR_CODE.SUCCESS:
                                x, y, z = point3D[0], point3D[1], point3D[2]
                                
                                # 判断深度数据是否有效 (剔除 NaN 值)
                                if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                                    text = f"Right Index Tip: X={x:.1f}, Y={y:.1f}, Z={z:.1f} mm"
                                    
                                    # 在图像上绘制绿点和坐标文本
                                    cv2.circle(image_bgr, (u, v), 5, (0, 255, 0), -1)
                                    cv2.putText(image_bgr, text, (u + 10, v - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                else:
                                    cv2.putText(image_bgr, "Depth invalid (NaN)", (u + 10, v - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # 可视化所有手部关键点
                        mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 显示画面
            cv2.imshow("ZED + MediaPipe Hand Tracking", image_bgr)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 关闭相机并释放资源
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()