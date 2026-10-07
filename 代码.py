import cv2
import mediapipe as mp


# 1. 获取MediaPipe姿态识别和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 2. 创建人体姿态识别对象
pose = mp_pose.Pose( min_detection_confidence=0.8, min_tracking_confidence=0.5)

# 3. 打开默认摄像头
cap = cv2.VideoCapture(0)

# 4. 检查摄像头是否成功打开
if not cap.isOpened():
    raise SystemExit("摄像头打开失败")

# 5. 不断读取并处理摄像头画面
while True:
    ret, frame = cap.read()

    if not ret:
        print("读取画面失败")
        break

    # OpenCV的BGR图像转换成MediaPipe使用的RGB图像
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # 识别人体关键点
    results = pose.process(rgb_frame)

    # 如果成功识别到人体，就绘制关键点和骨架
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # 显示处理后的画面
    cv2.imshow("Pose", frame)

    # 获取按键编码，按q退出循环
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# 6. 释放资源
pose.close()
cap.release()
cv2.destroyAllWindows()
