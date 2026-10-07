import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_draw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
pose=mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.8)
if cap.isOpened()==False:
    raise SystemExit("摄像头打开失败")
while True:
    ret,frame=cap.read()
    if  not ret:
        print("fail to read")
        break
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    end=pose.process(rgb_frame)
    if end.pose_landmarks is not None:
        mp_draw.draw_landmarks(frame,end.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    cv2.imshow("POSE",frame)
    key=cv2.waitKey(1)
    if key==ord("q"):
        break
cap.release()
pose.close()
cv2.destroyAllWindows()

