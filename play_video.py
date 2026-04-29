import cv2

cap = cv2.VideoCapture("pose_output.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("NeuroCoach - Pose Detection", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()