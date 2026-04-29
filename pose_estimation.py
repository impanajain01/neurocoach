import cv2
import mediapipe as mp
import csv
import os

# --- SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

VIDEO_PATH = "cricket.mp4"
OUTPUT_VIDEO = "pose_output.mp4"
CSV_FILE = "keypoints.csv"

# --- CSV SETUP ---
csv_file = open(CSV_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)

# Write header row
header = ["frame"]
for i in range(33):
    header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]  # v = visibility
csv_writer.writerow(header)

# --- VIDEO SETUP ---
cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_count = 0
detected_count = 0

print("✅ Processing video with pose estimation...")

# --- MAIN LOOP ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB (MediaPipe needs RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose detection
        results = pose.process(rgb_frame)

        # Draw skeleton if pose detected
        if results.pose_landmarks:
            detected_count += 1

            # Draw skeleton on frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Save keypoints to CSV
            row = [frame_count]
            for lm in results.pose_landmarks.landmark:
                row += [round(lm.x, 4), round(lm.y, 4), round(lm.z, 4), round(lm.visibility, 4)]
            csv_writer.writerow(row)

        # Write frame to output video
        out.write(frame)
        frame_count += 1

        # Progress update every 20 frames
        if frame_count % 20 == 0:
            print(f"   Processed {frame_count} frames...")

# --- CLEANUP ---
cap.release()
out.release()
csv_file.close()

print(f"\n✅ Done!")
print(f"   Total frames processed : {frame_count}")
print(f"   Frames with pose detected : {detected_count}")
print(f"   Skeleton video saved : {OUTPUT_VIDEO}")
print(f"   Keypoints CSV saved  : {CSV_FILE}")