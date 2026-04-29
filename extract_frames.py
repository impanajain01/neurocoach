import cv2
import os

# --- CONFIG ---
VIDEO_PATH = "cricket.mp4"   # put your video file name here
OUTPUT_FOLDER = "frames"
SAVE_EVERY_N_FRAMES = 5      # saves 1 frame every 5 frames

# --- CREATE OUTPUT FOLDER ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- OPEN VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error: Could not open video. Check the file name and path.")
    exit()

frame_count = 0
saved_count = 0

print("✅ Video opened successfully. Extracting frames...")

while True:
    ret, frame = cap.read()

    if not ret:
        break  # no more frames

    if frame_count % SAVE_EVERY_N_FRAMES == 0:
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Done! {saved_count} frames saved in the '{OUTPUT_FOLDER}' folder.")