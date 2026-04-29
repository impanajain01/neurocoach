import cv2
from ultralytics import YOLO

# --- SETUP ---
VIDEO_PATH = "cricket.mp4"
OUTPUT_VIDEO = "tracking_output.mp4"

# Load pre-trained YOLOv8 model (downloads automatically first time)
model = YOLO("yolov8n.pt")

# --- VIDEO SETUP ---
cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_count = 0
player_positions = {}  # stores positions per player ID

print("✅ Starting player tracking...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking (ByteTrack keeps player IDs consistent)
    results = model.track(frame, persist=True, classes=[0])  # class 0 = person

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()     # bounding boxes
        ids   = results[0].boxes.id.cpu().numpy()       # player IDs
        
        for box, player_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            pid = int(player_id)

            # Calculate center of player
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Store position history per player
            if pid not in player_positions:
                player_positions[pid] = []
            player_positions[pid].append((cx, cy))

            # Draw bounding box and ID on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    out.write(frame)
    frame_count += 1

    if frame_count % 20 == 0:
        print(f"   Processed {frame_count} frames...")

cap.release()
out.release()

print(f"\n✅ Tracking done!")
print(f"   Total frames     : {frame_count}")
print(f"   Players detected : {len(player_positions)}")
for pid, positions in player_positions.items():
    print(f"   Player {pid}        : {len(positions)} position points")