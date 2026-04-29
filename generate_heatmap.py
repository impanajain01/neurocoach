import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIG ---
VIDEO_PATH = "cricket.mp4"
TARGET_PLAYER_ID = 2       # most consistent player
MIN_POINTS = 50            # ignore ghost players below this

# --- RE-RUN TRACKING TO GET POSITIONS ---
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

player_positions = {}
frame_count = 0

print("✅ Re-collecting player positions...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy()

        for box, player_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            pid = int(player_id)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if pid not in player_positions:
                player_positions[pid] = []
            player_positions[pid].append((cx, cy))

    frame_count += 1

cap.release()

# --- FILTER: keep only real players (50+ points) ---
real_players = {pid: pos for pid, pos in player_positions.items() if len(pos) >= MIN_POINTS}
print(f"✅ Real players found: {list(real_players.keys())}")

# --- GENERATE HEATMAP FOR TARGET PLAYER ---
if TARGET_PLAYER_ID not in real_players:
    # fallback to player with most points
    TARGET_PLAYER_ID = max(real_players, key=lambda x: len(real_players[x]))
    print(f"⚠️  Player 2 not found, using Player {TARGET_PLAYER_ID} instead")

positions = real_players[TARGET_PLAYER_ID]
xs = [p[0] for p in positions]
ys = [p[1] for p in positions]

print(f"✅ Generating heatmap for Player {TARGET_PLAYER_ID} ({len(positions)} points)...")

# --- PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a2e')

# Left: movement trail
axes[0].set_facecolor('#1a1a2e')
axes[0].scatter(xs, ys, c=range(len(xs)), cmap='plasma', s=8, alpha=0.6)
axes[0].set_xlim(0, width)
axes[0].set_ylim(height, 0)
axes[0].set_title(f'Player {TARGET_PLAYER_ID} — Movement Trail', color='white', fontsize=13)
axes[0].tick_params(colors='white')
for spine in axes[0].spines.values():
    spine.set_edgecolor('#444')

# Right: density heatmap
axes[1].set_facecolor('#1a1a2e')
sns.kdeplot(x=xs, y=ys, fill=True, cmap='YlOrRd',
            bw_adjust=0.5, thresh=0.05, ax=axes[1])
axes[1].set_xlim(0, width)
axes[1].set_ylim(height, 0)
axes[1].set_title(f'Player {TARGET_PLAYER_ID} — Movement Heatmap', color='white', fontsize=13)
axes[1].tick_params(colors='white')
for spine in axes[1].spines.values():
    spine.set_edgecolor('#444')

plt.suptitle('NeuroCoach — Cricket Player Analysis', color='white', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

print("✅ Heatmap saved as heatmap.png")