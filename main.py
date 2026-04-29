from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- APP SETUP ---
app = FastAPI(
    title="NeuroCoach API",
    description="AI Cricket Player Analysis API",
    version="1.0.0"
)

# Allow React frontend to talk to this API later (Week 5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve heatmap images as static files
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Load YOLO model once at startup (not on every request)
model = YOLO("yolov8n.pt")

# --- HEALTH CHECK ROUTE ---
@app.get("/")
def home():
    return {
        "message": "NeuroCoach API is running 🏏",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/docs"]
    }

# --- MAIN ANALYSIS ROUTE ---
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):

    # Step 1: Validate file type
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a .mp4, .avi or .mov video."
        )

    # Step 2: Save uploaded video temporarily
    video_path = f"outputs/uploaded_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"✅ Video received: {file.filename}")

    # Step 3: Run player tracking
    print("🔍 Running player tracking...")
    player_positions, total_frames, width, height = track_players(video_path)

    # Step 4: Filter real players (50+ points)
    real_players = {
        pid: pos
        for pid, pos in player_positions.items()
        if len(pos) >= 50
    }

    if not real_players:
        raise HTTPException(
            status_code=422,
            detail="No players detected consistently. Try a clearer video."
        )

    # Step 5: Generate heatmap for most tracked player
    best_player_id = max(real_players, key=lambda x: len(real_players[x]))
    heatmap_filename = f"outputs/heatmap_{file.filename}.png"
    generate_heatmap(real_players, best_player_id, width, height, heatmap_filename)

    # Step 6: Build player stats
    player_stats = []
    for pid, positions in real_players.items():
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Coverage area (bounding box of movement)
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        coverage = round((x_range * y_range) / (width * height) * 100, 1)

        player_stats.append({
            "player_id": pid,
            "frames_tracked": len(positions),
            "coverage_percent": coverage,
            "avg_x_position": round(sum(xs) / len(xs)),
            "avg_y_position": round(sum(ys) / len(ys)),
        })

    # Sort by frames tracked
    player_stats.sort(key=lambda x: x["frames_tracked"], reverse=True)

    # Step 7: Return results
    return {
        "status": "success",
        "video": file.filename,
        "total_frames": total_frames,
        "real_players_detected": len(real_players),
        "best_tracked_player": best_player_id,
        "heatmap_url": f"/{heatmap_filename}",
        "player_stats": player_stats
    }


# --- HELPER: TRACK PLAYERS ---
def track_players(video_path):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    player_positions = {}
    frame_count = 0

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
    return player_positions, frame_count, width, height


# --- HELPER: GENERATE HEATMAP ---
def generate_heatmap(real_players, best_player_id, width, height, output_path):
    positions = real_players[best_player_id]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # Movement trail
    axes[0].set_facecolor('#1a1a2e')
    axes[0].scatter(xs, ys, c=range(len(xs)), cmap='plasma', s=8, alpha=0.6)
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    axes[0].set_title(f'Player {best_player_id} — Movement Trail', color='white', fontsize=13)
    axes[0].tick_params(colors='white')

    # Density heatmap
    axes[1].set_facecolor('#1a1a2e')
    sns.kdeplot(x=xs, y=ys, fill=True, cmap='YlOrRd',
                bw_adjust=0.5, thresh=0.05, ax=axes[1])
    axes[1].set_xlim(0, width)
    axes[1].set_ylim(height, 0)
    axes[1].set_title(f'Player {best_player_id} — Movement Heatmap', color='white', fontsize=13)
    axes[1].tick_params(colors='white')

    plt.suptitle('NeuroCoach — Cricket Player Analysis', color='white', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()


# --- RUN SERVER ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)