from report_generator import create_pdf_report

# Use the same stats from your Week 4 API response
test_stats = [
    {"player_id": 2,  "frames_tracked": 448, "coverage_percent": 15.4, "avg_x_position": 320, "avg_y_position": 1221},
    {"player_id": 25, "frames_tracked": 329, "coverage_percent": 1.8,  "avg_x_position": 202, "avg_y_position": 676},
    {"player_id": 4,  "frames_tracked": 287, "coverage_percent": 1.2,  "avg_x_position": 954, "avg_y_position": 685},
    {"player_id": 35, "frames_tracked": 276, "coverage_percent": 0.2,  "avg_x_position": 150, "avg_y_position": 584},
    {"player_id": 51, "frames_tracked": 256, "coverage_percent": 6.5,  "avg_x_position": 878, "avg_y_position": 1277},
]

create_pdf_report(
    player_stats=test_stats,
    total_frames=474,
    video_name="cricket.mp4",
    output_path="outputs/coaching_report.pdf"
)

print("✅ Open outputs/coaching_report.pdf to see your report!")