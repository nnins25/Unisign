import pickle
from pathlib import Path

pose_dir = Path("dataset/AIHub_KSL/pose_format")
pkl_files = list(pose_dir.glob("*.pkl"))
print(f"Files: {len(pkl_files)}")

total = 0
for p in pkl_files:
    with open(p, "rb") as f:
        d = pickle.load(f)
    total += len(d["keypoints"])

fps = 30
secs = total / fps
hours = secs / 3600
print(f"Total frames: {total}")
print(f"Avg frames/clip: {total/len(pkl_files):.1f}")
print(f"Total hours @ {fps}fps: {hours:.2f}h")
