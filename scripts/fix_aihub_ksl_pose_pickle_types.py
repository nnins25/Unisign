#!/usr/bin/env python
"""AIHub_KSL pose pkl 내부 frame 자료형을 numpy.ndarray로 보정한다."""

import argparse
import pickle
from pathlib import Path

import numpy as np


EXPECTED_KEYPOINT_SHAPE = (1, 133, 2)
EXPECTED_SCORE_SHAPE = (1, 133)


def convert_frames(frames, expected_shape, label):
    converted = []
    for idx, frame in enumerate(frames):
        arr = np.asarray(frame, dtype=np.float32)
        if arr.shape != expected_shape:
            raise ValueError(f"{label}[{idx}] shape 오류: {arr.shape}, expected={expected_shape}")
        converted.append(arr)
    return converted


def fix_one(path, dry_run=False):
    with path.open("rb") as f:
        pose = pickle.load(f)

    if "keypoints" not in pose or "scores" not in pose:
        raise ValueError("pose pkl에 keypoints 또는 scores key가 없습니다.")

    keypoints = convert_frames(pose["keypoints"], EXPECTED_KEYPOINT_SHAPE, "keypoints")
    scores = convert_frames(pose["scores"], EXPECTED_SCORE_SHAPE, "scores")

    if len(keypoints) != len(scores):
        raise ValueError(f"frame 수 불일치: keypoints={len(keypoints)}, scores={len(scores)}")

    if not dry_run:
        pose["keypoints"] = keypoints
        pose["scores"] = scores
        with path.open("wb") as f:
            pickle.dump(pose, f)

    return len(keypoints), type(keypoints[0]).__name__, type(scores[0]).__name__


def main():
    parser = argparse.ArgumentParser(description="AIHub_KSL pose pkl frame type fixer")
    parser.add_argument("--pose_dir", required=True, help="pose_format pkl 폴더")
    parser.add_argument("--max-files", type=int, default=0, help="처리할 최대 pkl 수. 0이면 전체")
    parser.add_argument("--dry-run", action="store_true", help="파일을 덮어쓰지 않고 검사만 수행")
    args = parser.parse_args()

    pose_dir = Path(args.pose_dir)
    if not pose_dir.is_dir():
        raise SystemExit(f"--pose_dir가 디렉터리가 아닙니다: {pose_dir}")

    files = sorted(pose_dir.glob("*.pkl"))
    if args.max_files:
        files = files[: args.max_files]

    print("=" * 80)
    print("AIHub_KSL pose pkl type fix")
    print("=" * 80)
    print(f"- pose_dir: {pose_dir}")
    print(f"- 대상 파일 수: {len(files)}")
    print(f"- dry_run: {args.dry_run}")

    failed = []
    fixed = 0
    for idx, path in enumerate(files, start=1):
        try:
            frames, kp_type, score_type = fix_one(path, dry_run=args.dry_run)
            fixed += 1
            if fixed <= 3:
                print(f"- OK: {path.name}, frames={frames}, keypoints[0]={kp_type}, scores[0]={score_type}")
        except Exception as exc:
            failed.append(f"{path}\t{type(exc).__name__}: {exc}")
            if len(failed) <= 10:
                print(f"FAILED: {path.name}: {type(exc).__name__}: {exc}")

        if idx % 1000 == 0:
            print(f"- progress: {idx}/{len(files)}")

    failure_path = Path("failed_fix_pose_pickle_types.txt")
    failure_path.write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")

    print("")
    print("=" * 80)
    print("요약")
    print("=" * 80)
    print(f"- 처리 성공: {fixed}")
    print(f"- 실패: {len(failed)}")
    print(f"- 실패 목록: {failure_path.resolve()}")


if __name__ == "__main__":
    main()


# 실행 예시:
# python scripts/fix_aihub_ksl_pose_pickle_types.py --pose_dir ".\dataset\AIHub_KSL\pose_format"
# python scripts/fix_aihub_ksl_pose_pickle_types.py --pose_dir ".\dataset\AIHub_KSL\pose_format" --dry-run --max-files 10
