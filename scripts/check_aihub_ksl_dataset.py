#!/usr/bin/env python
"""AIHub_KSL Uni-Sign 데이터셋 sanity check 스크립트."""

import argparse
import gzip
import pickle
from collections import Counter
from pathlib import Path


def load_label(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def nested_shape(value):
    """list/numpy array 모두 대략적인 shape 문자열로 출력한다."""
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if not isinstance(value, list):
        return f"not-list:{type(value).__name__}"
    shape = []
    cur = value
    while isinstance(cur, list):
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(shape)


def sample_meta(name):
    tokens = name.split("_")
    signer = tokens[0].replace("signer", "") if tokens else "UNKNOWN"
    view = tokens[-1] if tokens else "UNKNOWN"
    sentence_id = "_".join(tokens[1:-1]) if len(tokens) > 2 else "UNKNOWN"
    return signer, sentence_id, view


def summarize(split_name, labels):
    signer_counter = Counter()
    view_counter = Counter()
    sentence_ids = set()
    empty_gloss = []

    for name, sample in labels.items():
        signer, sentence_id, view = sample_meta(name)
        signer_counter[signer] += 1
        view_counter[view] += 1
        sentence_ids.add(sentence_id)
        if not sample.get("gloss"):
            empty_gloss.append(name)

    print(f"- {split_name}: samples={len(labels)}, unique_sentences={len(sentence_ids)}")
    print(f"  - signer 분포: {dict(sorted(signer_counter.items()))}")
    print(f"  - view 분포: {dict(sorted(view_counter.items()))}")
    print(f"  - empty gloss samples: {len(empty_gloss)}")
    if empty_gloss[:5]:
        print(f"  - empty gloss 예시: {empty_gloss[:5]}")
    return sentence_ids, empty_gloss


def check_pose_files(labels, pose_dir, limit=3):
    print("")
    print("=" * 80)
    print("sample / pose pkl 확인")
    print("=" * 80)

    missing = []
    checked = 0
    for name, sample in labels.items():
        pose_name = sample["video_path"].replace(".mp4", ".pkl")
        pose_path = pose_dir / pose_name
        if not pose_path.exists():
            missing.append(str(pose_path))
            continue

        if checked < limit:
            with pose_path.open("rb") as f:
                pose = pickle.load(f)
            keypoints = pose.get("keypoints", [])
            scores = pose.get("scores", [])
            print(f"- sample: {name}")
            print(f"  - video_path: {sample['video_path']}")
            print(f"  - pose_path: {pose_path}")
            print(f"  - gloss: {sample.get('gloss')}")
            print(f"  - frames: {len(keypoints)}")
            if keypoints:
                print(f"  - type(keypoints[0]) = {type(keypoints[0]).__module__}.{type(keypoints[0]).__name__}")
                print(f"  - keypoints[0] shape: {nested_shape(keypoints[0])}")
                if isinstance(keypoints[0], list):
                    print("  - WARNING: keypoints[0]가 list입니다. datasets.py 다차원 인덱싱에서 실패할 수 있습니다.")
            if scores:
                print(f"  - type(scores[0]) = {type(scores[0]).__module__}.{type(scores[0]).__name__}")
                print(f"  - scores[0] shape: {nested_shape(scores[0])}")
                if isinstance(scores[0], list):
                    print("  - WARNING: scores[0]가 list입니다. datasets.py 다차원 인덱싱에서 실패할 수 있습니다.")
            checked += 1

    print(f"- missing pose files: {len(missing)}")
    if missing[:10]:
        print(f"- missing 예시: {missing[:10]}")
    return missing


def main():
    parser = argparse.ArgumentParser(description="AIHub_KSL dataset sanity check")
    parser.add_argument("--data_dir", default="./data/AIHub_KSL", help="labels.train/dev/test 폴더")
    parser.add_argument("--pose_dir", default="./dataset/AIHub_KSL/pose_format", help="pose pkl 폴더")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pose_dir = Path(args.pose_dir)

    labels = {
        "train": load_label(data_dir / "labels.train"),
        "dev": load_label(data_dir / "labels.dev"),
        "test": load_label(data_dir / "labels.test"),
    }

    print("=" * 80)
    print("AIHub_KSL label 요약")
    print("=" * 80)
    sentence_sets = {}
    empty_gloss_all = []
    for split_name in ["train", "dev", "test"]:
        sentence_ids, empty_gloss = summarize(split_name, labels[split_name])
        sentence_sets[split_name] = sentence_ids
        empty_gloss_all.extend([(split_name, item) for item in empty_gloss])

    print("")
    print("=" * 80)
    print("split sentence_id 누수 확인")
    print("=" * 80)
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for left, right in pairs:
        overlap = sentence_sets[left] & sentence_sets[right]
        print(f"- {left}/{right} overlap: {len(overlap)}")
        if overlap:
            print(f"  - 예시: {sorted(list(overlap))[:10]}")

    check_pose_files(labels["train"], pose_dir, limit=3)

    total_samples = sum(len(item) for item in labels.values())
    total_sentences = len(set().union(*sentence_sets.values()))
    print("")
    print("=" * 80)
    print("최종 요약")
    print("=" * 80)
    print(f"- total samples: {total_samples}")
    print(f"- total unique sentences: {total_sentences}")
    print(f"- total empty gloss: {len(empty_gloss_all)}")


if __name__ == "__main__":
    main()


# 실행 예시:
# python scripts/check_aihub_ksl_dataset.py --data_dir ".\Uni-Sign-main\data\AIHub_KSL" --pose_dir ".\Uni-Sign-main\dataset\AIHub_KSL\pose_format"
