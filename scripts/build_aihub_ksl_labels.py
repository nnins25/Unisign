#!/usr/bin/env python
"""
AIHub 한국어 수어 morpheme JSON을 Uni-Sign label 파일(gzip+pickle)로 변환한다.

이 스크립트는 pose pkl 파일 목록을 기준으로 label sample을 생성한다.
RGB 영상은 사용하지 않으며, video_path는 Uni-Sign S2T_Dataset의 pkl 매칭 규칙에 맞추기
위해 ".mp4" 형태의 가상 파일명으로 저장한다.
"""

import argparse
import csv
import gzip
import json
import pickle
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path


POSITIONS = ["D", "U", "R", "L", "F"]


class Reporter:
    def __init__(self):
        self.warnings = []

    def info(self, text=""):
        print(text)

    def warn(self, text):
        message = f"WARNING: {text}"
        print(message)
        self.warnings.append(message)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_signers(value):
    result = []
    for item in parse_csv(value):
        result.append(item.zfill(2) if item.isdigit() else item)
    return result


def safe_load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)


def top_keys(obj):
    if isinstance(obj, dict):
        return list(obj.keys())
    if isinstance(obj, list):
        return [f"<list length={len(obj)}>"]
    return [f"<{type(obj).__name__}>"]


def find_morpheme_dir_candidates(aihub_root, signer):
    candidates = []
    direct = aihub_root / "morpheme" / signer
    if direct.is_dir():
        candidates.append(direct)

    for raw_path in glob(str(aihub_root / "**" / signer), recursive=True):
        path = Path(raw_path)
        if path.is_dir() and "morpheme" in str(path).lower():
            candidates.append(path)

    unique = []
    seen = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def parse_pose_filename(path):
    """
    pose pkl 파일명 예:
    signer01_NIA_SL_SEN0001_D.pkl
    """
    stem = path.stem
    tokens = stem.split("_")
    if len(tokens) < 5:
        return None

    signer_token = tokens[0]
    if not signer_token.startswith("signer"):
        return None
    signer = signer_token.replace("signer", "")

    view = tokens[-1]
    if view not in POSITIONS:
        return None

    sentence_id = "_".join(tokens[1:-1])
    return {
        "name": stem,
        "signer": signer,
        "sentence_id": sentence_id,
        "view": view,
        "path": path,
    }


def parse_morpheme_filename(path):
    """
    morpheme 파일명 예:
    NIA_SL_SEN0001_REAL01_D_morpheme.json
    """
    stem = path.stem
    if stem.endswith("_morpheme"):
        stem = stem[: -len("_morpheme")]
    tokens = stem.split("_")

    view = tokens[-1] if tokens and tokens[-1] in POSITIONS else None
    signer = None
    for token in tokens:
        if token.startswith("REAL"):
            signer = token.replace("REAL", "")
            break

    sentence_id = None
    if len(tokens) >= 3:
        sentence_id = "_".join(tokens[:3])

    if not signer or not view or not sentence_id:
        return None
    return signer, sentence_id, view


def flatten_strings(value):
    """중첩 list/dict 안의 문자열 후보를 조심스럽게 모은다."""
    items = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            items.append(text)
    elif isinstance(value, list):
        for item in value:
            items.extend(flatten_strings(item))
    elif isinstance(value, dict):
        for key in ["name", "morpheme", "gloss", "text", "label", "value"]:
            if key in value:
                items.extend(flatten_strings(value[key]))
    return items


def extract_morphemes(obj, reporter, path=None):
    """
    AIHub morpheme JSON에서 형태소 sequence를 추출한다.

    우선순위:
    1. data[*].attributes[*].name
    2. data[*] 안의 name/morpheme/gloss/text/label/value 후보
    3. 최상위 morpheme/gloss/text/label/value 후보
    """
    if not isinstance(obj, dict):
        reporter.warn(f"{path}: JSON 최상위가 dict가 아니어서 skip")
        return []

    data = obj.get("data")
    if isinstance(data, list):
        morphemes = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            attrs = entry.get("attributes")
            if isinstance(attrs, list):
                for attr in attrs:
                    if isinstance(attr, dict) and isinstance(attr.get("name"), str):
                        name = attr["name"].strip()
                        if name:
                            morphemes.append(name)
            if not morphemes:
                for key in ["name", "morpheme", "gloss", "text", "label", "value"]:
                    if key in entry:
                        morphemes.extend(flatten_strings(entry[key]))
        if morphemes:
            return morphemes

    for key in ["morpheme", "morphemes", "gloss", "glosses", "text", "label", "labels"]:
        if key in obj:
            morphemes = flatten_strings(obj[key])
            if morphemes:
                return morphemes

    reporter.warn(f"{path}: 형태소 후보를 찾지 못해서 skip")
    return []


def inspect_morpheme_sample(morpheme_files, reporter):
    reporter.info("")
    reporter.info("=" * 80)
    reporter.info("morpheme JSON 샘플 구조")
    reporter.info("=" * 80)
    if not morpheme_files:
        reporter.warn("morpheme JSON 후보가 없습니다.")
        return

    sample = morpheme_files[0]
    obj = safe_load_json(sample)
    reporter.info(f"- 샘플 파일: {sample}")
    reporter.info(f"- 최상위 key: {top_keys(obj)}")
    if isinstance(obj, dict) and "data" in obj:
        data = obj["data"]
        if isinstance(data, list):
            reporter.info(f"- data: list length={len(data)}")
            if data:
                reporter.info(f"- data[0] 구조: {top_keys(data[0])}")
                if isinstance(data[0], dict) and "attributes" in data[0]:
                    reporter.info(f"- data[0].attributes 구조: {top_keys(data[0]['attributes'])}")

    morphemes = extract_morphemes(obj, reporter, sample)
    reporter.info(f"- 추출 형태소 예시: {morphemes[:20]}")


def collect_morpheme_map(aihub_root, signers, views, reporter, max_samples=0):
    morpheme_map = {}
    all_files = []

    for signer in signers:
        candidates = find_morpheme_dir_candidates(aihub_root, signer)
        reporter.info(f"- signer {signer} morpheme 폴더 후보 수: {len(candidates)}")
        for candidate in candidates:
            reporter.info(f"  - 후보: {candidate}")
        if not candidates:
            reporter.warn(f"signer {signer} morpheme 폴더를 찾지 못했습니다.")
            continue

        morph_dir = candidates[0]
        reporter.info(f"  - 사용할 경로: {morph_dir}")
        files = sorted(morph_dir.glob("*.json"))
        all_files.extend(files)

    inspect_morpheme_sample(all_files, reporter)

    read_count = 0
    for path in all_files:
        parsed = parse_morpheme_filename(path)
        if parsed is None:
            reporter.warn(f"morpheme 파일명 파싱 실패: {path.name}")
            continue
        signer, sentence_id, view = parsed
        if signer not in signers or view not in views:
            continue

        obj = safe_load_json(path)
        morphemes = extract_morphemes(obj, reporter, path)
        if not morphemes:
            continue
        morpheme_map[(signer, sentence_id, view)] = morphemes

        read_count += 1
        if max_samples and read_count >= max_samples:
            break

    reporter.info(f"- morpheme 매핑 수: {len(morpheme_map)}")
    return morpheme_map


def collect_pose_samples(pose_dir, signers, views, max_samples=0):
    samples = []
    for path in sorted(pose_dir.glob("*.pkl")):
        parsed = parse_pose_filename(path)
        if parsed is None:
            continue
        if parsed["signer"] not in signers or parsed["view"] not in views:
            continue
        samples.append(parsed)
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def make_label_samples(pose_samples, morpheme_map):
    label_samples = {}
    unmatched = []

    for pose in pose_samples:
        key = (pose["signer"], pose["sentence_id"], pose["view"])
        morphemes = morpheme_map.get(key)
        if not morphemes:
            unmatched.append({
                "pose_file": pose["path"].name,
                "signer": pose["signer"],
                "sentence_id": pose["sentence_id"],
                "view": pose["view"],
                "reason": "morpheme_not_found",
            })
            continue

        text = " ".join(morphemes)
        label_samples[pose["name"]] = {
            "name": pose["name"],
            "video_path": f"{pose['name']}.mp4",
            "text": text,
            "gloss": morphemes,
        }

    return label_samples, unmatched


def split_by_sentence(label_samples):
    sentence_ids = sorted({
        "_".join(name.split("_")[1:-1])
        for name in label_samples.keys()
    })
    total = len(sentence_ids)
    train_end = int(total * 0.8)
    dev_end = int(total * 0.9)

    split_for_sentence = {}
    for idx, sentence_id in enumerate(sentence_ids):
        if idx < train_end:
            split_for_sentence[sentence_id] = "train"
        elif idx < dev_end:
            split_for_sentence[sentence_id] = "dev"
        else:
            split_for_sentence[sentence_id] = "test"

    splits = {"train": {}, "dev": {}, "test": {}}
    for name, sample in label_samples.items():
        sentence_id = "_".join(name.split("_")[1:-1])
        split = split_for_sentence[sentence_id]
        splits[split][name] = sample
    return splits


def save_gzip_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def save_unmatched_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pose_file", "signer", "sentence_id", "view", "reason"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_split(name, samples, reporter):
    signer_counter = Counter()
    view_counter = Counter()
    sentence_ids = set()

    for sample_name in samples.keys():
        tokens = sample_name.split("_")
        signer_counter[tokens[0].replace("signer", "")] += 1
        view_counter[tokens[-1]] += 1
        sentence_ids.add("_".join(tokens[1:-1]))

    reporter.info(f"- {name}: samples={len(samples)}, unique_sentences={len(sentence_ids)}")
    reporter.info(f"  - signer 분포: {dict(sorted(signer_counter.items()))}")
    reporter.info(f"  - view 분포: {dict(sorted(view_counter.items()))}")


def main():
    parser = argparse.ArgumentParser(description="AIHub KSL morpheme -> Uni-Sign labels 변환")
    parser.add_argument("--aihub_root", required=True, help="AIHub korean root 경로")
    parser.add_argument("--pose_dir", required=True, help="변환 완료된 pose_format pkl 폴더")
    parser.add_argument("--out_data_dir", required=True, help="labels.train/dev/test 저장 폴더")
    parser.add_argument("--signers", default="01,02", help="사용할 수어자. 예: 01,02")
    parser.add_argument("--views", default="D,U,R,L,F", help="사용할 view. 예: D,U,R,L,F")
    parser.add_argument("--max-samples", type=int, default=0, help="테스트용 pose/morpheme 샘플 제한. 0이면 전체")
    args = parser.parse_args()

    reporter = Reporter()
    aihub_root = Path(args.aihub_root)
    pose_dir = Path(args.pose_dir)
    out_data_dir = Path(args.out_data_dir)
    signers = parse_signers(args.signers)
    views = parse_csv(args.views)

    reporter.info("=" * 80)
    reporter.info("AIHub KSL label 생성 시작")
    reporter.info("=" * 80)
    reporter.info(f"- aihub_root: {aihub_root}")
    reporter.info(f"- pose_dir: {pose_dir}")
    reporter.info(f"- out_data_dir: {out_data_dir}")
    reporter.info(f"- signers: {signers}")
    reporter.info(f"- views: {views}")
    reporter.info(f"- max_samples: {args.max_samples}")

    if not aihub_root.is_dir():
        raise SystemExit(f"--aihub_root가 디렉터리가 아닙니다: {aihub_root}")
    if not pose_dir.is_dir():
        raise SystemExit(f"--pose_dir가 디렉터리가 아닙니다: {pose_dir}")

    morpheme_map = collect_morpheme_map(
        aihub_root, signers, views, reporter, max_samples=args.max_samples
    )
    pose_samples = collect_pose_samples(pose_dir, signers, views, max_samples=args.max_samples)
    reporter.info(f"- pose pkl 후보 수: {len(pose_samples)}")

    label_samples, unmatched = make_label_samples(pose_samples, morpheme_map)
    reporter.info(f"- label sample 생성 수: {len(label_samples)}")
    reporter.info(f"- morpheme 매칭 실패 수: {len(unmatched)}")

    splits = split_by_sentence(label_samples)
    save_gzip_pickle(out_data_dir / "labels.train", splits["train"])
    save_gzip_pickle(out_data_dir / "labels.dev", splits["dev"])
    save_gzip_pickle(out_data_dir / "labels.test", splits["test"])
    save_unmatched_csv(out_data_dir / "unmatched_morpheme.csv", unmatched)

    reporter.info("")
    reporter.info("=" * 80)
    reporter.info("split 요약")
    reporter.info("=" * 80)
    for split_name in ["train", "dev", "test"]:
        summarize_split(split_name, splits[split_name], reporter)

    reporter.info("")
    reporter.info(f"- 저장 완료: {out_data_dir}")
    reporter.info(f"- unmatched 목록: {out_data_dir / 'unmatched_morpheme.csv'}")


if __name__ == "__main__":
    main()


# 실행 예시:
# python scripts/build_aihub_ksl_labels.py --aihub_root ".\Uni-Sign-main\korean" --pose_dir ".\Uni-Sign-main\dataset\AIHub_KSL\pose_format" --out_data_dir ".\Uni-Sign-main\data\AIHub_KSL" --signers 01,02 --views D,U,R,L,F
