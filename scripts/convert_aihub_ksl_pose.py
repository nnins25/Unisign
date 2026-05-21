#!/usr/bin/env python
"""
AIHub 한국어 수어 keypoint JSON을 Uni-Sign pose_format pkl로 변환하기 위한 안전한 변환기.

중요:
- 이 스크립트는 RGB 영상을 읽지 않습니다.
- Uni-Sign datasets.py의 load_part_kp 인덱스 구조에 맞는 133 keypoint 구조가
  확인된 경우에만 pkl을 생성합니다.
- AIHub/OpenPose 137 구조(25 body + 70 face + 21 left + 21 right)는 현재 코드 근거만으로
  어떤 body 2개와 face 2개를 제외해야 하는지 확정할 수 없으므로 --strict 모드에서 중단합니다.
"""

import argparse
import json
import pickle
from glob import glob
from pathlib import Path

import numpy as np


POSITIONS = ["D", "U", "R", "L", "F"]

# Uni-Sign datasets.py load_part_kp 기준 인덱스 구조:
# - 전체 skeleton 길이: 133
# - body 관련 범위: 0..22 로 추정. 실제 사용 index는 [0] + [3..10]
# - face_all 관련 범위: 23..90. 실제 사용 index는 23..39 중 2칸 간격, 83..90, 53
# - left hand: 91..111
# - right hand: 112..132
UNISIGN_TOTAL_POINTS = 133
UNISIGN_BODY_POINTS = 23
UNISIGN_FACE_POINTS = 68
UNISIGN_LEFT_HAND_POINTS = 21
UNISIGN_RIGHT_HAND_POINTS = 21

# AIHub 샘플 inspect 결과 기준:
# - pose_keypoints_2d: 25 points
# - face_keypoints_2d: 70 points
# - hand_left_keypoints_2d: 21 points
# - hand_right_keypoints_2d: 21 points
#
# mapping 후보(아직 확정 아님):
# - AIHub/OpenPose 137 = body25 + face70 + left21 + right21
# - Uni-Sign 133 = body23 + face68 + left21 + right21
# - demo/pose_extraction.py는 RTMLib Wholebody(to_openpose=False)를 기본값으로 사용한다.
# - RTMLib의 coco133.py는 body 23 + face 68 + left hand 21 + right hand 21 순서를 정의한다.
# - RTMLib의 wholebody.py 주석은 OpenPose face 70과 RTMPose face 68 차이를 언급한다.
#
# 그러나 Uni-Sign 레포 안에서 AIHub/OpenPose 137 -> COCO WholeBody 133 전처리 코드는
# 발견되지 않았다. 그러므로 아래 candidate mapping은 레포/논문 근거로 확정된 mapping이
# 아니라 일반적인 OpenPose BODY_25와 COCO WholeBody 133 이름 대응에 기반한 후보이다.
# --strict 모드에서는 이 candidate mapping을 사용하지 않는다.
AIHUB_OPENPOSE_137_SIGNATURE = {
    "pose_keypoints_2d": 25,
    "face_keypoints_2d": 70,
    "hand_left_keypoints_2d": 21,
    "hand_right_keypoints_2d": 21,
}

UNISIGN_133_SIGNATURE = {
    "pose_keypoints_2d": 23,
    "face_keypoints_2d": 68,
    "hand_left_keypoints_2d": 21,
    "hand_right_keypoints_2d": 21,
}

# 후보 mapping: OpenPose BODY_25 -> COCO WholeBody 23 순서.
# OpenPose BODY_25 일반 순서:
# 0 nose, 1 neck, 2 RShoulder, 3 RElbow, 4 RWrist, 5 LShoulder, 6 LElbow,
# 7 LWrist, 8 MidHip, 9 RHip, 10 RKnee, 11 RAnkle, 12 LHip, 13 LKnee,
# 14 LAnkle, 15 REye, 16 LEye, 17 REar, 18 LEar, 19 LBigToe, 20 LSmallToe,
# 21 LHeel, 22 RBigToe, 23 RSmallToe, 24 RHeel
#
# COCO WholeBody 133 body 순서(coco133.py):
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle,
# 17 left_big_toe, 18 left_small_toe, 19 left_heel,
# 20 right_big_toe, 21 right_small_toe, 22 right_heel
CANDIDATE_BODY_KEEP_INDICES = [
    0, 16, 15, 18, 17,
    5, 2, 6, 3, 7, 4,
    12, 9, 13, 10, 14, 11,
    19, 20, 21, 22, 23, 24,
]
CANDIDATE_BODY_DROP_INDICES = [1, 8]  # neck, mid_hip

# 후보 mapping: OpenPose face70 -> COCO/RTMPose face68.
# RTMLib wholebody.py에는 OpenPose face가 총 70점이고 RTMPose는 68점이라
# 마지막 2점을 padding한다고 설명되어 있다. 따라서 후보로는 face 0..67만 유지한다.
CANDIDATE_FACE_KEEP_INDICES = list(range(68))
CANDIDATE_FACE_DROP_INDICES = [68, 69]

CUSTOM_BODY_KEEP_INDICES = None
CUSTOM_FACE_KEEP_INDICES = None

KEYPOINT_KEYS = [
    "pose_keypoints_2d",
    "face_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
]


class ConversionError(Exception):
    """샘플 변환 실패를 명확히 기록하기 위한 예외."""


class Reporter:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def info(self, text):
        print(text)

    def warn(self, text):
        message = f"WARNING: {text}"
        print(message)
        self.warnings.append(message)

    def fail(self, text):
        print(f"FAILED: {text}")
        self.failures.append(text)


def parse_index_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def keep_from_drop(total_count, drop_indices):
    drop_set = set(drop_indices or [])
    return [idx for idx in range(total_count) if idx not in drop_set]


def validate_mapping_indices(name, indices, source_count, target_count):
    if indices is None:
        raise ConversionError(f"{name} keep index가 없습니다.")
    if len(indices) != target_count:
        raise ConversionError(f"{name} keep index 개수가 {target_count}개가 아닙니다: {len(indices)}")
    bad = [idx for idx in indices if idx < 0 or idx >= source_count]
    if bad:
        raise ConversionError(f"{name} index 범위 오류: {bad}")
    if len(set(indices)) != len(indices):
        raise ConversionError(f"{name} index에 중복이 있습니다: {indices}")


def build_mapping(args, reporter):
    """
    mapping 설정을 만든다.

    반환값:
    - None: mapping 없음. 137 구조는 변환 중단.
    - dict: body_keep_indices, face_keep_indices, proven_by_repo 포함.
    """
    if args.mapping == "none":
        return None

    if args.mapping == "openpose137_to_unisign133_candidate":
        if args.strict:
            raise ConversionError(
                "--strict 모드에서는 레포 근거가 없는 candidate mapping을 사용할 수 없습니다. "
                "dry-run을 하려면 --strict를 빼거나, 검증된 custom mapping을 지정하세요."
            )
        reporter.warn(
            "openpose137_to_unisign133_candidate는 후보 mapping이며 논문/Uni-Sign 전처리 코드 "
            "근거로 확정된 mapping이 아닙니다."
        )
        reporter.warn(f"후보 body drop indices: {CANDIDATE_BODY_DROP_INDICES}")
        reporter.warn(f"후보 face drop indices: {CANDIDATE_FACE_DROP_INDICES}")
        return {
            "name": args.mapping,
            "body_keep_indices": CANDIDATE_BODY_KEEP_INDICES,
            "face_keep_indices": CANDIDATE_FACE_KEEP_INDICES,
            "proven_by_repo": False,
        }

    if args.mapping == "custom":
        body_keep = None
        face_keep = None

        if args.custom_mapping_json:
            mapping_path = Path(args.custom_mapping_json)
            custom = safe_load_json(mapping_path)
            body_keep = custom.get("body_keep_indices")
            face_keep = custom.get("face_keep_indices")
            body_drop = custom.get("body_drop_indices")
            face_drop = custom.get("face_drop_indices")
            if body_keep is None and body_drop is not None:
                body_keep = keep_from_drop(25, body_drop)
            if face_keep is None and face_drop is not None:
                face_keep = keep_from_drop(70, face_drop)

        if body_keep is None:
            body_keep = parse_index_list(args.body_keep_indices)
        if face_keep is None:
            face_keep = parse_index_list(args.face_keep_indices)

        body_drop = parse_index_list(args.body_drop_indices)
        face_drop = parse_index_list(args.face_drop_indices)
        if body_keep is None and body_drop is not None:
            body_keep = keep_from_drop(25, body_drop)
            reporter.warn(
                "custom body_drop_indices는 원래 순서를 보존해 keep list를 만듭니다. "
                "Uni-Sign/COCO133 순서와 맞는지 사용자가 반드시 확인해야 합니다."
            )
        if face_keep is None and face_drop is not None:
            face_keep = keep_from_drop(70, face_drop)

        if body_keep is None and CUSTOM_BODY_KEEP_INDICES is not None:
            body_keep = CUSTOM_BODY_KEEP_INDICES
        if face_keep is None and CUSTOM_FACE_KEEP_INDICES is not None:
            face_keep = CUSTOM_FACE_KEEP_INDICES

        validate_mapping_indices("custom body", body_keep, 25, 23)
        validate_mapping_indices("custom face", face_keep, 70, 68)
        reporter.warn("custom mapping을 사용합니다. 이 mapping의 과학적 근거는 사용자 책임으로 확인해야 합니다.")
        return {
            "name": args.mapping,
            "body_keep_indices": body_keep,
            "face_keep_indices": face_keep,
            "proven_by_repo": False,
        }

    raise ConversionError(f"알 수 없는 mapping 모드: {args.mapping}")


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_signers(value):
    """수어자 번호는 항상 '01', '02' 같은 문자열로 맞춘다."""
    signers = []
    for item in parse_csv(value):
        if item.isdigit():
            signers.append(item.zfill(2))
        else:
            signers.append(item)
    return signers


def safe_load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)


def get_people0(obj):
    if not isinstance(obj, dict):
        raise ConversionError("JSON 최상위가 dict가 아닙니다.")

    if "people" not in obj:
        raise ConversionError("JSON에 people key가 없습니다.")

    people = obj["people"]
    if isinstance(people, list):
        if not people:
            raise ConversionError("people list가 비어 있습니다.")
        return people[0]
    if isinstance(people, dict):
        return people

    raise ConversionError(f"people 타입이 예상과 다릅니다: {type(people).__name__}")


def split_flat_keypoints(values, key, reporter):
    if not isinstance(values, list):
        raise ConversionError(f"{key} 값이 list가 아닙니다.")

    if len(values) == 0:
        raise ConversionError(f"{key} 값이 비어 있습니다.")

    if len(values) % 3 == 0:
        coords = []
        scores = []
        for idx in range(0, len(values), 3):
            coords.append([float(values[idx]), float(values[idx + 1])])
            scores.append(float(values[idx + 2]))
        return coords, scores

    if len(values) % 2 == 0:
        reporter.warn(f"{key}: confidence 없는 x/y 구조로 추정되어 score를 1.0으로 채웁니다.")
        coords = []
        scores = []
        for idx in range(0, len(values), 2):
            coords.append([float(values[idx]), float(values[idx + 1])])
            scores.append(1.0)
        return coords, scores

    raise ConversionError(f"{key} 길이가 3 또는 2의 배수가 아닙니다: length={len(values)}")


def summarize_counts(point_counts):
    total = sum(point_counts.values())
    return (
        f"body{point_counts.get('pose_keypoints_2d', 0)}, "
        f"face{point_counts.get('face_keypoints_2d', 0)}, "
        f"left{point_counts.get('hand_left_keypoints_2d', 0)}, "
        f"right{point_counts.get('hand_right_keypoints_2d', 0)}, "
        f"total{total}"
    )


def apply_137_to_133_mapping(parts, scores, mapping):
    validate_mapping_indices("body", mapping["body_keep_indices"], 25, 23)
    validate_mapping_indices("face", mapping["face_keep_indices"], 70, 68)

    body_points = [parts["pose_keypoints_2d"][idx] for idx in mapping["body_keep_indices"]]
    body_scores = [scores["pose_keypoints_2d"][idx] for idx in mapping["body_keep_indices"]]
    face_points = [parts["face_keypoints_2d"][idx] for idx in mapping["face_keep_indices"]]
    face_scores = [scores["face_keypoints_2d"][idx] for idx in mapping["face_keep_indices"]]

    ordered_points = (
        body_points
        + face_points
        + parts["hand_left_keypoints_2d"]
        + parts["hand_right_keypoints_2d"]
    )
    ordered_scores = (
        body_scores
        + face_scores
        + scores["hand_left_keypoints_2d"]
        + scores["hand_right_keypoints_2d"]
    )
    return frame_to_numpy(ordered_points, ordered_scores)


def frame_to_numpy(points, confs):
    """datasets.py의 numpy식 다차원 인덱싱을 위해 frame 원소를 ndarray로 저장한다."""
    keypoints = np.asarray([points], dtype=np.float32)
    scores = np.asarray([confs], dtype=np.float32)
    if keypoints.shape != (1, UNISIGN_TOTAL_POINTS, 2):
        raise ConversionError(f"keypoints frame shape 오류: {keypoints.shape}")
    if scores.shape != (1, UNISIGN_TOTAL_POINTS):
        raise ConversionError(f"scores frame shape 오류: {scores.shape}")
    return keypoints, scores


def frame_to_unisign_133(frame_path, strict, reporter, mapping=None, dry_run=False):
    obj = safe_load_json(frame_path)
    person = get_people0(obj)

    missing = [key for key in KEYPOINT_KEYS if key not in person]
    if missing:
        raise ConversionError(f"필수 key 없음: {missing}")

    parts = {}
    scores = {}
    point_counts = {}
    for key in KEYPOINT_KEYS:
        coords, confs = split_flat_keypoints(person[key], key, reporter)
        parts[key] = coords
        scores[key] = confs
        point_counts[key] = len(coords)

    if dry_run:
        reporter.info(f"before: {summarize_counts(point_counts)}")

    if point_counts == UNISIGN_133_SIGNATURE:
        ordered_points = (
            parts["pose_keypoints_2d"]
            + parts["face_keypoints_2d"]
            + parts["hand_left_keypoints_2d"]
            + parts["hand_right_keypoints_2d"]
        )
        ordered_scores = (
            scores["pose_keypoints_2d"]
            + scores["face_keypoints_2d"]
            + scores["hand_left_keypoints_2d"]
            + scores["hand_right_keypoints_2d"]
        )
        if dry_run:
            reporter.info("after: body23, face68, left21, right21, total133")
        return frame_to_numpy(ordered_points, ordered_scores)

    if point_counts == AIHUB_OPENPOSE_137_SIGNATURE:
        if mapping is not None:
            frame_keypoints, frame_scores = apply_137_to_133_mapping(parts, scores, mapping)
            if dry_run:
                reporter.info("after: body23, face68, left21, right21, total133")
                reporter.info(f"mapping mode: {mapping['name']}")
                reporter.info(f"body_keep_indices: {mapping['body_keep_indices']}")
                reporter.info(f"face_keep_indices: {mapping['face_keep_indices']}")
            return frame_keypoints, frame_scores

        message = (
            "AIHub/OpenPose 137 구조가 감지되었습니다 "
            "(body25 + face70 + left21 + right21). "
            "Uni-Sign은 133 구조(body23 + face68 + left21 + right21)를 기대하지만, "
            "datasets.py 근거만으로 제외할 body 2개와 face 2개 index를 확정할 수 없습니다. "
            "임의로 앞 133개만 사용하지 않으므로 변환을 중단합니다."
        )
        if strict:
            raise ConversionError(message)
        raise ConversionError("비 strict 모드에서도 미확정 137->133 mapping은 변환하지 않습니다. " + message)

    raise ConversionError(
        "예상과 다른 keypoint 개수입니다: "
        + ", ".join([f"{key}={value}" for key, value in point_counts.items()])
    )


def unique_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def looks_like_keypoint_signer_root(path, signer):
    """경로에 keypoint가 포함되고, 폴더명이 signer 번호인 후보만 사용한다."""
    if not path.is_dir():
        return False
    if path.name != signer:
        return False
    return "keypoint" in str(path).lower()


def find_signer_root_candidates(aihub_root, signer):
    """
    AIHub 폴더명이 [라벨]01_real_sen_keypoint처럼 고정되어 있다고 가정하지 않고
    여러 패턴으로 수어자 keypoint root 후보를 찾는다.
    """
    candidates = []

    # 요청된 주요 패턴:
    # aihub_root/keypoint/*01*keypoint*/01
    # aihub_root/keypoint/*02*keypoint*/02
    patterns = [
        aihub_root / "keypoint" / f"*{signer}*keypoint*" / signer,
        aihub_root / "keypoint" / f"*{signer}*real_sen_keypoint*" / signer,
        aihub_root / "keypoint" / f"*{signer}*" / signer,
    ]
    for pattern in patterns:
        for raw_path in glob(str(pattern)):
            path = Path(raw_path)
            if looks_like_keypoint_signer_root(path, signer):
                candidates.append(path)

    # aihub_root 자체가 이미 .../keypoint/[라벨]01.../01 인 경우도 받아준다.
    if looks_like_keypoint_signer_root(aihub_root, signer):
        candidates.append(aihub_root)

    # aihub_root 아래에서 signer 번호 폴더를 재귀적으로 찾되, keypoint 경로만 후보로 사용한다.
    for raw_path in glob(str(aihub_root / "**" / signer), recursive=True):
        path = Path(raw_path)
        if looks_like_keypoint_signer_root(path, signer):
            candidates.append(path)

    return unique_paths(candidates)


def select_signer_root(aihub_root, signer, reporter):
    candidates = find_signer_root_candidates(aihub_root, signer)
    reporter.info(f"- 수어자 {signer} keypoint root 후보 수: {len(candidates)}")
    for candidate in candidates:
        reporter.info(f"  - 후보: {candidate}")

    if not candidates:
        return None

    selected = candidates[0]
    reporter.info(f"  - 사용할 경로: {selected}")
    return selected


def extract_sentence_id_and_view(sentence_dir):
    tokens = sentence_dir.name.split("_")
    view = tokens[-1] if tokens and tokens[-1] in POSITIONS else None

    sentence_id = None
    for token in tokens:
        if token.startswith("SEN"):
            sentence_id = token
            break

    if sentence_id is None and len(tokens) >= 3:
        # 추측: NIA_SL_SEN0001 같은 prefix를 문장 id 후보로 사용한다.
        sentence_id = "_".join(tokens[:3])

    return sentence_id, view


def output_name(signer, sentence_id, view):
    safe_sentence = sentence_id
    if sentence_id.startswith("NIA_SL_"):
        safe_sentence = sentence_id
    elif sentence_id.startswith("SEN"):
        safe_sentence = f"NIA_SL_{sentence_id}"
    return f"signer{signer}_{safe_sentence}_{view}.pkl"


def list_sentence_dirs(aihub_root, signers, views, reporter):
    sentence_dirs = []
    counts = {
        signer: {view: 0 for view in views}
        for signer in signers
    }

    for signer in signers:
        signer_root = select_signer_root(aihub_root, signer, reporter)
        if signer_root is None:
            reporter.fail(f"수어자 {signer} keypoint root를 찾지 못했습니다.")
            continue

        for child in sorted(signer_root.iterdir()):
            if not child.is_dir():
                continue
            # 문장 폴더는 root 바로 아래의 NIA_SL_SEN*_REAL*_{view} 형태만 대상으로 한다.
            if not child.name.startswith("NIA_SL_SEN"):
                continue
            sentence_id, view = extract_sentence_id_and_view(child)
            if view not in views:
                continue
            if sentence_id is None:
                reporter.fail(f"sentence_id 추출 실패: {child} / 후보={child.name.split('_')}")
                continue
            counts[signer][view] += 1
            sentence_dirs.append((signer, sentence_id, view, child))

    reporter.info("")
    reporter.info("- 변환 대상 문장 폴더 수 (signer별/view별)")
    for signer in signers:
        total = sum(counts[signer].values())
        reporter.info(f"  - signer {signer}: {total}")
        for view in views:
            reporter.info(f"    - {view}: {counts[signer][view]}")

    return sentence_dirs


def convert_sentence(signer, sentence_id, view, sentence_dir, out_pose_dir, strict, reporter, mapping=None, dry_run=False):
    frame_files = sorted(sentence_dir.glob("*_keypoints.json"))
    if not frame_files:
        raise ConversionError(f"프레임 keypoint JSON이 없습니다: {sentence_dir}")

    keypoints = []
    scores = []
    for frame_path in frame_files:
        frame_keypoints, frame_scores = frame_to_unisign_133(
            frame_path,
            strict,
            reporter,
            mapping=mapping,
            dry_run=dry_run and len(keypoints) == 0,
        )
        keypoints.append(frame_keypoints)
        scores.append(frame_scores)

    if dry_run:
        return None, len(frame_files), keypoints, scores

    out_pose_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_pose_dir / output_name(signer, sentence_id, view)
    with out_path.open("wb") as f:
        pickle.dump({"keypoints": keypoints, "scores": scores}, f)

    return out_path, len(frame_files), keypoints, scores


def shape_of_nested_frame(frame):
    # frame은 list 기반 (1, 133, 2) 형태를 기대한다.
    if not isinstance(frame, list):
        return f"not-list:{type(frame).__name__}"
    if not frame:
        return "(0,)"
    if isinstance(frame[0], list):
        if not frame[0]:
            return f"({len(frame)}, 0)"
        if isinstance(frame[0][0], list):
            return f"({len(frame)}, {len(frame[0])}, {len(frame[0][0])})"
        return f"({len(frame)}, {len(frame[0])})"
    return f"({len(frame)},)"


def save_failures(path, failures):
    if failures:
        path.write_text("\n".join(failures) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="AIHub KSL keypoint JSON -> Uni-Sign pose pkl 변환")
    parser.add_argument("--aihub_root", required=True, help="AIHub korean root 경로")
    parser.add_argument("--out_pose_dir", required=True, help="저장할 pose_format 폴더")
    parser.add_argument("--signers", default="01,02", help="변환할 수어자 목록. 예: 01 또는 01,02")
    parser.add_argument("--views", default="D,U,R,L,F", help="변환할 view 목록. 예: D,U,R,L,F")
    parser.add_argument("--strict", action="store_true", help="mapping 불확실 시 즉시 중단")
    parser.add_argument("--max-samples", type=int, default=0, help="테스트용 최대 문장 샘플 수. 0이면 전체")
    parser.add_argument(
        "--mapping",
        default="none",
        choices=["none", "openpose137_to_unisign133_candidate", "custom"],
        help="137->133 mapping 모드. 기본값 none은 mapping 없이 중단",
    )
    parser.add_argument("--dry-run", action="store_true", help="pkl 저장 없이 첫 샘플 mapping 전후 shape만 출력")
    parser.add_argument("--custom-mapping-json", default="", help="custom mapping JSON 경로")
    parser.add_argument("--body-keep-indices", default="", help="custom body keep indices, 예: 0,16,15,...")
    parser.add_argument("--face-keep-indices", default="", help="custom face keep indices, 예: 0,1,...,67")
    parser.add_argument("--body-drop-indices", default="", help="custom body drop indices. 순서 재배열은 하지 않음")
    parser.add_argument("--face-drop-indices", default="", help="custom face drop indices. 순서 재배열은 하지 않음")
    args = parser.parse_args()

    reporter = Reporter()
    aihub_root = Path(args.aihub_root)
    out_pose_dir = Path(args.out_pose_dir)
    signers = parse_signers(args.signers)
    views = parse_csv(args.views)

    reporter.info("=" * 80)
    reporter.info("AIHub KSL pose 변환 시작")
    reporter.info("=" * 80)
    reporter.info(f"- aihub_root: {aihub_root}")
    reporter.info(f"- out_pose_dir: {out_pose_dir}")
    reporter.info(f"- signers: {signers}")
    reporter.info(f"- views: {views}")
    reporter.info(f"- strict: {args.strict}")
    reporter.info(f"- max_samples: {args.max_samples}")
    reporter.info(f"- mapping: {args.mapping}")
    reporter.info(f"- dry_run: {args.dry_run}")

    if not aihub_root.is_dir():
        reporter.fail(f"--aihub_root 경로가 존재하지 않거나 디렉터리가 아닙니다: {aihub_root}")
        save_failures(Path("failed_pose_conversion.txt"), reporter.failures)
        return

    try:
        mapping = build_mapping(args, reporter)
    except ConversionError as exc:
        reporter.fail(str(exc))
        save_failures(Path("failed_pose_conversion.txt"), reporter.failures)
        return

    sentence_dirs = list_sentence_dirs(aihub_root, signers, views, reporter)
    if args.max_samples:
        sentence_dirs = sentence_dirs[: args.max_samples]

    reporter.info(f"- 변환 대상 문장 폴더 수: {len(sentence_dirs)}")
    if not sentence_dirs:
        reporter.fail("변환 대상 문장 폴더가 없습니다.")
        save_failures(Path("failed_pose_conversion.txt"), reporter.failures)
        return

    converted = []
    for signer, sentence_id, view, sentence_dir in sentence_dirs:
        try:
            out_path, frames, keypoints, scores = convert_sentence(
                signer,
                sentence_id,
                view,
                sentence_dir,
                out_pose_dir,
                args.strict,
                reporter,
                mapping=mapping,
                dry_run=args.dry_run,
            )
            converted.append((out_path, frames, keypoints, scores))
            if args.dry_run:
                reporter.info(f"DRY-RUN OK: {sentence_dir.name} ({frames} frames, 저장 안 함)")
                break
            reporter.info(f"OK: {out_path} ({frames} frames)")
        except ConversionError as exc:
            reporter.fail(f"{sentence_dir}: {exc}")
            if args.strict:
                reporter.warn("--strict 모드이므로 첫 실패에서 변환을 중단합니다.")
                break
        except Exception as exc:
            reporter.fail(f"{sentence_dir}: 예상하지 못한 오류: {type(exc).__name__}: {exc}")
            if args.strict:
                reporter.warn("--strict 모드이므로 첫 실패에서 변환을 중단합니다.")
                break

    reporter.info("")
    reporter.info("=" * 80)
    reporter.info("변환 후 샘플 shape 검증")
    reporter.info("=" * 80)
    for out_path, frames, keypoints, scores in converted[:3]:
        sample_name = out_path.name if out_path is not None else "dry-run-sample"
        reporter.info(f"sample: {sample_name}")
        reporter.info(f"frames: {frames}")
        reporter.info(f"keypoints[0] shape: {shape_of_nested_frame(keypoints[0])}")
        reporter.info(f"scores[0] shape: {shape_of_nested_frame(scores[0])}")

    reporter.info("")
    reporter.info("=" * 80)
    reporter.info("요약")
    reporter.info("=" * 80)
    success_count = 0 if args.dry_run else len(converted)
    reporter.info(f"- 성공 pkl 수: {success_count}")
    reporter.info(f"- 실패 수: {len(reporter.failures)}")
    reporter.info("- failed_pose_conversion.txt에 실패 목록을 저장합니다.")
    save_failures(Path("failed_pose_conversion.txt"), reporter.failures)


if __name__ == "__main__":
    main()


# 실행 예시:
# python scripts/convert_aihub_ksl_pose.py --aihub_root "C:\Users\spai\Documents\야르2\Uni-Sign-main\Uni-Sign-main\korean" --out_pose_dir "./dataset/AIHub_KSL/pose_format" --signers 01,02 --views D,U,R,L,F --strict --max-samples 3
#
# 후보 mapping dry-run 예시(--strict 사용 불가):
# python scripts/convert_aihub_ksl_pose.py --aihub_root "C:\Users\spai\Documents\야르2\Uni-Sign-main\Uni-Sign-main\korean" --out_pose_dir "./dataset/AIHub_KSL/pose_format" --signers 01 --views D --mapping openpose137_to_unisign133_candidate --dry-run --max-samples 1
#
# custom mapping 예시:
# python scripts/convert_aihub_ksl_pose.py --aihub_root "C:\Users\spai\Documents\야르2\Uni-Sign-main\Uni-Sign-main\korean" --out_pose_dir "./dataset/AIHub_KSL/pose_format" --signers 01 --views D --mapping custom --body-keep-indices "0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11,19,20,21,22,23,24" --face-keep-indices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67" --dry-run --max-samples 1
