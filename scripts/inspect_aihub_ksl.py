#!/usr/bin/env python
"""
AIHub 한국어 수어 Training 폴더 구조 점검용 스크립트.

이 스크립트는 변환을 수행하지 않습니다.
폴더/파일 구조, keypoint JSON 구조, morpheme JSON 구조, 파일명 기반 매칭 가능성만
샘플 중심으로 점검합니다.
"""

import argparse
import json
from glob import glob
from pathlib import Path


POSITIONS = ["D", "U", "R", "L", "F"]
SPEAKERS = ["01", "02"]
KEYPOINT_KEYS = [
    "pose_keypoints_2d",
    "face_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
]


class Reporter:
    """콘솔 출력과 inspect_result.txt 저장을 동시에 하기 위한 간단한 리포터."""

    def __init__(self):
        self.lines = []

    def write(self, text=""):
        print(text)
        self.lines.append(text)

    def section(self, title):
        self.write("")
        self.write("=" * 80)
        self.write(title)
        self.write("=" * 80)

    def save(self, output_path):
        output_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def safe_load_json(path):
    """샘플 JSON 하나만 안전하게 읽는다."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                return json.load(f), None
        except Exception as exc:
            return None, exc
    except Exception as exc:
        return None, exc


def list_top_keys(obj):
    if isinstance(obj, dict):
        return list(obj.keys())
    if isinstance(obj, list):
        return [f"<list length={len(obj)}>"]
    return [f"<{type(obj).__name__}>"]


def find_nested_key(obj, target_key, prefix="$", max_depth=8):
    """JSON 안에서 target_key를 robust하게 찾는다. 첫 번째 발견 위치만 반환한다."""
    if max_depth < 0:
        return None, None

    if isinstance(obj, dict):
        if target_key in obj:
            return f"{prefix}.{target_key}", obj[target_key]
        for key, value in obj.items():
            found_path, found_value = find_nested_key(
                value, target_key, f"{prefix}.{key}", max_depth - 1
            )
            if found_path is not None:
                return found_path, found_value

    if isinstance(obj, list):
        for idx, value in enumerate(obj[:5]):
            found_path, found_value = find_nested_key(
                value, target_key, f"{prefix}[{idx}]", max_depth - 1
            )
            if found_path is not None:
                return found_path, found_value

    return None, None


def flatten_people_if_needed(obj):
    """OpenPose류 JSON에서 people/person 내부에 keypoint가 있는 경우를 대비한다."""
    if not isinstance(obj, dict):
        return obj

    people = obj.get("people")
    if isinstance(people, list) and people:
        return people[0]
    if isinstance(people, dict):
        return people
    return obj


def is_probably_keypoint_file(path):
    name = path.name.lower()
    return path.suffix.lower() == ".json" and (
        "keypoint" in name or "keypoints" in name or "key_point" in name
    )


def is_probably_morpheme_file(path):
    lower_path = str(path).lower()
    return path.suffix.lower() == ".json" and "morpheme" in lower_path


def direction_from_name(path):
    """파일명에서 D/U/R/L/F 촬영 위치 후보를 추정한다."""
    parts = path.stem.split("_")
    for part in parts:
        if part in POSITIONS:
            return part
    return "UNKNOWN"


def speaker_from_path(path):
    """경로 또는 파일명에서 01/02 수어자 후보를 추정한다."""
    parts = list(path.parts)
    stem_parts = path.stem.split("_")

    for speaker in SPEAKERS:
        if speaker in parts:
            return speaker
        if f"REAL{speaker}" in stem_parts:
            return speaker
        if any(part.endswith(speaker) and "REAL" in part for part in stem_parts):
            return speaker
    return "UNKNOWN"


def normalize_stem_tokens(path):
    """프레임 번호, keypoints, morpheme 같은 suffix를 제거한 토큰 목록을 만든다."""
    tokens = path.stem.split("_")
    cleaned = []
    for token in tokens:
        low = token.lower()
        if low in ["keypoint", "keypoints", "morpheme"]:
            continue
        if token.isdigit() and len(token) >= 6:
            continue
        cleaned.append(token)
    return cleaned


def sentence_id_candidates(path):
    """
    파일명에서 sentence_id 후보를 여러 수준으로 추정한다.
    추측:
    - NIA_SL_SEN0001
    - NIA_SL_SEN0001_REAL01
    - NIA_SL_SEN0001_REAL01_D
    """
    tokens = normalize_stem_tokens(path)
    candidates = []

    if len(tokens) >= 3:
        candidates.append("_".join(tokens[:3]))
    if len(tokens) >= 4:
        candidates.append("_".join(tokens[:4]))
    if len(tokens) >= 5:
        candidates.append("_".join(tokens[:5]))
    if tokens:
        candidates.append("_".join(tokens))

    unique = []
    for item in candidates:
        if item and item not in unique:
            unique.append(item)
    return unique


def sample_paths(paths, limit=5):
    return paths[:limit]


def count_positions(paths):
    counts = {pos: 0 for pos in POSITIONS}
    counts["UNKNOWN"] = 0
    for path in paths:
        counts[direction_from_name(path)] = counts.get(direction_from_name(path), 0) + 1
    return counts


def collect_json_files(root, max_files=0):
    """
    전체 JSON을 로딩하지 않고 파일 경로만 수집한다.

    max_files가 0이면 끝까지 스캔한다. 큰 데이터셋에서 빠른 구조 점검만 하고 싶을 때는
    --max-files로 파일 목록 수집 개수를 제한할 수 있다.
    """
    all_json = []
    keypoint_files = []
    morpheme_files = []

    for raw_path in glob(str(root / "**" / "*.json"), recursive=True):
        path = Path(raw_path)
        all_json.append(path)
        if is_probably_keypoint_file(path):
            keypoint_files.append(path)
        if is_probably_morpheme_file(path):
            morpheme_files.append(path)

        if max_files and len(all_json) >= max_files:
            break

    return all_json, keypoint_files, morpheme_files


def report_speaker_dirs(root, reporter):
    reporter.section("1. 01 / 02 수어자 폴더 확인")
    for speaker in SPEAKERS:
        # 깊은 전체 탐색 대신 JSON 파일 경로 탐색 결과와 별도로 폴더명 후보만 가볍게 찾는다.
        direct_candidates = []
        for raw_path in glob(str(root / "**" / speaker), recursive=True):
            path = Path(raw_path)
            if path.is_dir():
                direct_candidates.append(path)
            if len(direct_candidates) >= 20:
                break
        reporter.write(f"- {speaker} 폴더 후보 수: {len(direct_candidates)}")
        for path in sample_paths(direct_candidates, 5):
            reporter.write(f"  - {path}")


def report_position_counts(keypoint_files, reporter):
    reporter.section("2. D/U/R/L/F 위치별 keypoint JSON 파일 개수")
    for speaker in SPEAKERS:
        speaker_files = [p for p in keypoint_files if speaker_from_path(p) == speaker]
        counts = count_positions(speaker_files)
        reporter.write(f"- 수어자 {speaker}: 총 {len(speaker_files)}개")
        for pos in POSITIONS + ["UNKNOWN"]:
            reporter.write(f"  - {pos}: {counts.get(pos, 0)}")

    unknown_files = [p for p in keypoint_files if speaker_from_path(p) == "UNKNOWN"]
    if unknown_files:
        reporter.write(f"- 수어자 추정 불가 keypoint 파일: {len(unknown_files)}개")
        for path in sample_paths(unknown_files, 5):
            reporter.write(f"  - {path}")


def report_morpheme_candidates(morpheme_files, reporter):
    reporter.section("3. morpheme 라벨 파일 후보")
    reporter.write(f"- morpheme 후보 JSON 수: {len(morpheme_files)}")
    for path in sample_paths(morpheme_files, 20):
        reporter.write(f"  - {path}")


def inspect_keypoint_sample(keypoint_files, reporter):
    reporter.section("4. keypoint JSON 샘플 구조")
    if not keypoint_files:
        reporter.write("- keypoint JSON 후보를 찾지 못했습니다.")
        return None

    sample = keypoint_files[0]
    obj, err = safe_load_json(sample)
    reporter.write(f"- 샘플 파일: {sample}")
    if err:
        reporter.write(f"- JSON 로딩 실패: {err}")
        return None

    reporter.write(f"- 최상위 구조: {list_top_keys(obj)}")
    search_obj = flatten_people_if_needed(obj)
    if search_obj is not obj:
        reporter.write("- 추측: keypoint 배열은 people 내부 첫 번째 객체에서 발견될 수 있습니다.")
        reporter.write(f"- people 내부 구조: {list_top_keys(search_obj)}")

    for key in KEYPOINT_KEYS:
        found_path, value = find_nested_key(search_obj, key)
        if found_path is None:
            found_path, value = find_nested_key(obj, key)

        if found_path is None:
            reporter.write(f"- {key}: 없음")
            continue

        length = len(value) if isinstance(value, list) else "list 아님"
        reporter.write(f"- {key}: 있음 ({found_path})")
        reporter.write(f"  - flat array length = {length}")
        if isinstance(value, list):
            if len(value) % 3 == 0:
                reporter.write("  - 추정: x, y, confidence 3개 단위 구성 가능")
                reporter.write(f"  - 추정 keypoint 개수: {len(value) // 3}")
            else:
                reporter.write("  - 추정: 길이가 3의 배수가 아니므로 3개 단위라고 단정 불가")

    reporter.write(f"- sentence_id 후보: {sentence_id_candidates(sample)}")
    return sample


def inspect_morpheme_sample(morpheme_files, reporter):
    reporter.section("5. morpheme JSON 샘플 구조")
    if not morpheme_files:
        reporter.write("- morpheme JSON 후보를 찾지 못했습니다.")
        return None

    sample = morpheme_files[0]
    obj, err = safe_load_json(sample)
    reporter.write(f"- 샘플 파일: {sample}")
    if err:
        reporter.write(f"- JSON 로딩 실패: {err}")
        return None

    reporter.write(f"- 최상위 구조: {list_top_keys(obj)}")
    if isinstance(obj, dict):
        for key in ["metaData", "data", "annotations", "label", "labels"]:
            if key in obj:
                value = obj[key]
                if isinstance(value, list):
                    reporter.write(f"- {key}: list length={len(value)}")
                    if value:
                        reporter.write(f"  - 첫 원소 구조: {list_top_keys(value[0])}")
                elif isinstance(value, dict):
                    reporter.write(f"- {key}: dict keys={list_top_keys(value)}")
                else:
                    reporter.write(f"- {key}: {type(value).__name__}")

    reporter.write(f"- sentence_id 후보: {sentence_id_candidates(sample)}")
    return sample


def report_matching_samples(keypoint_files, morpheme_files, reporter):
    reporter.section("6. keypoint / morpheme sentence_id 매칭 샘플 점검")
    if not keypoint_files or not morpheme_files:
        reporter.write("- keypoint 또는 morpheme 후보가 부족해 매칭을 점검하지 못했습니다.")
        return

    morpheme_ids = {}
    for path in sample_paths(morpheme_files, 200):
        for sid in sentence_id_candidates(path):
            morpheme_ids.setdefault(sid, []).append(path)

    checked = 0
    matched = 0
    for kp in sample_paths(keypoint_files, 50):
        kp_candidates = sentence_id_candidates(kp)
        hits = []
        for sid in kp_candidates:
            if sid in morpheme_ids:
                hits.append((sid, morpheme_ids[sid][0]))
        checked += 1
        if hits:
            matched += 1
            sid, morph_path = hits[0]
            reporter.write(f"- 매칭 가능: {kp.name} -> {morph_path.name} (id={sid})")
        else:
            reporter.write(f"- 매칭 불확실: {kp.name}, 후보={kp_candidates}")
        if checked >= 10:
            break

    reporter.write(f"- 샘플 매칭 결과: {matched}/{checked}개 매칭 가능")
    reporter.write("- 주의: 파일명 기반 샘플 추정이며, 전체 매칭 검증은 아닙니다.")


def report_position_id_sharing(keypoint_files, reporter):
    reporter.section("7. D/U/R/L/F가 같은 sentence_id를 공유하는지 샘플 점검")
    grouped = {}
    for path in sample_paths(keypoint_files, 1000):
        candidates = sentence_id_candidates(path)
        if not candidates:
            continue
        base_id = candidates[0]
        grouped.setdefault(base_id, set()).add(direction_from_name(path))

    shown = 0
    for sid, positions in grouped.items():
        known_positions = sorted([p for p in positions if p in POSITIONS])
        if known_positions:
            reporter.write(f"- {sid}: {', '.join(known_positions)}")
            shown += 1
        if shown >= 10:
            break

    if shown == 0:
        reporter.write("- 위치 공유 여부를 추정할 수 있는 샘플을 찾지 못했습니다.")
    reporter.write("- 추측: 위 id는 파일명 토큰 앞부분 기준입니다.")


def report_speaker_id_sharing(keypoint_files, reporter):
    reporter.section("8. 01과 02가 같은 sentence_id 목록을 공유하는지 샘플 점검")
    ids_by_speaker = {speaker: set() for speaker in SPEAKERS}
    for path in sample_paths(keypoint_files, 3000):
        speaker = speaker_from_path(path)
        if speaker in ids_by_speaker:
            candidates = sentence_id_candidates(path)
            if candidates:
                ids_by_speaker[speaker].add(candidates[0])

    common = ids_by_speaker["01"] & ids_by_speaker["02"]
    only_01 = ids_by_speaker["01"] - ids_by_speaker["02"]
    only_02 = ids_by_speaker["02"] - ids_by_speaker["01"]

    reporter.write(f"- 01 샘플 sentence_id 수: {len(ids_by_speaker['01'])}")
    reporter.write(f"- 02 샘플 sentence_id 수: {len(ids_by_speaker['02'])}")
    reporter.write(f"- 공통 sentence_id 수: {len(common)}")
    reporter.write(f"- 01에만 있는 샘플 id 예시: {sorted(list(only_01))[:10]}")
    reporter.write(f"- 02에만 있는 샘플 id 예시: {sorted(list(only_02))[:10]}")
    reporter.write(f"- 공통 샘플 id 예시: {sorted(list(common))[:10]}")
    reporter.write("- 주의: 전체가 아니라 앞쪽 파일 샘플 기준입니다.")


def main():
    parser = argparse.ArgumentParser(description="AIHub 한국어 수어 데이터 구조 점검")
    parser.add_argument("--root", required=True, help="AIHub Training 폴더 경로")
    parser.add_argument(
        "--output",
        default="inspect_result.txt",
        help="점검 결과 저장 파일 경로 (기본: inspect_result.txt)",
    )
    parser.add_argument(
        "--max-files",
        default=0,
        type=int,
        help="JSON 파일 목록 수집 상한. 0이면 전체 스캔 (기본: 0)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    output_path = Path(args.output)
    reporter = Reporter()

    reporter.section("AIHub 한국어 수어 데이터 inspect 결과")
    reporter.write(f"- 입력 root: {root}")
    reporter.write(f"- root 존재 여부: {root.exists()}")
    reporter.write(f"- root 디렉터리 여부: {root.is_dir()}")

    if not root.exists() or not root.is_dir():
        reporter.write("- 오류: --root 경로가 존재하지 않거나 디렉터리가 아닙니다.")
        reporter.save(output_path)
        return

    all_json, keypoint_files, morpheme_files = collect_json_files(root, args.max_files)
    reporter.write(f"- 전체 JSON 파일 후보 수: {len(all_json)}")
    reporter.write(f"- keypoint JSON 후보 수: {len(keypoint_files)}")
    reporter.write(f"- morpheme JSON 후보 수: {len(morpheme_files)}")
    if args.max_files:
        reporter.write(f"- 주의: --max-files={args.max_files} 제한으로 일부 파일만 수집했습니다.")

    report_speaker_dirs(root, reporter)
    report_position_counts(keypoint_files, reporter)
    report_morpheme_candidates(morpheme_files, reporter)
    inspect_keypoint_sample(keypoint_files, reporter)
    inspect_morpheme_sample(morpheme_files, reporter)
    report_matching_samples(keypoint_files, morpheme_files, reporter)
    report_position_id_sharing(keypoint_files, reporter)
    report_speaker_id_sharing(keypoint_files, reporter)

    reporter.section("9. 요약")
    reporter.write("- 이 스크립트는 변환을 수행하지 않았습니다.")
    reporter.write("- JSON 본문은 대표 샘플만 읽었습니다.")
    reporter.write("- sentence_id, 촬영 위치, 수어자 매칭은 파일명 기반 추측입니다.")
    reporter.write(f"- 결과 저장: {output_path.resolve()}")
    reporter.save(output_path)


if __name__ == "__main__":
    main()


# 실행 예시:
# python scripts/inspect_aihub_ksl.py --root "D:/AIHub/Training"
# 빠른 샘플 점검 예시:
# python scripts/inspect_aihub_ksl.py --root "D:/AIHub/Training" --max-files 50000
