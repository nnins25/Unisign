"""
AIHub_KSL train/dev/test split을 재구성한다.

문제: 기존 dev / test 가 train 과 도메인이 달라 평가가 부정확함
   (dev = 길 안내, test = 분실물, train = 길 안내 중심).

해결: 세 split 을 다 합친 뒤 "문장 그룹" (같은 SEN id 의 _D/_F/_L/_R/_U 다섯 각도)
   단위로 무작위 8:1:1 재분할. 같은 문장의 다른 각도가 train/dev/test 에
   동시에 들어가는 leak 을 막는다.

기존 라벨 파일은 *.original 로 백업한다.
"""
import argparse
import gzip
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

LABEL_DIR = Path("./data/AIHub_KSL")
SPLITS = ["train", "dev", "test"]

# signer01_NIA_SL_SEN1601_D  ->  group key = signer01_NIA_SL_SEN1601
GROUP_RE = re.compile(r"^(.*?_SEN\d+)_[A-Z]+$")


def load_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_gz(path, obj):
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def group_key(name):
    m = GROUP_RE.match(name)
    return m.group(1) if m else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio_train", type=float, default=0.8)
    parser.add_argument("--ratio_dev", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print stats without writing new files.")
    args = parser.parse_args()

    assert abs(args.ratio_train + args.ratio_dev - 0.9) < 1e-6 or \
        args.ratio_train + args.ratio_dev < 1.0, "ratios must leave room for test"

    # 1) load all
    all_samples = {}
    counts = {}
    for split in SPLITS:
        path = LABEL_DIR / f"labels.{split}"
        d = load_gz(path)
        counts[split] = len(d)
        for k, v in d.items():
            if k in all_samples:
                print(f"  [WARN] duplicate key across splits: {k}")
            all_samples[k] = v
    print(f"Loaded: train={counts['train']}, dev={counts['dev']}, test={counts['test']}, total={len(all_samples)}")

    # 2) group by sentence id
    groups = defaultdict(list)
    for key in all_samples:
        groups[group_key(key)].append(key)
    print(f"Total sentence groups: {len(groups)}")

    sizes = [len(v) for v in groups.values()]
    print(f"  Group size: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")

    # 3) shuffle groups
    group_keys = sorted(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(group_keys)

    # 4) split groups by ratio
    n_total = len(group_keys)
    n_train = int(n_total * args.ratio_train)
    n_dev = int(n_total * args.ratio_dev)
    train_groups = group_keys[:n_train]
    dev_groups = group_keys[n_train:n_train + n_dev]
    test_groups = group_keys[n_train + n_dev:]
    print(f"Group split: train={len(train_groups)}, dev={len(dev_groups)}, test={len(test_groups)}")

    # 5) expand groups to samples
    new_splits = {
        "train": {k: all_samples[k] for g in train_groups for k in groups[g]},
        "dev":   {k: all_samples[k] for g in dev_groups   for k in groups[g]},
        "test":  {k: all_samples[k] for g in test_groups  for k in groups[g]},
    }
    print(f"Sample split: train={len(new_splits['train'])}, dev={len(new_splits['dev'])}, test={len(new_splits['test'])}")

    # 6) sanity check: no group leak across splits
    g2split = {}
    for s, d in new_splits.items():
        for k in d:
            gk = group_key(k)
            if gk in g2split and g2split[gk] != s:
                raise RuntimeError(f"LEAK: group {gk} in both {g2split[gk]} and {s}")
            g2split[gk] = s
    print("  Leak check: OK (no group appears in multiple splits)")

    if args.dry_run:
        print("\n[DRY RUN] No files written. Run again without --dry_run to apply.")
        return

    # 7) backup originals + write new
    for split in SPLITS:
        src = LABEL_DIR / f"labels.{split}"
        backup = LABEL_DIR / f"labels.{split}.original"
        if not backup.exists():
            src.rename(backup)
            print(f"  Backed up: {src.name} -> {backup.name}")
        else:
            # backup already exists -> originals are preserved; just overwrite current
            if src.exists():
                src.unlink()
            print(f"  Backup already exists: {backup.name} (kept)")

    for split, data in new_splits.items():
        out = LABEL_DIR / f"labels.{split}"
        save_gz(out, data)
        print(f"  Wrote: {out.name} ({len(data)} samples)")

    print("\nDone. Re-run diagnose_split.py to verify the new distribution.")


if __name__ == "__main__":
    main()
