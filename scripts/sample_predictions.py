"""
평가 결과 (prediction vs ground-truth) 를 무작위로 N개 골라서 비교 출력.

사용법:
    python scripts/sample_predictions.py --phase test --n 15
    python scripts/sample_predictions.py --phase dev --n 20
"""
import argparse
import random
from pathlib import Path


def load_pairs(out_dir: Path, phase: str):
    pres_path = out_dir / f"{phase}_tmp_pres.txt"
    refs_path = out_dir / f"{phase}_tmp_refs.txt"

    preds, refs = {}, {}
    with open(pres_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "prediction: " not in line:
                continue
            sid = line.split(",", 1)[0].replace("sample: ", "").strip()
            preds[sid] = line.split("prediction: ", 1)[1]

    with open(refs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "ground-truth: " not in line:
                continue
            sid = line.split(",", 1)[0].replace("sample: ", "").strip()
            refs[sid] = line.split("ground-truth: ", 1)[1]

    return preds, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out/aihub_ksl_cslr_pose_only_stgcn_pretrained")
    parser.add_argument("--phase", default="test", choices=["dev", "test"])
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    preds, refs = load_pairs(out_dir, args.phase)

    common = sorted(set(preds) & set(refs))
    print(f"Total {args.phase} samples: {len(common)}")
    print()

    random.seed(args.seed)
    sampled = random.sample(common, min(args.n, len(common)))

    n_correct = 0
    for sid in sampled:
        gt = refs[sid]
        pred = preds[sid]
        ok = gt.strip() == pred.strip()
        if ok:
            n_correct += 1
        mark = "[OK]" if ok else "[XX]"
        print(f"{mark} {sid}")
        print(f"     GT  : {gt}")
        print(f"     Pred: {pred}")
        print()

    print(f"Exact match: {n_correct}/{len(sampled)}")


if __name__ == "__main__":
    main()
