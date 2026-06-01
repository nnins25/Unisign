"""
AIHub_KSL train/dev/test split의 단어 분포를 비교한다.

체크 항목:
1) 각 split sample 수, 평균 문장 길이
2) 각 split의 vocabulary (고유 단어 집합) 크기
3) test 단어 중 train 에 없는 비율 (OOD vocab rate)
4) test 문장 중 모든 단어가 train 에 등장하는 비율
5) dev vs test 차이
"""
import gzip
import pickle
from collections import Counter
from pathlib import Path

LABEL_PATHS = {
    "train": "./data/AIHub_KSL/labels.train",
    "dev":   "./data/AIHub_KSL/labels.dev",
    "test":  "./data/AIHub_KSL/labels.test",
}


def load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_text(sample):
    """gloss 우선 (CSLR target). 없으면 text."""
    if "gloss" in sample and sample["gloss"]:
        if isinstance(sample["gloss"], list):
            return " ".join(sample["gloss"])
        return sample["gloss"]
    return sample.get("text", "")


def tokenize(s):
    return s.strip().split()


def main():
    data = {k: load(p) for k, p in LABEL_PATHS.items()}

    vocabs = {}
    token_lists = {}
    sentences = {}
    for name, d in data.items():
        toks_all = []
        sents = []
        for key, sample in d.items():
            txt = get_text(sample)
            toks = tokenize(txt)
            toks_all.extend(toks)
            sents.append(toks)
        vocabs[name] = set(toks_all)
        token_lists[name] = toks_all
        sentences[name] = sents

    print("=" * 60)
    print("1) Split sizes")
    print("=" * 60)
    for name, d in data.items():
        sents = sentences[name]
        avg_len = sum(len(s) for s in sents) / max(len(sents), 1)
        print(f"  {name:6s}: {len(d):>6d} samples, avg sentence length = {avg_len:.2f} tokens")

    print()
    print("=" * 60)
    print("2) Vocabulary sizes (unique tokens)")
    print("=" * 60)
    for name, v in vocabs.items():
        print(f"  {name:6s}: {len(v):>5d} unique tokens, {len(token_lists[name]):>6d} total tokens")

    print()
    print("=" * 60)
    print("3) OOD vocab rate (test/dev token NOT in train)")
    print("=" * 60)
    train_vocab = vocabs["train"]
    for name in ["dev", "test"]:
        v = vocabs[name]
        ood_vocab = v - train_vocab
        print(f"  [{name}] OOD unique tokens: {len(ood_vocab)}/{len(v)}  ({100*len(ood_vocab)/max(len(v),1):.1f}%)")
        if ood_vocab and len(ood_vocab) <= 30:
            print(f"           OOD words: {sorted(ood_vocab)}")
        elif ood_vocab:
            print(f"           OOD words (first 30): {sorted(ood_vocab)[:30]}")

    print()
    print("=" * 60)
    print("4) Fully in-vocab sentence rate (all tokens are in train)")
    print("=" * 60)
    for name in ["dev", "test"]:
        sents = sentences[name]
        n_full = sum(1 for s in sents if all(t in train_vocab for t in s))
        print(f"  [{name}] {n_full}/{len(sents)} sentences fully covered  ({100*n_full/max(len(sents),1):.1f}%)")

    print()
    print("=" * 60)
    print("5) dev vs test overlap")
    print("=" * 60)
    overlap = vocabs["dev"] & vocabs["test"]
    only_dev = vocabs["dev"] - vocabs["test"]
    only_test = vocabs["test"] - vocabs["dev"]
    print(f"  shared tokens: {len(overlap)}")
    print(f"  only in dev:   {len(only_dev)}  {'-> ' + str(sorted(only_dev)[:20]) if only_dev else ''}")
    print(f"  only in test:  {len(only_test)}  {'-> ' + str(sorted(only_test)[:20]) if only_test else ''}")

    print()
    print("=" * 60)
    print("6) Top-10 most frequent tokens per split")
    print("=" * 60)
    for name in ["train", "dev", "test"]:
        c = Counter(token_lists[name]).most_common(10)
        print(f"  {name:6s}: {c}")


if __name__ == "__main__":
    main()
