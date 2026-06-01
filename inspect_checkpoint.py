"""
csl_stage1_weight.pth 파일 안에 어떤 key들이 들어있는지 확인하는 스크립트.
ST-GCN 부분만 골라서 로딩하려면 먼저 구조를 알아야 함.
"""
import torch
from collections import defaultdict

CKPT_PATH = "./pretrained_weight/csl_stage1_weight.pth"

print(f"Loading: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

print(f"\n[1] Top-level type: {type(ckpt)}")

if isinstance(ckpt, dict):
    print(f"[2] Top-level keys: {list(ckpt.keys())}")

    if "model" in ckpt:
        state = ckpt["model"]
        print("    -> using ckpt['model']")
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
        print("    -> using ckpt['state_dict']")
    else:
        state = ckpt
        print("    -> using ckpt directly (looks like raw state_dict)")
else:
    state = ckpt
    print("[2] Not a dict, using directly")

print(f"\n[3] Total params: {len(state)}")

prefixes = defaultdict(int)
for k in state.keys():
    top = k.split(".")[0]
    prefixes[top] += 1

print("\n[4] Top-level prefixes (param name -> count):")
for p, c in sorted(prefixes.items()):
    print(f"    {p}: {c}")

print("\n[5] ST-GCN related sample keys (first 5 each):")
for target in ["proj_linear", "gcn_modules", "fusion_gcn_modules", "part_para"]:
    matched = [k for k in state.keys() if k.startswith(target) or k.startswith(f"module.{target}")]
    print(f"\n  [{target}] (total {len(matched)}):")
    for k in matched[:5]:
        print(f"    {k}  shape={tuple(state[k].shape) if hasattr(state[k], 'shape') else 'scalar'}")

print("\n[6] Has 'module.' prefix?", any(k.startswith("module.") for k in state.keys()))
