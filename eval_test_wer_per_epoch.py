"""
stgcn_v2 모델의 각 에폭 체크포인트를 test 셋으로 평가해
epoch별 test WER을 test_wer_per_epoch.json에 저장한다.

실행:
    python eval_test_wer_per_epoch.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import utils
from config import test_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign
from SLRT_metrics import wer_list


OUT_DIR   = Path("out/aihub_ksl_cslr_pose_only_stgcn_v2")
RESULT_FILE = OUT_DIR / "test_wer_per_epoch.json"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",               default="AIHub_KSL")
    p.add_argument("--task",                  default="CSLR")
    p.add_argument("--batch_size",            default=4, type=int)
    p.add_argument("--num_workers",           default=4, type=int)
    p.add_argument("--hidden_dim",            default=256, type=int)
    p.add_argument("--mt5_init",              default="tiny_random")
    p.add_argument("--mt5_d_model",           default=256, type=int)
    p.add_argument("--mt5_d_kv",              default=64,  type=int)
    p.add_argument("--mt5_d_ff",              default=512, type=int)
    p.add_argument("--mt5_num_layers",        default=2,   type=int)
    p.add_argument("--mt5_num_decoder_layers",default=2,   type=int)
    p.add_argument("--mt5_num_heads",         default=4,   type=int)
    p.add_argument("--mt5_dropout",           default=0.1, type=float)
    p.add_argument("--label_smoothing",       default=0.0, type=float)
    p.add_argument("--max_length",            default=256, type=int)
    p.add_argument("--max_new_tokens",        default=100, type=int)
    p.add_argument("--num_beams",             default=4,   type=int)
    p.add_argument("--rgb_support",           action="store_true")
    p.add_argument("--pin_mem",               action="store_true")
    p.add_argument("--seed",                  default=0,   type=int)
    p.add_argument("--distributed",           default=False)
    p.add_argument("--stgcn_pretrained",      default="")
    p.add_argument("--freeze_stgcn",          action="store_true")
    p.add_argument("--finetune",              default="")
    p.add_argument("--skip_eval",             action="store_true")
    p.add_argument("--eval",                  action="store_true")
    return p.parse_args([])


@torch.no_grad()
def eval_checkpoint(ckpt_path, model, data_loader, tokenizer):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    preds, refs = [], []
    for src_input, tgt_input in data_loader:
        for k in src_input:
            if isinstance(src_input[k], torch.Tensor):
                src_input[k] = src_input[k].cuda(non_blocking=True)
        tgt_input["gt_sentence"] = tgt_input["gt_gloss"]

        stack_out = model(src_input, tgt_input)
        out_ids   = model.generate(stack_out,
                                   max_new_tokens=100,
                                   num_beams=4)
        batch_preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        preds.extend(batch_preds)
        refs.extend(tgt_input["gt_gloss"])

    wer_result = wer_list(hypotheses=preds, references=refs)
    return wer_result["wer"]


def main():
    args = get_args()
    utils.set_seed(args.seed)

    print("Loading test dataset...")
    test_data = S2T_Dataset(path=test_label_paths[args.dataset],
                            args=args, phase="test")
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=test_data.collate_fn,
                             shuffle=False,
                             pin_memory=args.pin_mem)
    print(f"  {len(test_data)} samples")

    print("Building model...")
    model = Uni_Sign(args=args).cuda()
    tokenizer = model.mt5_tokenizer

    ckpt_files = sorted(OUT_DIR.glob("checkpoint_*.pth"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not ckpt_files:
        print("체크포인트 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    results = {}
    # resume from existing results if interrupted
    if RESULT_FILE.exists():
        results = json.loads(RESULT_FILE.read_text())
        print(f"기존 결과 {len(results)}개 불러옴 ({RESULT_FILE})")

    for ckpt in ckpt_files:
        epoch = int(ckpt.stem.split("_")[1])
        if str(epoch) in results:
            print(f"  Epoch {epoch:2d}: 이미 평가됨 (WER={results[str(epoch)]:.2f}%), 스킵")
            continue

        print(f"  Epoch {epoch:2d}: {ckpt.name} 평가 중...", end=" ", flush=True)
        wer = eval_checkpoint(ckpt, model, test_loader, tokenizer)
        results[str(epoch)] = float(wer)
        print(f"Test WER = {wer:.2f}%")

        RESULT_FILE.write_text(json.dumps(results, indent=2))

    print(f"\n완료. 결과 저장: {RESULT_FILE}")


if __name__ == "__main__":
    main()
