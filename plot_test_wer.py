"""
eval_test_wer_per_epoch.py 실행 후 생성된 test_wer_per_epoch.json을 읽어
에폭별 Test WER 그래프를 그린다.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

result_path = Path("out/aihub_ksl_cslr_pose_only_stgcn_v2/test_wer_per_epoch.json")
log_path    = Path("out/aihub_ksl_cslr_pose_only_stgcn_v2/log.txt")

# test WER per epoch
data = json.loads(result_path.read_text())
epochs   = sorted(data.keys(), key=int)
test_wer = [data[e] for e in epochs]
epochs   = [int(e) for e in epochs]

# val WER per epoch (from log.txt)
val_wer = []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        val_wer.append(d["test_wer"])

best_test_epoch = epochs[test_wer.index(min(test_wer))]
best_test_wer   = min(test_wer)

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Test WER per Epoch — AIHub KSL CSLR (stgcn_v2)", fontsize=14, fontweight="bold")

if len(val_wer) == len(epochs):
    ax.plot(epochs, val_wer, "b-o", markersize=4, alpha=0.6, label="Val WER (%)")

ax.plot(epochs, test_wer, "r-o", markersize=4, label="Test WER (%)")

ax.axvline(best_test_epoch, color="orange", linestyle="--", linewidth=1.2,
           label=f"Best Test: Epoch {best_test_epoch}  WER={best_test_wer:.2f}%")
ax.annotate(f"  {best_test_wer:.2f}%",
            xy=(best_test_epoch, best_test_wer),
            xytext=(best_test_epoch + 0.5, best_test_wer + 2),
            fontsize=10, color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange"))

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("WER (%)", fontsize=12)
ax.set_title("Word Error Rate (WER) per Epoch")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_xticks(epochs)

plt.tight_layout()
out_path = "out/aihub_ksl_cslr_pose_only_stgcn_v2/test_wer_curve.png"
plt.savefig(out_path, dpi=150)
print(f"저장 완료: {out_path}")
plt.show()
