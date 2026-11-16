import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

log_path = "out/aihub_ksl_cslr_pose_only_stgcn_v2/log.txt"

epochs, train_loss, val_loss, wer = [], [], [], []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        epochs.append(d["epoch"])
        train_loss.append(d["train_loss"])
        val_loss.append(d["test_loss"])
        wer.append(d["test_wer"])

best_epoch = epochs[wer.index(min(wer))]
best_wer   = min(wer)
TEST_WER   = 12.418196328810854

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Training Curve — AIHub KSL CSLR", fontsize=15, fontweight="bold")

# ── Loss ──────────────────────────────────────────────────
ax1.plot(epochs, train_loss, "b-o", markersize=4, label="Train Loss")
ax1.plot(epochs, val_loss,   "r-o", markersize=4, label="Val Loss")
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Loss per Epoch")
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# ── WER ───────────────────────────────────────────────────
ax2.plot(epochs, wer, "g-o", markersize=4, label="Val WER (%)")
ax2.axvline(best_epoch, color="orange", linestyle="--", linewidth=1.2,
            label=f"Best: Epoch {best_epoch}  WER={best_wer:.2f}%")
ax2.annotate(f"  {best_wer:.2f}%",
             xy=(best_epoch, best_wer),
             xytext=(best_epoch + 0.5, best_wer + 2),
             fontsize=10, color="darkorange",
             arrowprops=dict(arrowstyle="->", color="darkorange"))
ax2.scatter([best_epoch], [TEST_WER], color="purple", s=120, zorder=5,
            marker="*", label=f"Test WER (Best ckpt) = {TEST_WER:.2f}%")
ax2.annotate(f"  Test {TEST_WER:.2f}%",
             xy=(best_epoch, TEST_WER),
             xytext=(best_epoch + 1, TEST_WER + 2),
             fontsize=10, color="purple",
             arrowprops=dict(arrowstyle="->", color="purple"))
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("WER (%)", fontsize=12)
ax2.set_title("Word Error Rate (WER) per Epoch")
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax2.set_xticks(epochs)
plt.tight_layout()

out_path = "out/aihub_ksl_cslr_pose_only_stgcn_v2/training_curve.png"
plt.savefig(out_path, dpi=150)
print(f"저장 완료: {out_path}")
plt.show()
