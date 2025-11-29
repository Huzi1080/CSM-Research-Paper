# scripts/analyze_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

RESULTS_DIR = "results"
ATTACKS_DIR = "attacks"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- 1) load summary metrics ----------
attacked = pd.read_csv(os.path.join(RESULTS_DIR, "attacked_results.csv"))  # clean/fgsm/pgd rows
# if you also want to splice in baseline_results.csv (val set), optional:
baseline_val = None
try:
    baseline_val = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
except FileNotFoundError:
    pass

# ---------- 2) bar charts: accuracy & f1 ----------
def bar_plot(metric, fn):
    plt.figure()
    order = ["clean", "fgsm", "pgd"]
    df = attacked.set_index("dataset").loc[order].reset_index()
    plt.bar(df["dataset"], df[metric])
    plt.title(f"{metric.replace('_',' ').upper()} Comparison (Clean vs FGSM vs PGD)")
    plt.xlabel("Dataset")
    plt.ylabel(metric.replace("_"," ").title())
    for i, v in enumerate(df[metric].values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fn), dpi=200)
    plt.close()

bar_plot("accuracy",   "accuracy_drop.png")
bar_plot("f1_score",   "f1_drop.png")

# ---------- 3) heatmap of metrics ----------
def performance_heatmap():
    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 3.6))
    view = attacked.set_index("dataset")[["accuracy","precision","recall","f1_score"]]
    data = view.values
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(range(view.shape[1])); ax.set_xticklabels(view.columns)
    ax.set_yticks(range(view.shape[0])); ax.set_yticklabels(view.index)
    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_title("Performance Degradation Heatmap")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "performance_heatmap.png"), dpi=200)
    plt.close()

performance_heatmap()

# ---------- 4) confusion matrices for each dataset ----------
def plot_cm(y_true, y_pred, title, fn):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal","attack"])
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fn), dpi=200)
    plt.close()

# load saved arrays from attacks_numpy.py
y_true_bin = np.load(os.path.join(RESULTS_DIR, "y_true_bin.npy"))
y_pred_clean = np.load(os.path.join(RESULTS_DIR, "y_pred_clean.npy"))
y_pred_fgsm  = np.load(os.path.join(RESULTS_DIR, "y_pred_fgsm.npy"))
y_pred_pgd   = np.load(os.path.join(RESULTS_DIR, "y_pred_pgd.npy"))

plot_cm(y_true_bin, y_pred_clean, "Confusion Matrix — Clean", "confusion_matrix_clean.png")
plot_cm(y_true_bin, y_pred_fgsm,  "Confusion Matrix — FGSM",  "confusion_matrix_fgsm.png")
plot_cm(y_true_bin, y_pred_pgd,   "Confusion Matrix — PGD",   "confusion_matrix_pgd.png")

# ---------- 5) ROC & PR curves ----------
def roc_and_pr_curves():
    p_clean = np.load(os.path.join(RESULTS_DIR, "p_clean.npy"))
    p_fgsm  = np.load(os.path.join(RESULTS_DIR, "p_fgsm.npy"))
    p_pgd   = np.load(os.path.join(RESULTS_DIR, "p_pgd.npy"))

    # ROC
    plt.figure()
    for name, p in [("Clean", p_clean), ("FGSM", p_fgsm), ("PGD", p_pgd)]:
        fpr, tpr, _ = roc_curve(y_true_bin, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (attack=positive)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=200)
    plt.close()

    # Precision-Recall
    plt.figure()
    for name, p in [("Clean", p_clean), ("FGSM", p_fgsm), ("PGD", p_pgd)]:
        prec, rec, _ = precision_recall_curve(y_true_bin, p)
        ap = average_precision_score(y_true_bin, p)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (attack=positive)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pr_curves.png"), dpi=200)
    plt.close()

roc_and_pr_curves()

# ---------- 6) simple adversarial risk grid map ----------
# you can tune these values; the idea is to tie practice -> governance concept
risk = pd.DataFrame({
    "attack": ["FGSM", "PGD"],
    "likelihood": [0.8, 0.9],  # how easy it was to cause degradation
    "impact":    [float(attacked.set_index("dataset").loc["fgsm","accuracy"]) - float(attacked.set_index("dataset").loc["clean","accuracy"]),
                  float(attacked.set_index("dataset").loc["pgd","accuracy"]) - float(attacked.set_index("dataset").loc["clean","accuracy"])]
})
# convert impact to positive magnitude drop
risk["impact"] = risk["impact"].abs()

plt.figure()
plt.scatter(risk["likelihood"], risk["impact"], s=400)
for _, row in risk.iterrows():
    plt.text(row["likelihood"]+0.01, row["impact"]+0.005, row["attack"])
plt.xlabel("Likelihood"); plt.ylabel("Impact (|ΔAccuracy|)")
plt.title("Adversarial Risk Grid (Simplified)")
plt.xlim(0,1.05); plt.ylim(0, max(0.01, risk["impact"].max()*1.2))
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "risk_grid_map.png"), dpi=200)
plt.close()

# ---------- 7) quick markdown summary ----------
lines = []
lines.append("# Adversarial Evaluation — Summary\n")
lines.append("## Aggregate Metrics\n")
lines.append(attacked.to_markdown(index=False))
if baseline_val is not None:
    lines.append("\n## Baseline Validation (hold-out)\n")
    lines.append(baseline_val.to_markdown(index=False))

lines += [
    "\n## Figures\n",
    "- accuracy_drop.png",
    "- f1_drop.png",
    "- performance_heatmap.png",
    "- confusion_matrix_clean.png / _fgsm.png / _pgd.png",
    "- roc_curves.png",
    "- pr_curves.png",
    "- risk_grid_map.png\n",
    "\n*(Attack=positive class; AUC and AP computed from predicted probabilities.)*\n"
]
with open(os.path.join(RESULTS_DIR, "summary_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ analysis complete. see the 'results/' folder.")

