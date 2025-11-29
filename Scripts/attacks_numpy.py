# attacks_numpy.py  (revised)
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- reproducibility (simple) ----
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# ---- load model & data ----
model = joblib.load("models/baseline_model.pkl")

test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values  # array of strings: "attack"/"normal"

# ---- attacks (numpy-noise approximations) ----
def fgsm_attack(X, epsilon=0.05):
    """Simple FGSM-like perturbation: add small signed noise."""
    perturb = epsilon * np.sign(np.random.randn(*X.shape))
    return X + perturb

def pgd_attack(X, epsilon=0.05, alpha=0.01, iters=5):
    """Simple PGD-like iterative attack."""
    X_adv = X.copy()
    for _ in range(iters):
        perturb = alpha * np.sign(np.random.randn(*X.shape))
        X_adv = np.clip(X_adv + perturb, X - epsilon, X + epsilon)
    return X_adv

# ---- evaluate clean ----
y_pred_clean = model.predict(X_test)
results_clean = {
    "dataset": "clean",
    "accuracy":  accuracy_score(y_test, y_pred_clean),
    "precision": precision_score(y_test, y_pred_clean, pos_label="attack"),
    "recall":    recall_score(y_test, y_pred_clean,   pos_label="attack"),
    "f1_score":  f1_score(y_test, y_pred_clean,       pos_label="attack"),
}

# ---- craft adversarial examples ----
X_fgsm = fgsm_attack(X_test)
y_pred_fgsm = model.predict(X_fgsm)
results_fgsm = {
    "dataset": "fgsm",
    "accuracy":  accuracy_score(y_test, y_pred_fgsm),
    "precision": precision_score(y_test, y_pred_fgsm, pos_label="attack"),
    "recall":    recall_score(y_test, y_pred_fgsm,   pos_label="attack"),
    "f1_score":  f1_score(y_test, y_pred_fgsm,       pos_label="attack"),
}

X_pgd = pgd_attack(X_test)
y_pred_pgd = model.predict(X_pgd)
results_pgd = {
    "dataset": "pgd",
    "accuracy":  accuracy_score(y_test, y_pred_pgd),
    "precision": precision_score(y_test, y_pred_pgd, pos_label="attack"),
    "recall":    recall_score(y_test, y_pred_pgd,   pos_label="attack"),
    "f1_score":  f1_score(y_test, y_pred_pgd,       pos_label="attack"),
}

# ---- helper: probability of "attack" for ROC/PR curves ----
def proba_attack(X):
    """Return P(y='attack') for each row. Falls back to hard preds if no proba."""
    try:
        proba = model.predict_proba(X)
        attack_idx = list(model.classes_).index("attack")
        return proba[:, attack_idx]
    except Exception:
        # e.g., if model lacks predict_proba
        return (model.predict(X) == "attack").astype(float)

# binary ground truth (attack=1, normal=0) for curves & confusion matrices
y_true_bin = (y_test == "attack").astype(int)

# probabilities
p_clean = proba_attack(X_test)
p_fgsm  = proba_attack(X_fgsm)
p_pgd   = proba_attack(X_pgd)

# ---- save artifacts ----
os.makedirs("results", exist_ok=True)
os.makedirs("attacks", exist_ok=True)

# metrics table
pd.DataFrame([results_clean, results_fgsm, results_pgd]) \
  .to_csv("results/attacked_results.csv", index=False)

# adversarial datasets
np.save("attacks/adv_fgsm.npy", X_fgsm)
np.save("attacks/adv_pgd.npy",  X_pgd)

# predictions (0/1) & probabilities for analysis plots
np.save("results/y_true_bin.npy", y_true_bin)
np.save("results/y_pred_clean.npy", (y_pred_clean == "attack").astype(int))
np.save("results/y_pred_fgsm.npy",  (y_pred_fgsm  == "attack").astype(int))
np.save("results/y_pred_pgd.npy",   (y_pred_pgd   == "attack").astype(int))
np.save("results/p_clean.npy", p_clean)
np.save("results/p_fgsm.npy",  p_fgsm)
np.save("results/p_pgd.npy",   p_pgd)

# ---- console summary ----
print("Attacks complete!")
print("Results saved → results/attacked_results.csv")
print("Adversarial arrays → attacks/adv_fgsm.npy, attacks/adv_pgd.npy")
print("Plot inputs → results/y_*.npy, results/p_*.npy")
print(results_clean, results_fgsm, results_pgd)
