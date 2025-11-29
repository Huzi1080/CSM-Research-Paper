import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

# reproducibility
import random
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# Load model
model = joblib.load("models/baseline_model.pkl")

# Load test data
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

# ---------------- FGSM Attack ----------------
def fgsm_attack(X, epsilon=0.05):
    """Simple FGSM-like attack using random noise."""
    perturb = epsilon * np.sign(np.random.randn(*X.shape))
    return X + perturb

# ---------------- PGD Attack ----------------
def pgd_attack(X, epsilon=0.05, alpha=0.01, iters=5):
    """Simple PGD-like iterative attack."""
    X_adv = X.copy()
    for _ in range(iters):
        perturb = alpha * np.sign(np.random.randn(*X.shape))
        X_adv = np.clip(X_adv + perturb, X - epsilon, X + epsilon)
    return X_adv

# ---------------- Helper for probabilities ----------------
def proba_attack(X):
    try:
        proba = model.predict_proba(X)
        attack_idx = list(model.classes_).index("attack")
        return proba[:, attack_idx]
    except Exception:
        return (model.predict(X) == "attack").astype(float)

# ---------------- Run evaluations ----------------
def evaluate(dataset_name, X):
    y_pred = model.predict(X)
    return {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="attack"),
        "recall": recall_score(y_test, y_pred, pos_label="attack"),
        "f1_score": f1_score(y_test, y_pred, pos_label="attack")
    }

results = []
results.append(evaluate("clean", X_test))

X_fgsm = fgsm_attack(X_test)
results.append(evaluate("fgsm", X_fgsm))

X_pgd = pgd_attack(X_test)
results.append(evaluate("pgd", X_pgd))

# ---------------- Save results ----------------
os.makedirs("results", exist_ok=True)
os.makedirs("attacks", exist_ok=True)

pd.DataFrame(results).to_csv("results/attacked_results.csv", index=False)
np.save("attacks/adv_fgsm.npy", X_fgsm)
np.save("attacks/adv_pgd.npy", X_pgd)

# save probabilities for later ROC/PR analysis
y_true_bin = (y_test == "attack").astype(int)
p_clean = proba_attack(X_test)
p_fgsm = proba_attack(X_fgsm)
p_pgd = proba_attack(X_pgd)

np.save("results/y_true_bin.npy", y_true_bin)
np.save("results/p_clean.npy", p_clean)
np.save("results/p_fgsm.npy", p_fgsm)
np.save("results/p_pgd.npy", p_pgd)

print("✅ Attacks complete!")
print("Results saved to results/attacked_results.csv")
print(results)
