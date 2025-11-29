# Scripts/attacks_numpy.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# reproducibility
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# --- Load model ---
MODEL_PATH = "models/baseline_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")

model = joblib.load(MODEL_PATH)

# --- Load test data ---
TEST_PATH = "data/test.csv"
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Test data not found at {TEST_PATH}. Run download_nsl_kdd.py first.")

test_df = pd.read_csv(TEST_PATH)
if "label" not in test_df.columns:
    raise ValueError("test.csv must contain a 'label' column.")

X_test = test_df.drop("label", axis=1).values
y_test_raw = test_df["label"].values  # keep raw for diagnostics

# Convert y_test to integer 0/1 to match typical classifier outputs
# If labels are strings like "attack"/"normal", map attack->1, normal->0
def to_binary_labels(y):
    # if already numeric (int/float), coerce to ints
    try:
        arr = np.asarray(y)
        # check if any element is non-numeric (dtype==object or contains non-digits)
        if arr.dtype.kind in ("i", "u", "f"):
            return arr.astype(int)
        # otherwise handle common string labels
        unique_vals = np.unique(arr)
        # common mapping
        if set(unique_vals) >= {"attack", "normal"} or set(unique_vals) <= {"attack", "normal"}:
            return np.where(arr == "attack", 1, 0).astype(int)
        # sometimes labels are 'normal' and many attack names -> treat 'normal' as 0 and others as 1
        if "normal" in unique_vals:
            return np.where(arr == "normal", 0, 1).astype(int)
        # fallback: if there are exactly two unique values, map first->0 second->1
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            return np.vectorize(lambda v: mapping[v])(arr).astype(int)
        # final fallback: try casting to int (may fail)
        return arr.astype(int)
    except Exception:
        # last resort
        return np.where(np.asarray(y) == "attack", 1, 0).astype(int)

y_test = to_binary_labels(y_test_raw)

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
    """
    Return the model's probability/confidence for the 'attack' class.
    Works whether model.classes_ are strings like ['normal','attack'] or integers [0,1].
    If predict_proba not available, returns float predictions (0/1) cast to float.
    """
    # try predict_proba
    try:
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        # prefer integer label 1 if present
        if 1 in classes:
            attack_idx = classes.index(1)
        # else prefer string 'attack' if present
        elif "attack" in classes:
            attack_idx = classes.index("attack")
        # else fallback to the class with higher "positive" semantic if only two classes (choose index 1)
        else:
            attack_idx = 1 if len(classes) > 1 else 0
        return proba[:, attack_idx]
    except Exception:
        # fallback: model.predict returns labels; convert to float probability-like values
        preds = model.predict(X)
        # if preds are strings like 'attack', map to 1.0/0.0
        try:
            preds_arr = np.asarray(preds)
            if preds_arr.dtype.kind in ("U", "O"):
                preds_num = np.where(preds_arr == "attack", 1.0, 0.0)
            else:
                preds_num = preds_arr.astype(float)
            return preds_num
        except Exception:
            # last resort: return zeros
            return np.zeros(X.shape[0], dtype=float)

# ---------------- Run evaluations ----------------
def evaluate(dataset_name, X):
    y_pred = model.predict(X)
    # ensure predictions are integer binary 0/1
    try:
        y_pred_int = np.asarray(y_pred).astype(int)
    except Exception:
        # if predictions are strings like 'attack'/'normal'
        y_pred_int = np.where(np.asarray(y_pred) == "attack", 1, 0).astype(int)

    return {
        "dataset": dataset_name,
        "accuracy": float(accuracy_score(y_test, y_pred_int)),
        "precision": float(precision_score(y_test, y_pred_int)),
        "recall": float(recall_score(y_test, y_pred_int)),
        "f1_score": float(f1_score(y_test, y_pred_int))
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
y_true_bin = (y_test == 1).astype(int)  # already binary 0/1
p_clean = proba_attack(X_test)
p_fgsm = proba_attack(X_fgsm)
p_pgd = proba_attack(X_pgd)

# ensure shapes are 1-D arrays for saving
p_clean = np.asarray(p_clean).ravel()
p_fgsm = np.asarray(p_fgsm).ravel()
p_pgd = np.asarray(p_pgd).ravel()

np.save("results/y_true_bin.npy", y_true_bin)
np.save("results/p_clean.npy", p_clean)
np.save("results/p_fgsm.npy", p_fgsm)
np.save("results/p_pgd.npy", p_pgd)

print("✅ Attacks complete!")
print("Results saved to results/attacked_results.csv")
print(results)

