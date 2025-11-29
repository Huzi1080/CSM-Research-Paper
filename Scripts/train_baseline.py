import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = joblib.load("models/baseline_model.pkl")

#Load test data 
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

def fgsm_attack(X, epsilon=0.05):
    """Simple FGSM-like perturbation: add small random noise"""
    perturb = epsilon * np.sign(np.random.randn(*X.shape))
    return X + perturb

def pgd_attack(X, epsilon=0.05, alpha=0.01, iters=5):
    """Simple PGD-like iterative attack"""
    X_adv = X.copy()
    for _ in range(iters):
        perturb = alpha * np.sign(np.random.randn(*X.shape))
        X_adv = np.clip(X_adv + perturb, X - epsilon, X + epsilon)  
    return X_adv


y_pred_clean = model.predict(X_test)
results_clean = {
    "dataset": "clean",
    "accuracy": accuracy_score(y_test, y_pred_clean),
    "precision": precision_score(y_test, y_pred_clean, pos_label="attack"),
    "recall": recall_score(y_test, y_pred_clean, pos_label="attack"),
    "f1_score": f1_score(y_test, y_pred_clean, pos_label="attack")
}

X_fgsm = fgsm_attack(X_test)
y_pred_fgsm = model.predict(X_fgsm)
results_fgsm = {
    "dataset": "fgsm",
    "accuracy": accuracy_score(y_test, y_pred_fgsm),
    "precision": precision_score(y_test, y_pred_fgsm, pos_label="attack"),
    "recall": recall_score(y_test, y_pred_fgsm, pos_label="attack"),
    "f1_score": f1_score(y_test, y_pred_fgsm, pos_label="attack")
}

X_pgd = pgd_attack(X_test)
y_pred_pgd = model.predict(X_pgd)
results_pgd = {
    "dataset": "pgd",
    "accuracy": accuracy_score(y_test, y_pred_pgd),
    "precision": precision_score(y_test, y_pred_pgd, pos_label="attack"),
    "recall": recall_score(y_test, y_pred_pgd, pos_label="attack"),
    "f1_score": f1_score(y_test, y_pred_pgd, pos_label="attack")
}

os.makedirs("results", exist_ok=True)
os.makedirs("attacks", exist_ok=True)

pd.DataFrame([results_clean, results_fgsm, results_pgd]).to_csv("results/attacked_results.csv", index=False)
np.save("attacks/adv_fgsm.npy", X_fgsm)
np.save("attacks/adv_pgd.npy", X_pgd)

print("Attacks complete!")
print("Results saved to results/attacked_results.csv")
print("Adversarial data saved to attacks/adv_fgsm.npy and attacks/adv_pgd.npy")
print(results_clean, results_fgsm, results_pgd)
