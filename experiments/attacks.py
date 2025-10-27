"""
Generates FGSM and PGD adversarial examples using a small PyTorch surrogate model,
then evaluates the surrogate and a saved RandomForest baseline on clean and adversarial data.

Run:
py "experiments/attacks.py"
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Config
DATA_TRAIN = "data/train.csv"
DATA_TEST = "data/test.csv"
RF_MODEL_PATH = "models/baseline_model.pkl"   # RandomForest saved earlier
RESULTS_DIR = "results"
ATTACKS_DIR = "attacks"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 10         
LR = 1e-3
FGSM_EPS = 0.02     
PGD_EPS = 0.02
PGD_ALPHA = 0.005
PGD_STEPS = 10

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ATTACKS_DIR, exist_ok=True)

def metrics(y_true, y_pred, pos_label="attack"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "f1_score": f1_score(y_true, y_pred, pos_label=pos_label)
    }

# ---------------------------
# Simple PyTorch MLP surrogate
# ---------------------------
class SurrogateNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   
        )
    def forward(self, x):
        return self.net(x)

# Attack functions
def fgsm_attack(model, X, y, eps):
    model.eval()
    X_adv = X.clone().detach().to(DEVICE)
    X_adv.requires_grad = True
    outputs = model(X_adv)
    loss = nn.CrossEntropyLoss()(outputs, y.to(DEVICE))
    model.zero_grad()
    loss.backward()
    grad = X_adv.grad.data.sign()
    X_adv = X_adv + eps * grad
    return X_adv.detach()

def pgd_attack(model, X, y, eps, alpha, iters):
    model.eval()
    X_orig = X.clone().detach().to(DEVICE)
    X_adv = X_orig + 0.001 * torch.randn_like(X_orig).to(DEVICE)
    X_adv = torch.clamp(X_adv, X_orig - eps, X_orig + eps)
    for i in range(iters):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        loss = nn.CrossEntropyLoss()(outputs, y.to(DEVICE))
        model.zero_grad()
        loss.backward()
        grad = X_adv.grad.data.sign()
        X_adv = X_adv + alpha * grad
        # projection
        delta = torch.clamp(X_adv - X_orig, -eps, eps)
        X_adv = torch.clamp(X_orig + delta, -10.0, 10.0).detach()  
    return X_adv.detach()

# Main
def main():
    print("Loading preprocessed data...")
    train_df = pd.read_csv(DATA_TRAIN)
    test_df = pd.read_csv(DATA_TEST)


    X = train_df.drop("label", axis=1).values.astype(np.float32)
    y = train_df["label"].values
    X_test = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test = test_df["label"].values


    le = LabelEncoder()
    y_enc = le.fit_transform(y)         
    y_test_enc = le.transform(y_test)

    input_dim = X.shape[1]

  
    X_train_surr, X_val_surr, y_train_surr, y_val_surr = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    train_ds = TensorDataset(torch.from_numpy(X_train_surr), torch.from_numpy(y_train_surr))
    val_ds = TensorDataset(torch.from_numpy(X_val_surr), torch.from_numpy(y_val_surr))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    surrogate = SurrogateNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(surrogate.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Training surrogate neural network (small)...")
    surrogate.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = surrogate(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(train_loader.dataset)
   
        surrogate.eval()
        with torch.no_grad():
            val_x = torch.from_numpy(X_val_surr).to(DEVICE)
            val_y = torch.from_numpy(y_val_surr).to(DEVICE)
            val_out = surrogate(val_x)
            val_pred = val_out.argmax(dim=1).cpu().numpy()
            val_metrics = metrics(le.inverse_transform(val_y.cpu().numpy()), le.inverse_transform(val_pred))
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg:.4f} - val_acc: {val_metrics['accuracy']:.4f}")
        surrogate.train()

    torch.save(surrogate.state_dict(), os.path.join(ATTACKS_DIR, "surrogate.pt"))

    X_test_t = torch.from_numpy(X_test).to(torch.float32)
    y_test_t = torch.from_numpy(y_test_enc).to(torch.long)

    print("Generating FGSM adversarial examples...")
    surrogate.to(DEVICE)
    adv_fgsm = []
    adv_labels = []
    B = 1024
    surrogate.eval()
    with torch.no_grad():
        for i in range(0, X_test_t.shape[0], B):
            xb = X_test_t[i:i+B].to(DEVICE)
            yb = y_test_t[i:i+B].to(DEVICE)
            xb_adv = fgsm_attack(surrogate, xb.clone(), yb, eps=FGSM_EPS)
            adv_fgsm.append(xb_adv.cpu().numpy())
            adv_labels.append(yb.cpu().numpy())
    adv_fgsm = np.vstack(adv_fgsm)
    adv_labels = np.hstack(adv_labels)
    np.save(os.path.join(ATTACKS_DIR, "adv_examples_fgsm.npy"), adv_fgsm)
    print("Saved FGSM adversarial examples to", os.path.join(ATTACKS_DIR, "adv_examples_fgsm.npy"))

    print("Generating PGD adversarial examples...")
    adv_pgd = []
    adv_labels = []
    for i in range(0, X_test_t.shape[0], B):
        xb = X_test_t[i:i+B].to(DEVICE)
        yb = y_test_t[i:i+B].to(DEVICE)
        xb_adv = pgd_attack(surrogate, xb.clone(), yb, eps=PGD_EPS, alpha=PGD_ALPHA, iters=PGD_STEPS)
        adv_pgd.append(xb_adv.cpu().numpy())
        adv_labels.append(yb.cpu().numpy())
    adv_pgd = np.vstack(adv_pgd)
    adv_labels = np.hstack(adv_labels)
    np.save(os.path.join(ATTACKS_DIR, "adv_examples_pgd.npy"), adv_pgd)
    print("Saved PGD adversarial examples to", os.path.join(ATTACKS_DIR, "adv_examples_pgd.npy"))

    print("Loading RandomForest baseline model...")
    rf = joblib.load(RF_MODEL_PATH)
  
    print("Evaluating on clean test set...")
    clean_preds_rf = rf.predict(X_test)
    clean_metrics_rf = metrics(y_test, clean_preds_rf)
    print("RF clean:", clean_metrics_rf)

    print("Evaluating RF on FGSM adversarial examples...")
    adv_fgsm_preds_rf = rf.predict(adv_fgsm)
    fgsm_metrics_rf = metrics(y_test, adv_fgsm_preds_rf)
    print("RF FGSM:", fgsm_metrics_rf)

    print("Evaluating RF on PGD adversarial examples...")
    adv_pgd_preds_rf = rf.predict(adv_pgd)
    pgd_metrics_rf = metrics(y_test, adv_pgd_preds_rf)
    print("RF PGD:", pgd_metrics_rf)

    surrogate.eval()
    with torch.no_grad():
        s_clean_out = surrogate(torch.from_numpy(X_test).to(DEVICE))
        s_clean_pred = s_clean_out.argmax(dim=1).cpu().numpy()
        s_clean_metrics = metrics(le.inverse_transform(y_test_enc), le.inverse_transform(s_clean_pred))
        print("Surrogate clean:", s_clean_metrics)

        s_fgsm_out = surrogate(torch.from_numpy(adv_fgsm).to(DEVICE))
        s_fgsm_pred = s_fgsm_out.argmax(dim=1).cpu().numpy()
        s_fgsm_metrics = metrics(le.inverse_transform(y_test_enc), le.inverse_transform(s_fgsm_pred))
        print("Surrogate FGSM:", s_fgsm_metrics)

        s_pgd_out = surrogate(torch.from_numpy(adv_pgd).to(DEVICE))
        s_pgd_pred = s_pgd_out.argmax(dim=1).cpu().numpy()
        s_pgd_metrics = metrics(le.inverse_transform(y_test_enc), le.inverse_transform(s_pgd_pred))
        print("Surrogate PGD:", s_pgd_metrics)

    # Save results to CSV
    results_df = pd.DataFrame([
        {"model":"RandomForest_clean", **clean_metrics_rf},
        {"model":"RandomForest_FGSM_transfer", **fgsm_metrics_rf},
        {"model":"RandomForest_PGD_transfer", **pgd_metrics_rf},
        {"model":"Surrogate_clean", **s_clean_metrics},
        {"model":"Surrogate_FGSM", **s_fgsm_metrics},
        {"model":"Surrogate_PGD", **s_pgd_metrics}
    ])
    results_df.to_csv(os.path.join(RESULTS_DIR, "attacked_results.csv"), index=False)
    print("Saved attacked results to", os.path.join(RESULTS_DIR, "attacked_results.csv"))

if __name__ == "__main__":
    main()

