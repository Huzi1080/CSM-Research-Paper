import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def train_baseline():
    print("Loading dataset...")

    train_df = pd.read_csv("data/train.csv")

    X = train_df.drop("label", axis=1)
    y = train_df["label"]

    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest baseline model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, pos_label="attack"),
        "recall": recall_score(y_val, y_pred, pos_label="attack"),
        "f1_score": f1_score(y_val, y_pred, pos_label="attack")
    }

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/baseline_model.pkl")
    pd.DataFrame([results]).to_csv("results/baseline_results.csv", index=False)

    print("Baseline training complete!")
    print("Results saved to results/baseline_results.csv")
    print("Model saved to models/baseline_model.pkl")
    print(results)

if __name__ == "__main__":
    train_baseline()

