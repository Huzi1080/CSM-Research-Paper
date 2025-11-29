import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_nsl_kdd():
    print("Downloading NSL-KDD dataset...")

    url_train = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    url_test = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

    col_names = [*map(str, range(41)), "label", "extra"]

    train_df = pd.read_csv(url_train, names=col_names)
    test_df = pd.read_csv(url_test, names=col_names)

    train_df.drop("extra", axis=1, inplace=True)
    test_df.drop("extra", axis=1, inplace=True)


    train_df['label'] = train_df['label'].apply(lambda x: "attack" if x != "normal" else "normal")
    test_df['label'] = test_df['label'].apply(lambda x: "attack" if x != "normal" else "normal")

    enc = LabelEncoder()
    for col in ['1', '2', '3']:  # protocol, service, flag
        train_df[col] = enc.fit_transform(train_df[col])
        test_df[col] = enc.transform(test_df[col])

    scaler = StandardScaler()
    feat_cols = train_df.columns[:-1]
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("✅ Dataset saved to data/train.csv and data/test.csv")


if __name__ == "__main__":
    load_nsl_kdd()
