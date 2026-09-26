import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train = np.load("data/raw/train.npz")
    test = np.load("data/raw/test.npz")
    x_train_full = np.clip(train["images"].astype("float32") / 255.0, 0.0, 1.0
    y_train_full = train["labels"]
    x_test = test["images"].astype("float32")/ 255.0, 0.0, 1.0
    y_test = test["labels"]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=params["test_size"],
        random_state=params["seed"],
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/train.npz", images=x_train, labels=y_train)
    np.savez("data/processed/val.npz", images=x_val, labels=y_val)
    np.savez("data/processed/test.npz", images=x_test, labels=y_test)
    print(f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

if __name__ == "__main__":
    main()