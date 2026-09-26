import os
import numpy as np
from tensorflow import keras

def main():
    os.makedirs("data/raw", exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    np.savez("data/raw/train.npz", images=x_train, labels=y_train)
    np.savez("data/raw/test.npz", images=x_test, labels=y_test)
    print(f"Saved raw data: train={x_train.shape}, test={x_test.shape}")

if __name__ == "__main__":
    main()