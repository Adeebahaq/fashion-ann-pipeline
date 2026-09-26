import yaml
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    train = np.load("data/processed/train.npz")
    val = np.load("data/processed/val.npz")

    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(params["dense_units"], activation="relu"),
        layers.Dropout(params["dropout_rate"]),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train["images"], train["labels"],
        validation_data=(val["images"], val["labels"]),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    model.save("models/model.h5")
    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
    print("Model and history saved.")

if __name__ == "__main__":
    main()