import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    model = keras.models.load_model("models/model.h5")
    test = np.load("data/processed/test.npz")
    x_test, y_test = test["images"], test["labels"]

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("models/confusion_matrix.png")

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(accuracy)}, f, indent=2)

    print(f"test_loss={loss:.4f}, test_accuracy={accuracy:.4f}")

if __name__ == "__main__":
    main()