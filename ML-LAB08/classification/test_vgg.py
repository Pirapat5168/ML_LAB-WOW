
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from vgg_model import predict_images

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    model = keras.models.load_model(OUT / "vgg_model.keras")
    classes = json.loads((OUT / "classes.json").read_text(encoding="utf-8"))
    X_test = np.load(OUT / "X_test.npy")
    y_test = np.load(OUT / "y_test.npy")

    idx = np.random.default_rng().choice(len(X_test), size=4, replace=False)
    pred, conf = predict_images(model, X_test[idx])

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, i, p, c in zip(axes, idx, pred, conf):
        true, guess = classes[y_test[i]], classes[p]
        ax.imshow(X_test[i])
        ax.axis("off")
        ax.set_title(f"True: {true}\nPred: {guess} ({c:.1%})",
                     color="green" if true == guess else "red", fontsize=10)
        print(f"True: {true:<6} | Pred: {guess:<6} | Confidence: {c:.1%}")
    fig.tight_layout()
    fig.savefig(OUT / "prediction_sample.png", dpi=150)
    print(f"Saved {OUT / 'prediction_sample.png'}")


if __name__ == "__main__":
    main()
