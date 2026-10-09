import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2

CONFIGS = {
    # VGG-style trained from scratch (baseline)
    "VGG_2block_128": {"filters": [32, 64], "dense": 128, "dropout": 0.5},
    "VGG_3block_128": {"filters": [32, 64, 128], "dense": 128, "dropout": 0.5},
    "VGG_3block_256": {"filters": [32, 64, 128], "dense": 256, "dropout": 0.5},
    "VGG_4block_256": {"filters": [32, 64, 128, 256], "dense": 256, "dropout": 0.5},
    # MobileNetV2 transfer learning (unfreeze_last = number of backbone layers fine-tuned)
    "MNv2_frozen_64": {"transfer": "mobilenet_v2", "dense": 64, "dropout": 0.5, "unfreeze_last": 0},
    "MNv2_frozen_256": {"transfer": "mobilenet_v2", "dense": 256, "dropout": 0.5, "unfreeze_last": 0},
    "MNv2_ft30_128": {"transfer": "mobilenet_v2", "dense": 128, "dropout": 0.5, "unfreeze_last": 30},
    "MNv2_ft60_128": {"transfer": "mobilenet_v2", "dense": 128, "dropout": 0.5, "unfreeze_last": 60},
}


def build_vgg(config, input_shape, num_classes, mean, std):
    """Input = raw RGB pixels (0-255); scaling/standardization happens inside the model."""
    if config.get("transfer") == "mobilenet_v2":
        return _build_mobilenet_transfer(config, input_shape, num_classes)

    inputs = keras.Input(shape=input_shape)
    x = layers.Normalization(mean=mean.tolist(), variance=(std ** 2).tolist())(inputs)
    x = layers.RandomFlip("horizontal")(x)       # light augmentation (training only)
    x = layers.RandomRotation(0.05)(x)

    for f in config["filters"]:
        for _ in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False,
                              kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    # GlobalAveragePooling instead of Flatten: with 128x128 input, Flatten after
    # 2 pools gives 65,536 features -> ~8M weights in Dense, which collapsed to 0.5000.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config["dense"], activation="relu")(x)
    x = layers.Dropout(config["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _build_mobilenet_transfer(config, input_shape, num_classes):
    unfreeze = config.get("unfreeze_last", 0)

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(scale=1. / 127.5, offset=-1.0)(inputs)   # -> [-1, 1] as MobileNetV2 expects
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)

    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = unfreeze > 0
    if unfreeze > 0:
        for layer in base.layers[:-unfreeze]:
            layer.trainable = False
    x = base(x, training=False)                  # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config["dense"], activation="relu")(x)
    x = layers.Dropout(config["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    lr = 1e-3 if unfreeze == 0 else 1e-4         # fine-tuning needs a lower learning rate
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class EpochCheckpointEval(keras.callbacks.Callback):
    """Record train/val/test accuracy at chosen epochs (to compare different epoch counts)."""

    def __init__(self, X_test, y_test, checkpoints):
        super().__init__()
        self.X_test, self.y_test = X_test, y_test
        self.checkpoints = set(checkpoints)
        self.records = {}

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep in self.checkpoints:
            _, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            self.records[ep] = {
                "train_acc": float(logs["accuracy"]),
                "val_acc": float(logs["val_accuracy"]),
                "test_acc": float(test_acc),
            }


def train_vgg(model, splits, epochs_list, batch_size=32):
    """Train for max(epochs_list) epochs; record accuracy at every epoch in epochs_list."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    ckpt = EpochCheckpointEval(X_test, y_test, epochs_list)
    lr_sched = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    t0 = time.time()
    hist = model.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=max(epochs_list), batch_size=batch_size,
                     callbacks=[ckpt, lr_sched], verbose=2)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return history, ckpt.records, time.time() - t0


def save_model(model, path):
    model.save(path)


def predict_images(model, X):
    """Return (predicted class ids, confidence) for a batch of RGB uint8 images."""
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1), np.max(probs, axis=1)