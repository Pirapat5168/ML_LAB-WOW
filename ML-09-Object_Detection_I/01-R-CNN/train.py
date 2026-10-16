"""R-CNN (simplified, trainable on CPU): region proposals -> warp each region to 64x64
-> CNN classifies it (background / real / fake) and refines the box (regression head).

    python train.py --limit 1000 --epochs 15
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import *   # noqa: E402,F403

OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"
IMG_SIZE = 64


def box_loss(y_true, y_pred):
    """Smooth-L1 on box deltas, only for positive regions (y_true[:, 4] is the mask)."""
    d, m = y_true[..., :4], y_true[..., 4]
    err = tf.abs(d - y_pred)
    per = tf.reduce_sum(tf.where(err < 1.0, 0.5 * err * err, err - 0.5), axis=-1)
    return tf.reduce_sum(per * m) / tf.maximum(tf.reduce_sum(m), 1.0)


def warp(img, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = max(x2, x1 + 2), max(y2, y1 + 2)
    crop = img[y1:y2, x1:x2]
    return cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)


def build_regions(split, limit, per_image, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, bt = [], [], []
    for path in list_images(split, limit, seed):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gb, gc = read_labels(path, w, h)
        if len(gb) == 0:
            continue
        rois, labels, gi = sample_rois(gb, gc, w, h, per_image, rng, pos_frac=0.5)
        if len(rois) == 0:
            continue
        deltas = np.zeros((len(rois), 5), np.float32)
        pos = labels > 0
        if pos.any():
            deltas[pos, :4] = encode(rois[pos], gb[gi[pos]]); deltas[pos, 4] = 1
        xs += [warp(img, r) for r in rois]; ys.append(labels); bt.append(deltas)
    x = np.asarray(xs, np.uint8)          # uint8 (4x smaller); scaled to 0-1 inside the model
    return x, {"cls": np.concatenate(ys), "box": np.concatenate(bt)}


def make_model():
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inp)
    for f in (32, 64, 128):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    cls = tf.keras.layers.Dense(NUM_OUT, activation="softmax", name="cls")(x)
    box = tf.keras.layers.Dense(4, name="box")(x)
    model = tf.keras.Model(inp, [cls, box], name="RCNN_region_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss={"cls": "sparse_categorical_crossentropy", "box": box_loss},
                  loss_weights={"cls": 1.0, "box": 1.0},
                  metrics={"cls": ["accuracy"]})
    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found - run train.py first")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def detect(model, img, max_regions=150, score_thr=0.5, nms_thr=0.4):
    """Full detection on one BGR image -> [(x1,y1,x2,y2,cls,score)]."""
    h, w = img.shape[:2]
    props = make_proposals(w, h, max_regions)
    batch = np.asarray([warp(img, b) for b in props], np.float32)   # model rescales 0-255 -> 0-1
    probs, deltas = model.predict(batch, batch_size=256, verbose=0)
    return postprocess(props, probs, deltas, w, h, score_thr, nms_thr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1000, help="max source images for training (0 = all)")
    p.add_argument("--val-limit", type=int, default=300)
    p.add_argument("--per-image", type=int, default=12, help="regions sampled per image")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()

    print("Building training regions ...")
    x, y = build_regions("train", a.limit, a.per_image)
    xv, yv = build_regions("val", a.val_limit, a.per_image, seed=1)
    print(f"train regions: {len(x)} (foreground {(y['cls'] > 0).sum()}), val regions: {len(xv)}")

    model = make_model()
    cb = [tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
          tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
    def to_ds(xs, ys, shuffle):          # batches go to the GPU one at a time (no huge GPU copy)
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        if shuffle:
            ds = ds.shuffle(min(len(xs), 10000), reshuffle_each_iteration=True)
        return ds.batch(a.batch).prefetch(2)

    hist = model.fit(to_ds(x, y, True), validation_data=to_ds(xv, yv, False),
                     epochs=a.epochs, callbacks=cb)
    model.save(MODEL_PATH)
    save_history(hist, OUT, "R-CNN")
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
