"""Fast R-CNN (simplified, trainable on CPU): the WHOLE image goes through the CNN once,
each region proposal is pooled from the shared feature map (ROI pooling 7x7), then two
heads predict the class (background / real / fake) and the box refinement.

    python train.py --limit 1000 --epochs 20
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
IMG_SIZE = 128
N_ROI = 32          # regions sampled per image during training
POOL = 7


@tf.keras.utils.register_keras_serializable(package="lab9")
class ROIPool(tf.keras.layers.Layer):
    """Pools each ROI (normalized [y1,x1,y2,x2]) from the shared feature map to POOLxPOOL."""

    def __init__(self, pool=POOL, **kw):
        super().__init__(**kw)
        self.pool = pool

    def call(self, inputs):
        feats, rois = inputs
        b, n, c = tf.shape(feats)[0], tf.shape(rois)[1], feats.shape[-1]
        crops = tf.image.crop_and_resize(
            feats, tf.reshape(rois, [-1, 4]), tf.repeat(tf.range(b), n), (self.pool, self.pool))
        return tf.reshape(crops, [b, n, self.pool * self.pool * c])

    def compute_output_shape(self, input_shapes):
        f, r = input_shapes
        return (f[0], r[1], self.pool * self.pool * f[-1])

    def get_config(self):
        return {**super().get_config(), "pool": self.pool}


def box_loss(y_true, y_pred):
    """Smooth-L1 on box deltas, only for positive ROIs (y_true[..., 4] is the mask)."""
    d, m = y_true[..., :4], y_true[..., 4]
    err = tf.abs(d - y_pred)
    per = tf.reduce_sum(tf.where(err < 1.0, 0.5 * err * err, err - 0.5), axis=-1)
    return tf.reduce_sum(per * m) / tf.maximum(tf.reduce_sum(m), 1.0)


def make_model():
    image = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    rois = tf.keras.Input((None, 4), name="rois")
    x = image
    for f in (32, 64, 128):                       # stride 8 -> 16x16 feature map
        x = tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = ROIPool()([x, rois])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    cls = tf.keras.layers.Dense(NUM_OUT, activation="softmax", name="cls")(x)
    box = tf.keras.layers.Dense(4, name="box")(x)
    model = tf.keras.Model({"image": image, "rois": rois}, [cls, box], name="Fast_RCNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss={"cls": "sparse_categorical_crossentropy", "box": box_loss},
                  loss_weights={"cls": 1.0, "box": 1.0},
                  metrics={"cls": ["accuracy"]}, jit_compile=False)
    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found - run train.py first")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(jit_compile=False)   # avoid XLA on GPU (CropAndResize is not supported by XLA)
    return model


def to_norm(boxes, w, h):
    """pixel xyxy -> normalized [y1,x1,y2,x2] used by ROI pooling."""
    return np.stack([boxes[:, 1] / h, boxes[:, 0] / w, boxes[:, 3] / h, boxes[:, 2] / w], 1).astype(np.float32)


def prep(img):
    return cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def detect(model, img, max_regions=150, score_thr=0.5, nms_thr=0.4):
    """Full detection on one BGR image -> [(x1,y1,x2,y2,cls,score)]."""
    h, w = img.shape[:2]
    props = make_proposals(w, h, max_regions)
    probs, deltas = model.predict({"image": prep(img)[None], "rois": to_norm(props, w, h)[None]}, verbose=0)
    return postprocess(props, probs[0], deltas[0], w, h, score_thr, nms_thr)


# ---------------------------------------------------------------- training data
def prep_u8(img):
    return cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)


def load_split(split, limit, seed=0):
    imgs, metas = [], []
    for path in list_images(split, limit, seed):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gb, gc = read_labels(path, w, h)
        if len(gb):
            imgs.append(prep_u8(img)); metas.append((w, h, gb, gc))
    return np.asarray(imgs, np.uint8), metas      # uint8 saves RAM; scaled in the generator


def make_sample(meta, rng):
    w, h, gb, gc = meta
    rois, labels, gi = sample_rois(gb, gc, w, h, N_ROI, rng, pos_frac=0.25)
    if len(rois) < N_ROI:                                  # pad (rare) by repeating
        k = rng.integers(0, len(rois), N_ROI - len(rois))
        rois, labels, gi = (np.concatenate([a, a[k]]) for a in (rois, labels, gi))
    t = np.zeros((N_ROI, 5), np.float32)
    pos = labels > 0
    if pos.any():
        t[pos, :4] = encode(rois[pos], gb[gi[pos]]); t[pos, 4] = 1
    return to_norm(rois, w, h), labels.astype(np.int32), t


def make_dataset(imgs, metas, batch, seed, repeat):
    def gen():
        rng = np.random.default_rng(seed)
        while True:
            for i in rng.permutation(len(metas)) if repeat else range(len(metas)):
                r, l, t = make_sample(metas[i], rng)
                yield {"image": imgs[i].astype(np.float32) / 255.0, "rois": r}, {"cls": l, "box": t}
            if not repeat:
                return
    sig = ({"image": tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32), "rois": tf.TensorSpec((N_ROI, 4), tf.float32)},
           {"cls": tf.TensorSpec((N_ROI,), tf.int32), "box": tf.TensorSpec((N_ROI, 5), tf.float32)})
    return tf.data.Dataset.from_generator(gen, output_signature=sig).batch(batch).prefetch(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1000, help="max source images for training (0 = all)")
    p.add_argument("--val-limit", type=int, default=300)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16, help="images per batch (each has %d ROIs)" % N_ROI)
    a = p.parse_args()

    print("Loading images ...")
    imgs, metas = load_split("train", a.limit)
    vimgs, vmetas = load_split("val", a.val_limit, seed=1)
    print(f"train images: {len(metas)}, val images: {len(vmetas)}")
    train_ds = make_dataset(imgs, metas, a.batch, seed=0, repeat=True)
    val_ds = make_dataset(vimgs, vmetas, a.batch, seed=1, repeat=False).cache()   # fixed ROIs -> stable val curve

    model = make_model()
    cb = [tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
          tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
    hist = model.fit(train_ds, steps_per_epoch=int(np.ceil(len(metas) / a.batch)),
                     validation_data=val_ds, epochs=a.epochs, callbacks=cb)
    model.save(MODEL_PATH)
    save_history(hist, OUT, "Fast R-CNN")
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
