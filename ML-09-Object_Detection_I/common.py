"""Shared helpers for ML-09 (R-CNN / Fast R-CNN) on the Lab 7 real-vs-fake faces data.

Dataset layout (YOLO format, created by prepare_dataset.py):
    dataset/{train,val,test}/{images,labels}/   labels: "cls xc yc w h" (normalized)
Classes: 0 = real, 1 = fake.  Classifier output index 0 = background, 1 = real, 2 = fake.
"""
from functools import lru_cache
from pathlib import Path
import random
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
NAMES = ["real", "fake"]          # object classes (YOLO ids 0, 1)
NUM_FG = len(NAMES)
NUM_OUT = NUM_FG + 1              # + background at index 0
BOX_VAR = np.array([0.1, 0.1, 0.2, 0.2], np.float32)   # box-delta scaling
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}   # BGR: real=green, fake=red


# ---------------------------------------------------------------- dataset
def list_images(split, limit=0, seed=0):
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {images_dir} or {labels_dir}. Run prepare_dataset.py first.")
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMG_EXT and (labels_dir / (p.stem + ".txt")).exists())
    if limit > 0 and len(files) > limit:
        random.Random(seed).shuffle(files)
        files = sorted(files[:limit])
    if not files:
        raise RuntimeError(f"No labeled images in {images_dir}")
    return files


def read_labels(path, w, h):
    """YOLO label file -> (boxes Nx4 float xyxy in pixels, classes N)."""
    label = path.parent.parent / "labels" / (path.stem + ".txt")
    boxes, cls = [], []
    if label.exists():
        for line in label.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            try:
                c = int(p[0]); xc, yc, bw, bh = map(float, p[1:])
            except ValueError:
                continue
            if not 0 <= c < NUM_FG:
                continue
            x1, y1 = max(0, (xc - bw / 2) * w), max(0, (yc - bh / 2) * h)
            x2, y2 = min(w, (xc + bw / 2) * w), min(h, (yc + bh / 2) * h)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                boxes.append([x1, y1, x2, y2]); cls.append(c)
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(cls, np.int32)


# ---------------------------------------------------------------- boxes
def iou_matrix(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4); b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def encode(props, gt):
    """Box deltas (tx,ty,tw,th)/BOX_VAR that turn proposal -> ground truth."""
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    pcx, pcy = props[:, 0] + pw / 2, props[:, 1] + ph / 2
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gcx, gcy = gt[:, 0] + gw / 2, gt[:, 1] + gh / 2
    d = np.stack([(gcx - pcx) / pw, (gcy - pcy) / ph, np.log(gw / pw), np.log(gh / ph)], 1)
    return (d / BOX_VAR).astype(np.float32)


def decode(props, deltas, w, h):
    d = deltas * BOX_VAR
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    pcx, pcy = props[:, 0] + pw / 2, props[:, 1] + ph / 2
    cx, cy = pcx + d[:, 0] * pw, pcy + d[:, 1] * ph
    bw, bh = pw * np.exp(np.clip(d[:, 2], -4, 4)), ph * np.exp(np.clip(d[:, 3], -4, 4))
    out = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w); out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out.astype(np.float32)


def nms(boxes, scores, thr=0.5):
    order = np.argsort(-scores); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        order = order[1:][iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0] < thr]
    return keep


# ---------------------------------------------------------------- proposals
@lru_cache(maxsize=64)
def make_proposals(w, h, max_regions=150):
    """Multi-scale sliding-window region proposals (cheap stand-in for Selective Search).

    Side lengths are fractions of the shorter image side; aspect ratios cover
    upright faces.  If more than max_regions exist, an even subset is kept.
    """
    s = min(w, h)
    boxes = []
    for frac in (0.35, 0.5, 0.65, 0.8, 0.95):
        for ratio in (1.0, 1.25):                       # box height / width
            bw = min(w, frac * s / ratio ** 0.5); bh = min(h, bw * ratio)
            sx, sy = max(bw * 0.4, 1), max(bh * 0.4, 1)
            xs = np.arange(0, max(w - bw, 0) + 1e-3, sx); ys = np.arange(0, max(h - bh, 0) + 1e-3, sy)
            if len(xs) == 0 or xs[-1] < w - bw - 1: xs = np.append(xs, max(w - bw, 0))
            if len(ys) == 0 or ys[-1] < h - bh - 1: ys = np.append(ys, max(h - bh, 0))
            for y in ys:
                for x in xs:
                    boxes.append([x, y, x + bw, y + bh])
    boxes = np.array(boxes, np.float32)
    if len(boxes) > max_regions:
        idx = np.linspace(0, len(boxes) - 1, max_regions).astype(int)
        boxes = boxes[idx]
    return boxes


def jitter_box(box, w, h, rng, amount=0.15):
    bw, bh = box[2] - box[0], box[3] - box[1]
    dx, dy = rng.uniform(-amount, amount, 2) * (bw, bh)
    sw, sh = np.exp(rng.uniform(-amount, amount, 2))
    cx, cy = (box[0] + box[2]) / 2 + dx, (box[1] + box[3]) / 2 + dy
    nb = np.array([cx - bw * sw / 2, cy - bh * sh / 2, cx + bw * sw / 2, cy + bh * sh / 2], np.float32)
    nb[[0, 2]] = np.clip(nb[[0, 2]], 0, w); nb[[1, 3]] = np.clip(nb[[1, 3]], 0, h)
    return nb


def sample_rois(gt_boxes, gt_cls, w, h, n_total, rng, pos_frac=0.25, pos_iou=0.5, neg_iou=0.3):
    """Training ROIs for one image -> (rois Nx4, labels N (0=bg,1..), gt-box index N)."""
    props = make_proposals(w, h, 400)
    extra = [jitter_box(b, w, h, rng) for b in gt_boxes for _ in range(6)] + list(gt_boxes)
    props = np.concatenate([props, np.array(extra, np.float32).reshape(-1, 4)], 0)
    iou = iou_matrix(props, gt_boxes)
    best, best_iou = iou.argmax(1), iou.max(1)
    pos = np.where(best_iou >= pos_iou)[0]; neg = np.where(best_iou < neg_iou)[0]
    n_pos = min(len(pos), int(n_total * pos_frac)) if len(pos) else 0
    n_neg = n_total - n_pos
    pick_pos = rng.choice(pos, n_pos, replace=False) if n_pos else np.array([], int)
    pick_neg = rng.choice(neg, n_neg, replace=len(neg) < n_neg) if len(neg) else np.array([], int)
    idx = np.concatenate([pick_pos, pick_neg]).astype(int)
    labels = np.zeros(len(idx), np.int32)
    labels[:n_pos] = gt_cls[best[pick_pos]] + 1
    return props[idx], labels, best[idx]


# ---------------------------------------------------------------- post-processing
def postprocess(props, probs, deltas, w, h, score_thr=0.5, nms_thr=0.4):
    """Classifier outputs for each region -> final detections [(x1,y1,x2,y2,cls,score)]."""
    boxes = decode(props, deltas, w, h)
    dets = []
    for c in range(NUM_FG):
        sc = probs[:, c + 1]
        m = sc >= score_thr
        if not m.any():
            continue
        b, s = boxes[m], sc[m]
        for i in nms(b, s, nms_thr):
            dets.append((*map(float, b[i]), c, float(s[i])))
    return sorted(dets, key=lambda d: -d[5])


def draw(img, dets, gt=None):
    out = img.copy()
    if gt is not None:
        for (x1, y1, x2, y2), c in zip(*gt):
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 160, 0), 2)
            cv2.putText(out, f"GT {NAMES[int(c)]}", (int(x1), int(y2) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)
    for x1, y1, x2, y2, c, s in dets:
        col = COLORS[int(c)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(out, f"{NAMES[int(c)]} {s:.2f}", (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out


# ---------------------------------------------------------------- metrics
def average_precision(records, n_gt):
    """VOC all-point AP. records = [(score, is_true_positive)]."""
    if n_gt == 0:
        return float("nan")
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records]); fp = np.cumsum([not r[1] for r in records])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    rec = np.concatenate([[0], rec, [1]]); prec = np.concatenate([[0], prec, [0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


def evaluate_detector(detect_fn, split, limit, max_regions, iou_thr=0.5, score_thr=0.05):
    """Runs detect_fn(img_bgr, max_regions, score_thr) on a split; returns metrics dict."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    files = list_images(split, limit)
    recs = {c: [] for c in range(NUM_FG)}; n_gt = {c: 0 for c in range(NUM_FG)}
    y_true, y_pred, ious = [], [], []
    for k, path in enumerate(files):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gb, gc = read_labels(path, w, h)
        dets = detect_fn(img, max_regions, score_thr)
        used = set()
        for d in dets:                                   # sorted by score desc
            c = int(d[4]); cand = [i for i in range(len(gb)) if gc[i] == c and i not in used]
            if cand:
                ov = iou_matrix(np.array([d[:4]]), gb[cand])[0]
                j = int(ov.argmax())
                if ov[j] >= iou_thr:
                    used.add(cand[j]); recs[c].append((d[5], True)); ious.append(float(ov[j])); continue
            recs[c].append((d[5], False))
        for c in gc:
            n_gt[int(c)] += 1
        if len(gc):                                      # image-level: class of the top detection
            y_true.append(int(gc[0])); y_pred.append(int(dets[0][4]) if dets else -1)
        if (k + 1) % 50 == 0:
            print(f"  evaluated {k + 1}/{len(files)}")
    aps = {NAMES[c]: average_precision(recs[c], n_gt[c]) for c in range(NUM_FG)}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(NUM_FG)),
                                                 average="macro", zero_division=0)
    return {
        "split": split, "images": len(files),
        "AP50": {k: round(v, 4) for k, v in aps.items()},
        "mAP50": round(float(np.nanmean(list(aps.values()))), 4),
        "mean_IoU_of_matched": round(float(np.mean(ious)), 4) if ious else 0.0,
        "image_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "no_detection_rate": round(float(np.mean(np.array(y_pred) == -1)), 4),
        "precision_macro": round(float(p), 4), "recall_macro": round(float(r), 4), "f1_macro": round(float(f), 4),
    }


def save_history(history, out_dir, title):
    """Save history.json and (if matplotlib is installed) a loss/accuracy plot."""
    import json
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    acc_keys = [k for k in hist if k.endswith("accuracy") and not k.startswith("val_")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k in ("loss", "val_loss"):
        if k in hist: ax[0].plot(hist[k], label=k)
    ax[0].set_title(f"{title} - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    for k in acc_keys + ["val_" + k for k in acc_keys]:
        if k in hist: ax[1].plot(hist[k], label=k)
    ax[1].set_title(f"{title} - region accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=120); plt.close(fig)
