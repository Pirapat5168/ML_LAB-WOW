# ML-09 Object Detection I: R-CNN and Fast R-CNN on real vs fake faces

Detect a face in an image **and** classify it as `real` or `fake`. The data is the Lab 7
real/fake faces dataset (6,000 images per class) instead of BDD100K. Two detectors, R-CNN and
Fast R-CNN, share one dataset and one set of helpers (`common.py`), so their results can be compared.

## 1. Results

Evaluated on **200 test images** (`python evaluate.py --limit 200 --split test`, 150 region
proposals per image), trained on the full training split with GPU (RTX 3050 6 GB, WSL2).

| Metric | R-CNN | Fast R-CNN |
|---|---|---|
| AP50 - real | 0.4492 | **0.9267** |
| AP50 - fake | 0.8251 | **0.9474** |
| **mAP50** | 0.6371 | **0.9370** |
| Mean IoU of matched boxes | 0.8238 | **0.8835** |
| **Image accuracy** (real vs fake) | 0.4850 | **0.8450** |
| Precision (macro) | 0.2425 | **0.8470** |
| Recall (macro) | 0.5000 | **0.8438** |
| F1 (macro) | 0.3266 | **0.8443** |
| No-detection rate | 0.0 | 0.0 |

Raw numbers: `01-R-CNN/outputs/evaluation.json`, `02-Fast-R-CNN/outputs/evaluation.json`.
Training curves: `outputs/training_curves.png` in each model folder.

### What the numbers say

- **Fast R-CNN works.** About 84.5% of test images are classified correctly as real or fake, and its
  boxes overlap the reference face boxes well (mean IoU 0.88).
- **R-CNN localizes faces but does not separate real from fake.** Image accuracy is 0.485, which is
  chance level. Precision 0.2425 is half of 0.485 and recall is exactly 0.5, which is what you get
  when the top detection is the same class for (almost) every image. Its mAP50 of 0.64 mostly
  reflects finding the face box, not telling the classes apart, so image accuracy is the more
  honest number for this task.
- **R-CNN overfits.** In the last epoch shown, training region accuracy was 0.976 but validation
  region accuracy was 0.759 (validation loss 0.83 vs training loss 0.07).
- **Fast R-CNN generalizes.** Validation region accuracy 0.967 against 0.997 on training data.
  Region accuracy counts background regions too, so it is higher than image accuracy.

### Why Fast R-CNN is better here (likely, not tested)

R-CNN warps every region to 64x64 and judges it alone, which probably discards the fine texture
that separates fake faces from real ones. Fast R-CNN runs the whole 128x128 image through the CNN
once and pools each region from the shared feature map. The two models also differ in input size,
depth and batch size (see section 3), so the gap is not a controlled comparison of the two
methods. A fair test would train R-CNN with larger crops (for example 128x128).

## 2. Dataset (Lab 7 -> YOLO format)

Lab 7 only has class labels, no bounding boxes. `prepare_dataset.py` creates **pseudo ground-truth
boxes**: an OpenCV Haar face detector finds the face (10% margin); if it finds nothing, a centered
box covering 90% of the image is used. How many images used each case is saved in
`dataset/prepare_report.json`.

```bash
python prepare_dataset.py --data-dir "C:\Users\NITRO V15\Documents\ML\ML-07-CNN\ML-07-CNN\DATASET"
# quick subset: add  --max-per-class 1500
```

Result: `dataset/{train,val,test}/{images,labels}/` and `data.yaml`. Stratified 70/15/15 split
with seed 42: 8,400 train, 1,800 val and 1,800 test images. YOLO classes: `0 = real`, `1 = fake`.
Folder names under `--data-dir` must contain "real" or "fake".

## 3. How the two detectors differ

Both use the same region proposals (multi-scale sliding windows, a cheap stand-in for Selective
Search), the same labelling rule (IoU >= 0.5 with the face box = `real`/`fake`, IoU < 0.3 =
`background`), the same two outputs (class softmax + box refinement) and the same post-processing
(score threshold + per-class NMS).

| | R-CNN | Fast R-CNN |
|---|---|---|
| CNN input | each region cropped and warped to 64x64 | the whole image resized to 128x128 |
| CNN passes per image | one per region (~150) | **one**; regions are pooled from the shared feature map |
| Region feature | CNN output of the warped crop | 7x7 ROI pooling on the feature map |
| Backbone | 3 conv blocks (32-64-128) + Dense 256 | 3 conv blocks + 1 extra conv + ROI pool + Dense 256 |
| Training samples | 12 regions per image (~100,800 per epoch) | 32 regions per image, resampled every epoch |
| Batch size | 64 regions | 16 images (512 regions) |

## 4. Project structure

```text
ML-09-Object_Detection_I/
├── common.py               shared: dataset reading, IoU, proposals, NMS, box encode/decode, mAP
├── prepare_dataset.py      Lab 7 DATASET -> dataset/ (YOLO format)
├── 01-R-CNN/               train.py  evaluate.py  image_test.py  realtime_test.py  outputs/
├── 02-Fast-R-CNN/          train.py  evaluate.py  image_test.py  realtime_test.py  outputs/
├── requirements.txt
└── README.md
```

## 5. Setup and run

Python 3.12 is recommended (TensorFlow does not install on 3.13+ yet), and OpenCV must be 4.x
(`requirements.txt` pins `opencv-python<5`; OpenCV 5 removes the Haar face detector).

```bash
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Commands used for the results above (run inside each model folder):

```bash
python train.py --limit 0 --epochs 30           # --limit 0 = all training images
python evaluate.py --limit 200 --split test     # writes outputs/evaluation.json
python image_test.py                            # random test image -> outputs/image_result.jpg
python image_test.py --image path/to/face.jpg
python realtime_test.py --camera 0 --threshold 0.85 --regions 25   # webcam, press Q to quit
```

`train.py` saves `model.keras`, `history.json` and `training_curves.png` in `outputs/`.
Training uses early stopping (patience 6, best weights restored) and learning-rate reduction.
In result images: green = predicted real, red = predicted fake, orange = reference box.

## 6. Metrics (`evaluation.json`)

| Key | Meaning |
|---|---|
| `AP50`, `mAP50` | box IoU >= 0.5 **and** correct class (VOC all-point AP) |
| `mean_IoU_of_matched` | average overlap of correct detections with the reference boxes |
| `image_accuracy` | class of the highest-scoring detection vs the image's real/fake label |
| `no_detection_rate` | images where nothing passed the score threshold (counted as wrong) |
| `precision_macro`, `recall_macro`, `f1_macro` | image-level, macro-averaged over real/fake |

## 7. Limitations

- The reference boxes are pseudo-labels (Haar or the centered fallback), so IoU and AP measure
  agreement with those boxes, not with hand-drawn faces.
- The test set has 200 images, so image accuracy has roughly +/-5 points of uncertainty.
- Proposals are sliding windows, not Selective Search, and there is no region proposal network
  (that is Faster R-CNN).
- Both backbones are small CNNs trained from scratch; a pretrained backbone (as in Lab 8) is the
  obvious next step to raise accuracy, especially for R-CNN.

## 8. Troubleshooting (GPU / WSL2)

| Problem | Fix |
|---|---|
| GPU out of memory with `--limit 0` | Already handled: crops are stored as uint8 and sent to the GPU batch by batch. If it still happens, lower `--batch`. |
| `CropAndResize ... XLA_GPU_JIT` error | XLA cannot compile ROI pooling. `jit_compile=False` is set in `make_model` and `load_model` of Fast R-CNN. |
| `AttributeError: cv2 has no attribute CascadeClassifier` | Wrong OpenCV: `pip install "opencv-python<5"` (use `opencv-python-headless<5` in WSL). |
| `pip` says "externally managed environment" in WSL | Use `python -m pip ...` with the venv's Python. |

## 9. References

- [R-CNN](https://arxiv.org/abs/1311.2524)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)