import cv2
import numpy as np
from skimage.feature import hog

# HOG parameters (kept in one place so data_loader / main / test_svm
# all agree on the same feature shape)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def preprocess_image(image, img_size=128):
    """Resize one image and convert it to grayscale. None if unusable."""

    if image is None or image.size == 0:
        return None

    # Convert BGR to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image (INTER_AREA is the right filter for shrinking)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def extract_hog(image):
    """(h, w) grayscale uint8 -> 1D HOG feature vector (float)."""

    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def to_features(images):
    """(n, h, w) grayscale uint8 -> (n, hog_dim) float32 HOG feature matrix."""

    features = [extract_hog(img) for img in images]
    return np.stack(features).astype(np.float32)


def preprocess_images(images, img_size=128):
    """Raw image list -> HOG feature matrix (for small/ad-hoc batches)."""

    processed = [preprocess_image(img, img_size) for img in images]
    processed = [img for img in processed if img is not None]

    return to_features(np.stack(processed))
