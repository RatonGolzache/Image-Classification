import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans


def extract_color_histogram(image: np.ndarray, bins=(4, 4, 4)) -> np.ndarray:
    """
    Computes a simple RGB histogram.

    adapted from https://tuwel.tuwien.ac.at/mod/resource/view.php?id=2758047
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # OpenCV expects BGR
    image_bgr = image[:, :, ::-1]

    hist = cv2.calcHist(
        [image_bgr],
        [0, 1, 2],
        None,
        list(bins),
        [0, 256, 0, 256, 0, 256],
    )

    return hist.flatten()


def extract_sift_descriptors(image: np.ndarray):
    """Extract SIFT descriptors for one image."""

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def collect_sift_descriptors(images):
    """Extract SIFT descriptors for all images."""

    descriptor_list = []

    for img in images:
        des = extract_sift_descriptors(img)
        if des is not None:
            descriptor_list.append(des)
    return descriptor_list


def build_bovw_vocabulary(descriptor_list, vocab_size=200, random_state=19):
    """Fit MiniBatchKMeans to build the visual vocabulary."""

    all_descriptors = np.vstack([d for d in descriptor_list if d is not None])

    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=1000,
        random_state=random_state
    )
    kmeans.fit(all_descriptors)
    return kmeans


def compute_bovw_histogram(descriptors, kmeans) -> np.ndarray:
    """Convert descriptors to a normalized BoVW histogram."""

    vocab_size = kmeans.n_clusters
    histogram = np.zeros(vocab_size, dtype=np.float32)

    if descriptors is None:
        return histogram

    word_indices = kmeans.predict(descriptors)
    for idx in word_indices:
        histogram[idx] += 1.0

    # L2 normalization
    histogram /= (np.linalg.norm(histogram) + 1e-6)
    return histogram


def extract_sift_bovw_features(images, vocab_size=200, random_state=19):
    """
    Full pipeline:
    1) extract descriptors
    2) build vocab
    3) encode histograms for each image
    """
    descriptor_list = collect_sift_descriptors(images)
    kmeans = build_bovw_vocabulary(descriptor_list, vocab_size=vocab_size, random_state=random_state)

    X_bovw = []
    for img in images:
        des = extract_sift_descriptors(img)
        hist = compute_bovw_histogram(des, kmeans)
        X_bovw.append(hist)

    return np.array(X_bovw), kmeans
