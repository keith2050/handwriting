"""Page preprocessing: grayscale conversion, binarization, deskew."""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale. No-op if already 2-D."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Binarize a grayscale image using adaptive Gaussian thresholding.

    Returns an image where ink pixels = 255 on a black background (inverted).
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    return bw


def deskew(img_bgr: np.ndarray) -> np.ndarray:
    """
    Deskew a page image by estimating the dominant text angle.

    Uses the minimum-area bounding rectangle of all foreground pixels to
    estimate the rotation angle, then applies a rotation to correct it.
    Returns the rotated BGR image.
    """
    gray = to_gray(img_bgr)
    bw = binarize(gray)

    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return img_bgr

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # cv2.minAreaRect returns angle in [-90, 0); adjust for text orientation
    if angle < -45:
        angle = 90 + angle

    (h, w) = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_page(img_bgr: np.ndarray) -> dict:
    """
    Full preprocessing pipeline for a single page image.

    Returns a dict with keys:
      - ``img_bgr``: deskewed BGR image
      - ``gray``:    grayscale image
      - ``bw``:      binarized image (ink=255, background=0)
    """
    img_bgr = deskew(img_bgr)
    gray = to_gray(img_bgr)
    bw = binarize(gray)
    return {"img_bgr": img_bgr, "gray": gray, "bw": bw}
