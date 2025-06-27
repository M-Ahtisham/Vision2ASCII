import streamlit as st
import cv2
import numpy as np

def apply_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def apply_blur(img: np.ndarray, blur_type: str, ksize: int) -> np.ndarray:
    """Apply blur to an image based on the specified type and kernel size."""
    if blur_type == "Mean (Box)":
        return cv2.blur(img, (ksize, ksize))
    elif blur_type == "Median":
        return cv2.medianBlur(img, ksize)
    return img

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to enhance image contrast."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    return cv2.equalizeHist(img)

def apply_sobel_edge_detection(img: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detection to highlight edges."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sobelx**2 + sobely**2)
    return np.clip(edge, 0, 255).astype(np.uint8)

def apply_threshold(img: np.ndarray, thresh: int) -> np.ndarray:
    """Apply binary thresholding to an image."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary

def apply_morphology(img: np.ndarray, operation: str, ksize: int) -> np.ndarray:
    """Apply morphological operation (erode or dilate) to a binary image."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    kernel = np.ones((ksize, ksize), np.uint8)
    if operation == "Erode":
        return cv2.erode(img, kernel, iterations=1)
    elif operation == "Dilate":
        return cv2.dilate(img, kernel, iterations=1)
    return img
