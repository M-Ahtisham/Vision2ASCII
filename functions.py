import streamlit as st
import cv2
import numpy as np
from typing import Tuple, Optional
import sys
from enum import Enum

# For seam carving (content-aware resizing)
try:
    import seam_carving
except ImportError:
    sys.exit("The 'seam_carving' package is required. Please install it using 'pip install seam-carving'")


class MorphOperation(Enum):
    ERODE = "Erode"
    DILATE = "Dilate"
    OPEN = "Open"
    CLOSE = "Close"
    GRADIENT = "Gradient"
    TOPHAT = "Top Hat"
    BLACKHAT = "Black Hat"

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
    elif blur_type == "Gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to enhance image contrast."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    return cv2.equalizeHist(img)

def apply_sobel_edge_detection(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Sobel edge detection to highlight edges."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    edge = np.sqrt(sobelx**2 + sobely**2)
    return np.clip(edge, 0, 255).astype(np.uint8)

def apply_canny_edge_detection(img: np.ndarray, 
                             threshold1: int = 100, 
                             threshold2: int = 200) -> np.ndarray:
    """Apply Canny edge detection."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    return cv2.Canny(img, threshold1, threshold2)

def apply_threshold(img: np.ndarray, 
                   thresh: int = 127, 
                   maxval: int = 255, 
                   type: str = "Binary") -> np.ndarray:
    """Apply thresholding to an image with different types."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    
    thresh_type = {
        "Binary": cv2.THRESH_BINARY,
        "Binary Inv": cv2.THRESH_BINARY_INV,
        "Trunc": cv2.THRESH_TRUNC,
        "To Zero": cv2.THRESH_TOZERO,
        "To Zero Inv": cv2.THRESH_TOZERO_INV,
        "Otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        "Triangle": cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    }.get(type, cv2.THRESH_BINARY)
    
    _, binary = cv2.threshold(img, thresh, maxval, thresh_type)
    return binary

def apply_morphology(img: np.ndarray, 
                    operation: MorphOperation, 
                    ksize: int = 3,
                    iterations: int = 1) -> np.ndarray:
    """Apply morphological operation to an image."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    
    kernel = np.ones((ksize, ksize), np.uint8)
    
    if operation == MorphOperation.ERODE:
        return cv2.erode(img, kernel, iterations=iterations)
    elif operation == MorphOperation.DILATE:
        return cv2.dilate(img, kernel, iterations=iterations)
    elif operation == MorphOperation.OPEN:
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == MorphOperation.CLOSE:
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif operation == MorphOperation.GRADIENT:
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    elif operation == MorphOperation.TOPHAT:
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif operation == MorphOperation.BLACKHAT:
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return img

def apply_seam_carving(img: np.ndarray, 
                      new_width: Optional[int] = None, 
                      new_height: Optional[int] = None,
                      energy_mode: str = 'backward',
                      order: str = 'width-first') -> np.ndarray:
    """
    Content-aware image resizing using seam carving.
    
    Args:
        img: Input image (BGR format)
        new_width: Target width (None to keep original)
        new_height: Target height (None to keep original)
        energy_mode: 'backward' or 'forward'
        order: 'width-first' or 'height-first'
    
    Returns:
        Resized image
    """
    if new_width is None and new_height is None:
        return img
    
    if img.ndim == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    if new_width is None:
        new_width = img.shape[1]
    if new_height is None:
        new_height = img.shape[0]
    
    resized = seam_carving.resize(
        img_rgb, 
        (new_width, new_height),
        energy_mode=energy_mode,
        order=order
    )
    
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

def apply_downscale(img: np.ndarray, 
                   scale_percent: float = 50.0, 
                   interpolation: str = "area") -> np.ndarray:
    """
    Downscale an image by percentage.
    
    Args:
        img: Input image
        scale_percent: Scaling percentage (0-100)
        interpolation: One of ['area', 'linear', 'cubic', 'lanczos']
    
    Returns:
        Downscaled image
    """
    interp_methods = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    return cv2.resize(img, dim, interpolation=interp_methods.get(interpolation, cv2.INTER_AREA))

def apply_laplacian(img: np.ndarray) -> np.ndarray:
    """Apply Laplacian edge detection."""
    if img.ndim == 3:
        img = apply_grayscale(img)
    return cv2.Laplacian(img, cv2.CV_64F, ksize=3)

def apply_adaptive_threshold(img: np.ndarray,
                           method: str = "Gaussian",
                           block_size: int = 11,
                           C: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding.
    
    Args:
        method: "Mean" or "Gaussian"
        block_size: Size of a pixel neighborhood (odd number)
        C: Constant subtracted from the mean
    """
    if img.ndim == 3:
        img = apply_grayscale(img)
    
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY, block_size, C)

def apply_contour_detection(img: np.ndarray, 
                           mode: str = "external",
                           method: str = "simple") -> np.ndarray:
    """
    Find and draw contours on the image.
    
    Args:
        mode: "external", "list", "tree", etc.
        method: "none", "simple", "tcos", "l1"
    """
    if img.ndim == 3:
        img = apply_grayscale(img)
    
    mode_map = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE
    }
    
    method_map = {
        "none": cv2.CHAIN_APPROX_NONE,
        "simple": cv2.CHAIN_APPROX_SIMPLE,
        "tcos": cv2.CHAIN_APPROX_TCOS_L1,
        "l1": cv2.CHAIN_APPROX_TCOS_L1
    }
    
    contours, _ = cv2.findContours(
        img,
        mode_map.get(mode, cv2.RETR_EXTERNAL),
        method_map.get(method, cv2.CHAIN_APPROX_SIMPLE)
    )
    
    # Draw contours on blank image
    output = np.zeros_like(img)
    cv2.drawContours(output, contours, -1, (255), 1)
    return output