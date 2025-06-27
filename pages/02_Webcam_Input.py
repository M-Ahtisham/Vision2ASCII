import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import streamlit.components.v1 as components
import av
import cv2
import numpy as np




##########################################################


from enum import Enum


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

# def apply_seam_carving(img: np.ndarray, 
#                       new_width: Optional[int] = None, 
#                       new_height: Optional[int] = None,
#                       energy_mode: str = 'backward',
#                       order: str = 'width-first') -> np.ndarray:
#     """
#     Content-aware image resizing using seam carving.
    
#     Args:
#         img: Input image (BGR format)
#         new_width: Target width (None to keep original)
#         new_height: Target height (None to keep original)
#         energy_mode: 'backward' or 'forward'
#         order: 'width-first' or 'height-first'
    
#     Returns:
#         Resized image
#     """
#     if new_width is None and new_height is None:
#         return img
    
#     if img.ndim == 3:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     else:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
#     if new_width is None:
#         new_width = img.shape[1]
#     if new_height is None:
#         new_height = img.shape[0]
    
#     resized = seam_carving.resize(
#         img_rgb, 
#         (new_width, new_height),
#         energy_mode=energy_mode,
#         order=order
#     )

#     return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


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






##########################################################





















st.set_page_config(layout="wide")
st.title("📷 Real-Time Video Processing")

# Initialize session state for effect parameters
if 'effects' not in st.session_state:
    st.session_state.effects = {
        'grayscale': False,
        'downscale': False,
        'downscale_percent': 50,
        'blur': False,
        'blur_type': "Mean (Box)",
        'blur_ksize': 3,
        'edge_detection': None,
        'edge_type': "Sobel",
        'sobel_ksize': 3,
        'canny_thresh1': 100,
        'canny_thresh2': 200,
        'threshold': False,
        'thresh_value': 127,
        'thresh_type': "Binary",
        'morphology': False,
        'morph_op': MorphOperation.ERODE,
        'morph_ksize': 3
    }

class VideoProcessor(VideoProcessorBase):
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Use try/except to avoid session state errors in worker threads
        try:
            st.session_state.original_frame = img
        except Exception:
            pass

        try:
            processed = self.apply_effects(img)
        except Exception:
            processed = img

        try:
            st.session_state.processed_frame = processed
        except Exception:
            pass

        return av.VideoFrame.from_ndarray(processed, format="bgr24")
    
    
    def apply_effects(self, img: np.ndarray) -> np.ndarray:
        effects = st.session_state.effects
        processed = img.copy()
        
        # Apply effects in logical order
        if effects['downscale']:
            processed = apply_downscale(processed, effects['downscale_percent'])
        
        if effects['grayscale']:
            processed = apply_grayscale(processed)
            if processed.ndim == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        if effects['blur']:
            processed = apply_blur(
                processed, 
                effects['blur_type'], 
                effects['blur_ksize']
            )
        
        if effects['edge_detection']:
            if effects['edge_type'] == "Sobel":
                edge_img = apply_sobel_edge_detection(
                    processed, 
                    ksize=effects['sobel_ksize']
                )
            else:
                edge_img = apply_canny_edge_detection(
                    processed,
                    threshold1=effects['canny_thresh1'],
                    threshold2=effects['canny_thresh2']
                )
            processed = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        
        if effects['threshold']:
            thresh_img = apply_threshold(
                processed,
                thresh=effects['thresh_value'],
                type=effects['thresh_type']
            )
            processed = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        
        if effects['morphology']:
            morph_img = apply_morphology(
                processed,
                operation=effects['morph_op'],
                ksize=effects['morph_ksize']
            )
            processed = cv2.cvtColor(morph_img, cv2.COLOR_GRAY2BGR)
        
        return processed

# Sidebar controls
with st.sidebar:
    st.header("Processing Controls")
    
    st.session_state.effects['grayscale'] = st.checkbox("Grayscale")
    
    st.session_state.effects['downscale'] = st.checkbox("Downscale")
    if st.session_state.effects['downscale']:
        st.session_state.effects['downscale_percent'] = st.slider(
            "Downscale Percentage", 10, 90, 50, 5
        )
    
    st.session_state.effects['blur'] = st.checkbox("Blur")
    if st.session_state.effects['blur']:
        st.session_state.effects['blur_type'] = st.selectbox(
            "Blur Type", ["Mean (Box)", "Median", "Gaussian"]
        )
        st.session_state.effects['blur_ksize'] = st.slider(
            "Blur Kernel Size", 1, 15, 3, 2
        )
    
    st.session_state.effects['edge_detection'] = st.checkbox("Edge Detection")
    if st.session_state.effects['edge_detection']:
        st.session_state.effects['edge_type'] = st.selectbox(
            "Edge Type", ["Sobel", "Canny"]
        )
        if st.session_state.effects['edge_type'] == "Sobel":
            st.session_state.effects['sobel_ksize'] = st.slider(
                "Sobel Kernel Size", 1, 7, 3, 2
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.effects['canny_thresh1'] = st.slider(
                    "Canny Threshold 1", 0, 255, 100
                )
            with col2:
                st.session_state.effects['canny_thresh2'] = st.slider(
                    "Canny Threshold 2", 0, 255, 200
                )
    
    st.session_state.effects['threshold'] = st.checkbox("Threshold")
    if st.session_state.effects['threshold']:
        st.session_state.effects['thresh_value'] = st.slider(
            "Threshold Value", 0, 255, 127
        )
        st.session_state.effects['thresh_type'] = st.selectbox(
            "Threshold Type", ["Binary", "Binary Inv", "Trunc", "To Zero", "To Zero Inv", "Otsu", "Triangle"]
        )
    
    st.session_state.effects['morphology'] = st.checkbox("Morphology")
    if st.session_state.effects['morphology']:
        st.session_state.effects['morph_op'] = st.selectbox(
            "Operation", list(MorphOperation)
        )
        st.session_state.effects['morph_ksize'] = st.slider(
            "Kernel Size", 1, 15, 3, 2
        )

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Video")
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Processed Video")
    components.iframe("http://localhost:8000/canvas.html", height=550, scrolling=True)


# Display current frame information
if webrtc_ctx.state.playing and 'original_frame' in st.session_state:
    st.write(f"Frame size: {st.session_state.original_frame.shape[1]}x{st.session_state.original_frame.shape[0]}")