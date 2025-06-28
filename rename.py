# temp.py — Smooth & Faster ASCII Art app (image + webcam)
# Run with:  streamlit run temp.py
# -------------------------------------------------------------
"""
Streamlit app that converts uploaded **images** and **live webcam frames**
into colored **or** grayscale ASCII art.

Changes in this version
———————————————
1. **Faster grayscale path** ─ draws one whole row of characters at once,
   instead of per-character drawing. This makes the webcam usable even on
   modest CPUs when colour output is OFF.
2. **Frame-skip slider** ─ process every *n*-th webcam frame (default: 2) to
   trade quality for speed on slower machines.
3. Still no external font files; depends only on
   `streamlit opencv-python pillow numpy streamlit-webrtc av`.

If performance is still an issue, uncheck the **Colour ASCII** box and/or
increase the **Frame gap** slider.
"""

from __future__ import annotations
import io, time
from typing import Tuple

import cv2, numpy as np, streamlit as st
from PIL import Image, ImageDraw, ImageFont

# live video (smooth) ---------------------------------------------
import av  # streamlit‑webrtc backend
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Pillow ≥10 font compat -----------------------------------------
if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def _getsize(self, txt, *a, **k):
        x0, y0, x1, y1 = self.getbbox(txt, *a, **k)
        return x1 - x0, y1 - y0
    ImageFont.FreeTypeFont.getsize = _getsize  # type: ignore

GRADIENT = " .:-=+*#%@"

# -------------------------- helpers -----------------------------

def preprocess(gray: np.ndarray, mode: str, lo: int, hi: int, rad: int) -> np.ndarray:
    if mode == "Sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        gray = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif mode == "Canny":
        gray = cv2.Canny(gray, lo, hi)
    elif mode == "Erode":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1,) * 2)
        gray = cv2.erode(gray, k)
    elif mode == "Dilate":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1,) * 2)
        gray = cv2.dilate(gray, k)
    return gray


def _ascii_row(gray_row: np.ndarray) -> str:
    # vectorised mapping of a row of pixels to ASCII chars
    idx = (gray_row.astype(np.float32) / 255 * (len(GRADIENT) - 1)).astype(np.int32)
    return "".join([GRADIENT[i] for i in idx])


def to_ascii(pil_img: Image.Image, cfg: Tuple[int, bool, str, str, int, int, int]) -> Image.Image:
    cols, colour, bg, mode, lo, hi, rad = cfg
    font = ImageFont.load_default()
    cw, ch = font.getsize("A")
    ca = ch / cw
    W, H = pil_img.size
    rows = max(1, int(H / W * cols * ca))

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = preprocess(gray, mode, lo, hi, rad)
    gray_s = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

    if colour:
        bgr_s = cv2.resize(bgr, (cols, rows), interpolation=cv2.INTER_AREA)

    canvas = Image.new("RGB", (cols * cw, rows * ch), bg)
    d = ImageDraw.Draw(canvas)

    if not colour:  # fast path: one draw per row
        fg = (255, 255, 255) if bg == "black" else (0, 0, 0)
        for y in range(rows):
            line = _ascii_row(gray_s[y])
            d.text((0, y * ch), line, font=font, fill=fg)
    else:  # colour path: per‑char draw (slower)
        for y in range(rows):
            for x in range(cols):
                lum = gray_s[y, x] / 255
                char = GRADIENT[int(lum * (len(GRADIENT) - 1))]
                fill = tuple(int(v) for v in bgr_s[y, x][::-1])
                d.text((x * cw, y * ch), char, font=font, fill=fill)
    return canvas

# ---------------- VideoProcessor for webcam ----------------------

class AsciiProcessor(VideoProcessorBase):
    def __init__(self, cfg, gap):
        self.cfg_fn = cfg
        self.gap = gap
        self.counter = 0

    def recv(self, frame):
        self.counter += 1
        if self.counter % self.gap != 0:
            # skip processing: return original frame to keep FPS
            return frame
        img = frame.to_ndarray(format="bgr24")
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out = to_ascii(pil, self.cfg_fn())
        return av.VideoFrame.from_ndarray(np.array(out), format="rgb24")

# ------------------------ Streamlit UI --------------------------

st.set_page_config(page_title="ASCII Art", page_icon="🅰️")
st.title("🅰️ ASCII Art Converter")

st.sidebar.header("Pre‑processing")
filt = st.sidebar.selectbox("Filter", ["None", "Sobel", "Canny", "Erode", "Dilate"])
lo = st.sidebar.slider("Canny low", 0, 255, 100) if filt == "Canny" else 100
hi = st.sidebar.slider("Canny high", 0, 255, 200) if filt == "Canny" else 200
rad = st.sidebar.slider("Morph radius", 1, 10, 1) if filt in ("Erode", "Dilate") else 1

st.sidebar.header("ASCII options")
cols = st.sidebar.slider("Characters across", 50, 400, 160, 10)
colour = st.sidebar.checkbox("Colour ASCII", True)
bg = st.sidebar.radio("Background", ["black", "white"], horizontal=True)

st.sidebar.header("Webcam speed")
frame_gap = st.sidebar.slider("Frame gap (process every n‑th frame)", 1, 5, 2)

cfg_tuple = (cols, colour, bg, filt, lo, hi, rad)

# -------------------- Image tab --------------------------------

tab_img, tab_cam = st.tabs(["Image", "Webcam"])

with tab_img:
    upl = st.file_uploader("Upload image", ["png", "jpg", "jpeg", "webp"])
    if upl:
        src = Image.open(upl).convert("RGB")
        st.image(src, caption="Original", use_container_width=True)
        with st.spinner("Rendering ASCII …"):
            out = to_ascii(src, cfg_tuple)
        st.image(out, caption="ASCII", use_container_width=True)
        buf = io.BytesIO(); out.save(buf, format="PNG")
        st.download_button("Download PNG", buf.getvalue(), "ascii.png", "image/png")

# -------------------- Webcam tab (webrtc) ----------------------

with tab_cam:
    st.write("Enable your webcam to see live ASCII video.")

    def cfg():  # closure so VideoProcessor sees live widget values
        return (cols, colour, bg, filt, lo, hi, rad)

    webrtc_streamer(
        key="ascii-cam",
        video_processor_factory=lambda: AsciiProcessor(cfg, frame_gap),
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
    )
