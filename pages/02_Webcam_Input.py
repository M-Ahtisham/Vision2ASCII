import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

st.title("📷 Webcam Input")

st.write("""
This page allows you to capture video from your webcam directly in the browser.
You can also select different available video sources if supported by your system.
""")

# Frame processing class
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        st.session_state.latest_webcam_frame = img  # Store frame in session state
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Show webcam stream
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor
)

# Show status info
if "latest_webcam_frame" in st.session_state:
    st.success("Webcam is running. Latest frame is stored in memory.")
else:
    st.info("Start the webcam to capture frames.")
