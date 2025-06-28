import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import streamlit.components.v1 as components
import av
import cv2
import numpy as np



# st.markdown(
#     """
#     <style>
#         /* Do not hide sidebar, just adjust main content */
#         .main .block-container {
#             padding-top: 0rem;
#             padding-right: 0rem;
#             padding-left: 0rem;
#             padding-bottom: 0rem;
#             max-width: 100vw;
#         }
#         html, body, iframe {
#             height: 100%;
#             width: 100%;
#             margin: 0;
#             padding: 0;
#             overflow: hidden;
#         }
#         #fullframe {
#             position: absolute;
#             top: 0; left: 0;
#             width: calc(100vw - 21rem); /* leave space for sidebar (default ~21rem) */
#             height: 100vh;
#             border: none;
#             z-index: 9999;
#         }
#         /* Shift iframe right to account for sidebar width */
#         [data-testid="stSidebar"] ~ .main #fullframe {
#             left: 21rem;
#         }
#     </style>
#     <iframe id="fullframe" src="http://localhost:8000/processing.html" scrolling="yes"></iframe>
#     """,
#     unsafe_allow_html=True
# )


components.iframe("http://localhost:8000/processing.html", height=900, width=1400, scrolling=True)