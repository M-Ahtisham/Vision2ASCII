import streamlit as st
import os
import pandas as pd

# Configuration settings
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Disable auto-reload for better performance

# Page configuration
st.set_page_config(
    page_title="Vision2ASCII",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available pages in the application
PAGES = [
    "Preprocess Image",
    "Image to ASCII Art",
    "Webcam to ASCII Art"
]

# # Create tabs for navigation
# tab1, tab2, tab3 = st.tabs(PAGES)

st.title("Vision2ASCII")
# st.image("imgs/logo.png")

ASCII_GRADIENTS = {
    "Single block": "█",
    "Solid block (instead of text)": "▓▒░ ",
    "Minimalist": ".:-=+*#%@",
    "Medium set": "@%#*+=-:. ",
    "Longer set": "@#S%?*+;:,.",
    "Full set": "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "Max": "$@B%8WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "Alphabetic": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "Alphanumeric": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "Arrow": "←↑→↓↖↗↘↙",
    "Extended High": "█▓▒░@#%xo+:-. ",
    "Math Symbols": "+-*/=<>^∑√π∞≈≠",
    "Numerical": "0123456789"
}

if "ASCII_GRADIENTS" not in st.session_state:
    st.session_state["ASCII_GRADIENTS"] = ASCII_GRADIENTS