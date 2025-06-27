import streamlit as st
import cv2
import numpy as np

st.title("🖼️ Image Import Page")

st.write("""
Upload an image that you would like to work with. You can drag and drop the image file into the box below,
or click the **Browse Files** button to select one from your computer.
Supported formats: **JPG, JPEG, PNG**.
""")

# Check if image is already in session state or if a file is already uploaded
if "imported_image" in st.session_state:
    st.info("An image is already loaded in memory. You can upload a new one to replace it.")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    key="pre_file"
)

if uploaded_file:
    st.success("Image successfully uploaded.")

    # Read image as OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # BGR format

    st.image(image, caption="Uploaded Image", channels="BGR", use_container_width=False, width=500)

    # Save to session state
    st.session_state.imported_image = image

    st.info("The image is now stored in memory and ready for the next steps.")
        
elif "imported_image" in st.session_state:
    image = st.session_state.imported_image
    st.image(image, caption="Uploaded Image", channels="BGR", use_container_width=False, width=500)


else:
    st.warning("Please upload an image to continue.")
