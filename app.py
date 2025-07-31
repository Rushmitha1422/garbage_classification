import streamlit as st
import cv2
import numpy as np
import joblib

# Set up Streamlit page config
st.set_page_config(page_title="Garbage Classifier", page_icon="♻️", layout="centered")

# Load the trained SVM model
model = joblib.load("svm_model.pkl")
IMAGE_SIZE = (64, 64)
CLASSES = model.classes_

# App Title and Description
st.markdown("<h1 style='text-align: center; color: green;'>♻️ Garbage Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of waste and let the AI classify it into one of six categories.</p>", unsafe_allow_html=True)
st.markdown("---")

# Info for users
st.info("Supported categories: **cardboard, glass, metal, paper, plastic, trash**")

# Add warning about random inputs
st.warning("⚠️ This model is trained only on garbage images. Uploading random photos (like people, scenery, etc.) may result in incorrect predictions.")

# Feature extraction function
def extract_features_from_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# File uploader
uploaded_file = st.file_uploader("📁 Upload a garbage image...", type=["jpg", "jpeg", "png"])

# If user uploads an image
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(opencv_image, channels="BGR", caption="📷 Uploaded Image", use_column_width=True)

    try:
        # Extract features and predict
        features = extract_features_from_image(opencv_image)
        prediction = model.predict([features])[0]

        # Show prediction
        st.markdown(
            f"<h3 style='text-align: center; color: royalblue;'>🧠 Prediction: <span style='color: green;'>{prediction}</span></h3>",
            unsafe_allow_html=True
        )
    except:
        st.error("⚠️ Oops! Couldn't process this image. Please upload a clear, recognizable garbage image.")

#  Footer (optional)
st.markdown("---")