import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Constants
IMG_SIZE = 224
CLASSES = ['0', 'R', 'MR', 'MRMS', 'MS', 'S']

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("best_model.keras")

model = load_trained_model()

st.title("🌾 Yellow Rust Classifier")
st.write("Upload a wheat leaf image to predict yellow rust stage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)

    st.success(f"**Prediction:** {CLASSES[class_idx]} (Confidence: {confidence:.2%})")
