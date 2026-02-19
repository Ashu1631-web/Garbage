import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model("../models/best_model.h5")

class_names = ['cardboard', 'glass', 'metal',
               'paper', 'plastic', 'trash']

st.title("♻️ RecycleVision – Garbage Classification App")

st.write("Upload an image and get waste category prediction.")

uploaded_file = st.file_uploader("Upload Waste Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess Image
    img = image.resize((224,224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Waste Type: {predicted_class.upper()}")
    st.info(f"Confidence Score: {confidence*100:.2f}%")
