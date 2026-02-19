import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="♻️ RecycleVision",
    page_icon="♻️",
    layout="wide"
)

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "models/best_model.h5"
HISTORY_PATH = "history/history.npy"
TEST_DIR = "dataset/test"

# Class Labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("♻️ RecycleVision Dashboard")
st.sidebar.write("Garbage Classification using Deep Learning")

option = st.sidebar.radio(
    "Select Feature",
    ["Upload & Predict", "Model Evaluation Report", "About Project"]
)

# ======================================================
# 1. Upload & Predict Section
# ======================================================
if option == "Upload & Predict":

    st.title("♻️ Waste Classification App")
    st.write("Upload a garbage image and predict its category.")

    uploaded_file = st.file_uploader(
        "Upload Waste Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", width=350)

        # Preprocess Image
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        with col2:
            st.success(f"✅ Predicted Category: **{predicted_class.upper()}**")
            st.info(f"Confidence Score: **{confidence*100:.2f}%**")

            # Top-3 Predictions
            st.subheader("🔝 Top 3 Predictions")
            top3 = np.argsort(prediction[0])[::-1][:3]

            for i in top3:
                st.write(f"➡️ {class_names[i]} : {prediction[0][i]*100:.2f}%")

# ======================================================
# 2. Model Evaluation Section
# ======================================================
elif option == "Model Evaluation Report":

    st.title("📊 Model Evaluation Metrics")

    # -------------------------------
    # Load Training History
    # -------------------------------
    if os.path.exists(HISTORY_PATH):
        history = np.load(HISTORY_PATH, allow_pickle=True).item()

        acc = history["accuracy"]
        val_acc = history["val_accuracy"]
        loss = history["loss"]
        val_loss = history["val_loss"]

        # Accuracy Curve
        st.subheader("✅ Accuracy Curve")

        fig1 = plt.figure()
        plt.plot(acc, label="Train Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        st.pyplot(fig1)

        # Loss Curve
        st.subheader("📉 Loss Curve")

        fig2 = plt.figure()
        plt.plot(loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        st.pyplot(fig2)

    else:
        st.warning("⚠️ Training history not found. Please train model first.")

    # -------------------------------
    # Confusion Matrix from Test Data
    # -------------------------------
    st.subheader("🧾 Real Confusion Matrix (Test Dataset)")

    if os.path.exists(TEST_DIR):

        test_datagen = ImageDataGenerator(rescale=1./255)

        test_data = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            shuffle=False
        )

        # Predictions
        y_pred = np.argmax(model.predict(test_data), axis=1)

        # Confusion Matrix
        cm = confusion_matrix(test_data.classes, y_pred)

        fig3 = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap="Blues")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig3)

        # Classification Report
        st.subheader("📌 Classification Report")

        report = classification_report(test_data.classes, y_pred, target_names=class_names)
        st.text(report)

    else:
        st.warning("⚠️ Test dataset folder not found!")

# ======================================================
# 3. About Section
# ======================================================
elif option == "About Project":

    st.title("📌 About RecycleVision Project")

    st.markdown("""
    ## ♻️ Project Title:
    **RecycleVision – Garbage Image Classification Using Deep Learning**

    ## 🎯 Objective:
    Classify waste images into 6 categories:

    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic
    - Trash

    ## 🏢 Business Use Cases:
    - Smart Recycling Bins  
    - Municipal Waste Sorting  
    - Environmental Monitoring  
    - Educational Tools  

    ## ⚙️ Technologies Used:
    - Python  
    - TensorFlow / Keras  
    - CNN + Transfer Learning (MobileNetV2)  
    - Streamlit Deployment  
    - Confusion Matrix & Metrics  

    ## 🚀 Final Outcome:
    A web app that predicts waste category with high accuracy (>85%).
    """)

    st.success("Developed as Capstone Project 🌍")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("♻️ **RecycleVision App | Deep Learning Capstone Project**")
