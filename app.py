import streamlit as st
from PIL import Image
import numpy as np

# import predictors
from predictors.swin import predictor as swin_predictor, CLASS_NAMES as SWIN_CLASSES
from predictors.densenet import predictor as densenet_predictor, CLASSES as DENSENET_CLASSES
from predictors.efficientnet import predictor as efficientnet_predictor, CLASSES as EFFICIENTNET_CLASSES
from predictors.resnet import predictor as resnet_predictor, CLASSES as RESNET_CLASSES

st.set_page_config(page_title="Chest X-ray Classification", layout="centered")

# --- TITLE ---
st.title("Chest X-ray Classification System")

# --- MODEL SELECTION ---
model_options = ["Swin-B", "DenseNet", "EfficientNet", "ResNet"]
model_choice = st.selectbox("Choose a model for classification", model_options)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image…",
    type=["jpg", "jpeg", "png"],
    help="Limit 20MB per file · JPG, PNG, JPEG"
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Classification button
    if st.button("Classify"):
        with st.spinner("Running inference…"):
            img_arr = np.array(image)

            # Select correct model
            if model_choice == "Swin-B":
                preds = swin_predictor(img_arr)
                classes = SWIN_CLASSES
            elif model_choice == "DenseNet":
                preds = densenet_predictor(img_arr)
                classes = DENSENET_CLASSES
            elif model_choice == "EfficientNet":
                preds = efficientnet_predictor(img_arr)
                classes = EFFICIENTNET_CLASSES
            elif model_choice == "ResNet":
                preds = resnet_predictor(img_arr)
                classes = RESNET_CLASSES
            else:
                st.error("Model not implemented.")
                st.stop()

        # Show results
        st.subheader("Prediction Scores")
        st.table({
            "Condition": list(preds.keys()),
            "Probability": list(preds.values())
        })

        # as a bar chart
        st.bar_chart(data={k: v for k, v in preds.items()})
