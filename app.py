import streamlit as st
from PIL import Image
import numpy as np
import tempfile

# import predictors
from predictors.swin_s import predictor as swin_s_predictor, CLASS_NAMES as SWIN_S_CLASSES
from predictors.swin_b import predictor as swin_b_predictor, CLASS_NAMES as SWIN_B_CLASSES
from predictors.densenet import predictor as densenet_predictor, CLASSES as DENSENET_CLASSES
from predictors.efficientnet import predictor as efficientnet_predictor, CLASSES as EFFICIENTNET_CLASSES
from predictors.resnet import predictor as resnet_predictor, CLASSES as RESNET_CLASSES
from predictors.inception import predictor as inception_predictor, CLASSES as INCEPTION_CLASSES
from predictors.yolov11s import predictor as yolo_predictor, EXPECTED_CLASSES as YOLO_CLASSES
from ensemble import EnsembleModelClassifier

ENESEMBLE_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Nodule', 'Pneumothorax']


st.set_page_config(page_title="Chest X-ray Classification", layout="centered")

# --- TITLE ---
st.title("Chest X-ray Classification System")

# --- MODEL SELECTION ---
model_options = ["Swin-B", "DenseNet121", "EfficientNetB0", "ResNet50", "InceptionV3", "YOLOv11s", "Ensemble"]
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

    # Save to a temporary file if image_path is needed
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name  
    # st.write(f"Image saved to: {image_path}")
    # True label selection     
    true_labels = {
        "Atelectasis": "Atelectasis",
        "Cardiomegaly": "Cardiomegaly",
        "Effusion": "Effusion",
        "Nodule": "Nodule",
        "Pneumothorax": "Pneumothorax"
    }   
    true_label = st.selectbox("True label (optional):", options=true_labels, index=0)
    st.markdown(f"### ✅ True Label: `{true_label}`")


    # Classification button
    if st.button("Classify"):
        with st.spinner("Running inference…"):
            img_arr = np.array(image)

            # Select correct model
            if model_choice == "Swin-S":
                preds = swin_s_predictor(img_arr)
                classes = SWIN_S_CLASSES
            elif model_choice == "Swin-B":
                preds = swin_b_predictor(img_arr)
                classes = SWIN_B_CLASSES    
            elif model_choice == "DenseNet121":
                preds = densenet_predictor(img_arr)
                classes = DENSENET_CLASSES
            elif model_choice == "EfficientNetB0":
                preds = efficientnet_predictor(img_arr)
                classes = EFFICIENTNET_CLASSES
            elif model_choice == "ResNet50":
                preds = resnet_predictor(img_arr)
                classes = RESNET_CLASSES
            elif model_choice == "InceptionV3":
                preds = inception_predictor(img_arr)
                classes = INCEPTION_CLASSES
            elif model_choice == "YOLOv11s":
                preds = yolo_predictor(img_arr)
                classes = YOLO_CLASSES
            elif model_choice == "Ensemble":
                ensemble = EnsembleModelClassifier()
                preds, predicted_class = ensemble.predict(image_path, method="voting")
                classes = ENESEMBLE_CLASSES

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
