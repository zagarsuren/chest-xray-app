# 🩺 Chest X-ray Classification System

A Streamlit-powered web application that classifies chest X-ray images using state-of-the-art deep learning models. The system supports multiple backbone architectures and ensemble predictions for robust diagnosis across five major thoracic conditions.

## 🔍 Overview

This app enables users (medical professionals, researchers, or students) to:

* Upload a chest X-ray image (JPG, PNG)
* Choose from various models such as Swin Transformers, DenseNet, ResNet, EfficientNet, Inception, YOLOv11, or a custom Ensemble
* View prediction scores for conditions like **Atelectasis**, **Cardiomegaly**, **Effusion**, **Nodule**, and **Pneumothorax**
* Optionally specify the true label for performance tracking
* Visualize results via a table and bar chart

## 🧠 Supported Models

| Model          | Type              | Highlights                                      |
| -------------- | ----------------- | ----------------------------------------------- |
| Swin-B         | Transformer-based | Global Attention, Hierarchical design |
| DenseNet121    | CNN               | Efficient deep feature propagation              |
| EfficientNetB0 | CNN               | Scalable and resource-efficient                 |
| ResNet50       | CNN               | Deep residual learning                          |
| InceptionV3    | CNN               | Multi-scale convolutional feature maps          |
| YOLOv11s       | CNN (Detector)    | More detailed feature extraction |
| Ensemble       | Hybrid            | Combines outputs of all models via voting       |

## 📁 Folder Structure

```
project_root/
│
├── app.py                        # Streamlit app
├── ensemble.py                   # Ensemble logic
├── predictors/
│   ├── swin_s.py
│   ├── swin_b.py
│   ├── densenet.py
│   ├── efficientnet.py
│   ├── resnet.py
│   ├── inception.py
│   └── yolov11s.py
├── requirements.txt              # Dependencies
└── README.md                     
```

## ⚙️ Features

* 🖼 Upload and visualize chest X-ray images
* 🤖 Multi-model prediction
* 📊 Class probability scores
* 🧪 True label selection (for evaluation)
* 📉 Result charting for easy interpretation

## 🏥 Target Conditions

* Atelectasis
* Cardiomegaly
* Effusion
* Nodule
* Pneumothorax

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/zagarsuren/chest-xray-app.git
cd chest-xray-classification-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

## 🧪 Example Use

* Select a model (e.g., Swin-B)
* Upload a chest X-ray image
* Click **Classify**
* Review the predicted probabilities per condition
* Optionally assign a **true label** to track evaluation accuracy

![img](/assets/ss1.png)

## 📦 Requirements

* Python 3.10+
* Streamlit
* Pillow
* NumPy
* Torch / TensorFlow (as per model requirements)

## 🧑‍⚕️ Medical Disclaimer

> This tool is intended for research and educational purposes only and **must not** be used for clinical diagnosis or treatment decisions.

## 🧑‍💻 Contributors
**Zagarsuren Sukhbaatar** <br>
**Diego Alejandro Ramirez Vargas**<br>
**Shudarshan Singh Kongkham**<br>
**Dhruv Harish Punjwani**

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.