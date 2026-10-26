# 🩻 X-Ray Pneumonia Detection

A deep learning project that explores chest X-ray image classification to distinguish between **NORMAL** and **PNEUMONIA** cases using Convolutional Neural Networks (CNNs) and transfer learning with VGG16. The project also includes a **Streamlit web application** for interactive image uploads and model predictions.

## 🚀 Project Overview

Pneumonia is a respiratory infection that can affect the lungs. This project explores how computer vision and deep learning can be applied to chest X-ray image classification.

The project implements two approaches:

- **Custom CNN:** A convolutional neural network trained for binary image classification.
- **VGG16 Transfer Learning:** A pretrained VGG16 architecture adapted to the binary classification task.
- **Streamlit App:** A web interface that allows users to upload an X-ray image and view the model's prediction.

This is an educational machine learning project, not a clinically validated diagnostic system.

## 🎯 Objectives

- Build a deep learning model for chest X-ray image classification.
- Explore CNN architectures for medical image analysis.
- Apply transfer learning using VGG16 pretrained on ImageNet.
- Compare custom CNN and transfer-learning approaches.
- Deploy a trained model through a Streamlit web application.

## 🧠 Models

### 1. Custom Convolutional Neural Network (CNN)

The custom CNN learns image features through convolutional and pooling layers.

Key components include:
- Convolutional layers with ReLU activation.
- Max-pooling layers.
- Dropout for regularization.
- Dense layers for classification.
- Sigmoid output for binary prediction.

### 2. VGG16 Transfer Learning

VGG16 is a convolutional neural network pretrained on ImageNet. Its pretrained convolutional base can be used to extract image features for the chest X-ray classification task.

The transfer-learning workflow includes:
- Loading pretrained VGG16 weights.
- Adapting the model for binary classification.
- Freezing the pretrained base during initial training.
- Adding classification layers for NORMAL and PNEUMONIA predictions.

## 🌐 Streamlit Web Application

The project includes an interactive web application that allows users to:

- Upload a chest X-ray image.
- Preview the uploaded image.
- Run the trained model.
- View the predicted class and model score.

### Application Preview

Run the app locally to explore the interface.

## 🗂️ Project Structure

```text
X-ray-detection/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── app.py
├── pretrained.py
├── pretrained_model.h5
├── model_10.h5
└── README.md
```

*Note: Adjust this structure to match the actual files in your repository. Only include model files that you have saved and intend to track in Git.*

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- VGG16
- NumPy
- Pillow (PIL)
- Matplotlib
- OpenCV
- Streamlit

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/X-ray-detection.git
cd X-ray-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install tensorflow streamlit numpy pillow matplotlib opencv-python
```

## ▶️ Run the Streamlit App

Ensure that the trained model file exists at the path configured in `app.py`. Update the model path if necessary.

Run:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

## 📊 Evaluation

The models should be evaluated using a held-out test dataset. Useful metrics include:

- Accuracy
- Precision
- Recall / sensitivity
- F1-score
- Confusion matrix

For this classification problem, recall for pneumonia cases is especially important to examine alongside false positives and false negatives.

*Add your actual evaluation results here once measured. No performance figures are claimed in this README.*

## 🔬 Learning Outcomes

Through this project, I am exploring:

- Image preprocessing and preparation for deep learning.
- CNN architecture design and training.
- Transfer learning with pretrained image-classification models.
- Binary classification with TensorFlow/Keras.
- Deploying machine learning models using Streamlit.
- Building interactive interfaces for computer vision projects.

## 🔮 Future Improvements

- Compare CNN and VGG16 performance using the same test dataset.
- Add precision, recall, F1-score, and confusion matrix reporting.
- Experiment with data augmentation and fine-tuning.
- Improve image preprocessing and prediction-time validation.
- Deploy the application online.
- Investigate model interpretability using Grad-CAM.

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It has not been established as a clinically validated medical device. Predictions may be incorrect and must not be used as the sole basis for diagnosis or treatment. Medical decisions should be made by qualified healthcare professionals.

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

- GitHub: [@Vedline2547](https://github.com/Vedline2547)
- LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

---

*Exploring the intersection of deep learning, computer vision, and healthcare through practical machine learning projects.*
