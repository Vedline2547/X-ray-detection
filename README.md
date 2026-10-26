# 🩻 X-ray Pneumonia Detection Using CNN

## Overview

This project uses **Convolutional Neural Networks (CNNs)** with TensorFlow and Keras to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project demonstrates how deep learning and computer vision can be applied to medical image classification. Images are loaded using OpenCV, converted to grayscale, resized to a standard resolution, and normalized before being used to train a CNN model.

The trained model can then process an unseen chest X-ray image and predict its classification.

## 🎯 Project Objectives

- Develop a CNN model for binary classification of chest X-ray images.
- Apply image preprocessing techniques using OpenCV.
- Train and validate a deep learning model using TensorFlow/Keras.
- Save and reload a trained model for inference.
- Visualize X-ray images alongside their predicted classifications.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **TensorFlow & Keras** — Deep learning and CNN development
- **OpenCV** — Image loading, grayscale conversion, and resizing
- **NumPy** — Numerical operations and array manipulation
- **Matplotlib** — Image visualization
- **OS & Random** — File handling and dataset shuffling

## 📂 Dataset Structure

The dataset is organized into two classes:

```text
X-ray-detection/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset folders are excluded from the Git repository because of their size. Obtain the images from the original dataset source and arrange them according to the structure above.

**Note:** The current implementation uses a folder named `test` for validation. For a rigorous evaluation, use a separate `val` directory during training and reserve `test` for final evaluation.

## ⚙️ Data Preprocessing

The project applies the following preprocessing steps:

1. Load chest X-ray images from the dataset directories.
2. Convert images to grayscale using OpenCV.
3. Resize images to **100 × 100 pixels**.
4. Assign numerical labels to each class:
   - `NORMAL` → 0
   - `PNEUMONIA` → 1
5. Shuffle the training and validation samples.
6. Reshape images into CNN-compatible arrays.
7. Normalize pixel values from the range 0–255 to 0–1.

## 🧠 CNN Architecture

The model uses a sequential CNN architecture consisting of:

- Three convolutional layers with 64, 128, and 256 filters.
- ReLU activation functions.
- Max-pooling layers for spatial downsampling.
- Dropout layers for regularization.
- A flattening layer.
- A fully connected dense layer.
- A sigmoid output layer for binary classification.

The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as an evaluation metric.

### Training Configuration

| Parameter | Value |
|---|---|
| Image dimensions | 100 × 100 |
| Image channels | 1 (grayscale) |
| Batch size | 4 |
| Maximum epochs | 10 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Output activation | Sigmoid |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/X-ray-detection.git
cd X-ray-detection
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If you do not yet have a `requirements.txt`, install the main dependencies:

```powershell
python -m pip install tensorflow opencv-python numpy matplotlib
```

### 4. Prepare the dataset

Download the chest X-ray dataset from its original source and place the images into the appropriate class folders.

Update the dataset paths in `main.py` to match your local directory structure.

### 5. Train the model

Run:

```powershell
python main.py
```

The script loads and preprocesses the images, trains the CNN, and saves the trained model.

### 6. Make predictions

After training, load the saved model and pass a chest X-ray image through the preprocessing function to obtain a predicted class.

## 💾 Model Saving and Inference

The trained model can be saved in Keras format:

```python
model.save("model_10.keras")
```

Load it later using:

```python
import tensorflow as tf

model = tf.keras.models.load_model("model_10.keras")
```

The model outputs a sigmoid score. With the current label mapping, a score of at least 0.5 is classified as `PNEUMONIA`, while a lower score is classified as `NORMAL`.

## 📊 Evaluation

The model uses validation data during training to monitor performance. Further evaluation should include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Actual evaluation results should be added here after running the model on a separate, held-out test dataset.

## 🔮 Future Improvements

- Use data augmentation to improve generalization.
- Introduce early stopping and learning-rate scheduling.
- Compare the CNN against transfer-learning models such as MobileNetV2.
- Evaluate performance using precision, recall, F1-score, and a confusion matrix.
- Develop a simple web interface for uploading X-ray images and displaying predictions.
- Optimize memory usage by loading images in batches.
- Explore explainability techniques such as Grad-CAM to visualize image regions influencing predictions.

## ⚠️ Disclaimer

This project is an educational machine-learning experiment and is **not intended for clinical diagnosis or medical decision-making**. Predictions may be incorrect and require rigorous validation before any potential real-world medical application.

## 👨‍💻 Author

**Vedline Ochieng**

- GitHub: [@Vedline2547](https://github.com/Vedline2547)
- LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

*Exploring the intersection of artificial intelligence, deep learning, and practical problem-solving.*
