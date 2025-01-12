# Deepfake Detection Model

This repository contains a deep learning-based model for detecting deepfake images. The model uses a combination of multiple pre-trained models, aggregated to provide robust and accurate predictions. The goal is to detect whether an image is real or fake.

## Table of Contents
- [Overview](#overview)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output](#inputoutput)
- [LIME Explanations](#lime-explanations)
- [Model on Kaggle](#model-on-kaggle)
- [License](#license)

## Overview

This model leverages deep learning techniques to classify images as real or fake. By using multiple pre-trained models and an aggregation method for predictions, it ensures accurate and reliable results.

**Key Features:**
- **Multiple Model Approaches:** Utilizes diverse pre-trained models for enhanced performance.
- **Prediction Voting System:** Aggregates predictions from multiple models to improve accuracy.
- **Face Detection:** Uses OpenCV's Haar Cascade to focus the model’s attention on the face area in the image.
- **LIME Integration:** Provides an explanation of model predictions by visualizing which parts of the image contributed to the decision.

## Models

The model was trained using the following datasets:

- **[140K Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)**  
- **[Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)**

These datasets provided the images necessary to train and evaluate the deepfake detection models.

## Installation

To use this project, you need to install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection

2. Install dependencies
   ```bash
   pip install -r requirements.txt   

# Usage

## 1. Running the Model Prediction

Once the repository is cloned and dependencies are installed, you can use the model to predict if an image is real or fake.

**Functionality:**

* The model expects an image input (in formats such as .jpg, .png).
* The model preprocesses the image (including face detection, cropping, and resizing).
* Then, the model aggregates the predictions from multiple pre-trained models.

**Example of Predicting with the Model:**

```python
from deepfake_detector import DeepfakeDetector

# Initialize the DeepfakeDetector
detector = DeepfakeDetector()

# Predict whether the image is real or fake
image_path = 'path_to_image.jpg'
prediction, real_score, fake_score = detector.predict(image_path)

print(f"Prediction: {prediction}")
print(f"Real Score: {real_score}%")
print(f"Fake Score: {fake_score}%")

## 2. Preprocessing the Image

Before making predictions, the image goes through several preprocessing steps:

* **Face Detection:** Using OpenCV's Haar Cascade, the face(s) in the image are detected.
* **Cropping and Resizing:** Once the face is detected, the image is cropped and resized to fit the model's input size.
* **Normalization:** The pixel values are normalized to be within the [0,1] range to match the training conditions.

**Example Code:**

```python
from deepfake_detector import ImagePreprocessor

# Initialize the ImagePreprocessor
preprocessor = ImagePreprocessor()

# Preprocess the image
preprocessed_image = preprocessor.process_image(image_path)

# Display the preprocessed image (optional)
import matplotlib.pyplot as plt
plt.imshow(preprocessed_image)
plt.show()

## 3. LIME Integration for Model Explanations

LIME (Local Interpretable Model-Agnostic Explanations) helps explain which parts of the image contributed to the prediction. The explanation helps visualize how the model makes its decisions by highlighting key regions.

**Example Code:**

```python
from deepfake_detector import LimeExplainer

# Initialize LIME Explainer
lime_explainer = LimeExplainer()

# Get the LIME explanation for a given image
lime_explanation = lime_explainer.explain(image_path) 

# Display the explanation 
lime_explanation.show_image()

## 4. Batch Prediction

If you have multiple images to predict at once, you can process them in a batch:

**Example Code:**

```python
# List of image paths
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# Predict for each image
predictions = [detector.predict(image) for image in image_paths]

# Output the predictions
for idx, (prediction, real_score, fake_score) in enumerate(predictions):
  print(f"Image {idx + 1}: {prediction}, Real: {real_score}%, Fake: {fake_score}%")

## 5. Saving and Loading Models

You can save and load the trained models for later use:

**Saving the Model:**

```python
detector.save_model('deepfake_model.h5')

## 6. Evaluating the Model Performance

After training the model, you can evaluate its performance using accuracy, F1 score, ROC curve, and AUC:

```python
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

# Load test images and ground truth labels
test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
true_labels = [1, 0, 1]  # 1 for real, 0 for fake

# Predict the labels for the test images
predictions = [detector.predict(image)[0] for image in test_images]

# Convert predictions to binary labels (0 or 1)
predicted_labels = [1 if pred == 'Real' else 0 for pred in predictions]

# Calculate accuracy and F1 score
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Display the results
print(f"Accuracy: {accuracy * 100}%")
print(f"F1 Score: {f1}")

**Input/Output**

* **Input:** Image File: The system accepts an image file in JPG, PNG, or other common formats.
* **Output:**
    * **Prediction:** The output will indicate whether the image is real or fake.
    * **Confidence Scores:** The output also provides the confidence levels for both real and fake predictions.

**Example:**

```json
{
  "prediction": "Real",
  "real_score": 76.23,
  "fake_score": 23.77
}

