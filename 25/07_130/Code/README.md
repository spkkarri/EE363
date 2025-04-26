Me and my batch mates had made a youtube video where there is the full explanation and this is the [link](https://youtu.be/_AzU0--vYmM).
Segmentation using DeepLabV3+ with pretrained resnet50v2 model
Dataset
- **Type**: RGB Sentinel-2 satellite images
- **Classes**: Includes land cover categories such as Forest, Residential, Pasture, River, and more.

Models Used
- **DeepLabV3+**: A semantic segmentation model known for high performance on pixel-level tasks.
- **Backbone**: ResNet50V2 (pre-trained on ImageNet)

Project Workflow
1. **Dataset Downloading**
   - Automatically downloads and extracts the EuroSAT dataset.
2. **Data Preparation**
   - Loads satellite images and corresponding masks.
   - Normalizes and resizes images to a uniform input shape.
3. **Data Augmentation**
   - Applies techniques such as random flipping and rotation to increase generalization.
4. **Model Setup**
   - Builds a DeepLabV3+ model with ResNet50V2 as the feature extractor.
5. **Model Compilation**
   - Loss: `SparseCategoricalCrossentropy`
   - Optimizer: `Adam`
   - Metrics: `Accuracy`, `IoU (Intersection over Union)`
6. **Model Training**
   - Trains in batches with checkpointing for the best model.
7. **Prediction & Testing**
   - Predicts segmentation on test data.
   - Visualizes results and compares them to ground truth.

Sample Output
- Visual overlays of model predictions vs ground truth masks.
Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, PIL, scikit-learn, etc.

Install dependencies:
import os
import requests
from zipfile import ZipFile
import glob
from dataclasses import dataclass, field

import random
import numpy as np
import cv2

import tensorflow as tf
import keras_cv

import matplotlib.pyplot as plt

Notes for New Visitors
- This project focuses on **semantic segmentation**, not classification.
- Ideal for those exploring **remote sensing**, **satellite imagery**, or **deep learning for Earth observation**.
- GPU recommended for training.
- Google colab notebook format for interactive experimentation.



Land cover classification with eurosat dataset 
# Land Cover Classification with EuroSAT Dataset 🌍🛰️
Actually we trained 7 models but we shown and ran only one model due to time constraints and that one model is convNext which is latest model than any other model that is there in this project

This project performs land cover classification using the [EuroSAT RGB dataset](https://www.kaggle.com/datasets/nilesh789/eurosat-rgb), which contains satellite images grouped into various land cover categories.

The goal is to classify land types like forest, residential, industrial, and agricultural areas using a convolutional neural network (CNN) built with TensorFlow/Keras.

Dataset

- **Source**: [EuroSAT RGB Dataset on Kaggle](https://www.kaggle.com/datasets/nilesh789/eurosat-rgb)
- **Format**: RGB images in `.jpg` format
- **Classes**: 10 land use and land cover classes

Features

- Automatic dataset download and extraction from Kaggle
- Data preprocessing with TensorFlow image pipelines
- CNN-based classification model using Keras
- Training, validation, and performance visualization
- Model evaluation with accuracy and confusion matrix

Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy / Matplotlib
- scikit-learn
- Google Colab (for GPU acceleration)

Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jaiganesh5555/land-cover-classification-eurosat.git
   cd land-cover-classification-eurosat

