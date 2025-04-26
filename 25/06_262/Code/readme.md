# EE363 Project Submission

## Team Info
- **Team Number**: 6
- **Roll Numbers**: 522262, 522216, 522222, 522259, 522256, 522239

## Project Title
Classification and segmentation of satellite images

## Description
This project uses  Vision Transformer (ViT) models to classify satellite images from the EuroSAT dataset into various land cover categories and to perform binary segmentation of satellite images using Deepglobelandcover dataset.

## Files & Folders
- `Code/`: Contains Python code and this readme file
- `Code/assets/`: Contains the one-slide project summary (PPT)
- `Code/vit_eurosat.pth`: Pretrained ViT model weights for classification for satellite images
-`Code/binary_seg_models`:Folder containing files having pretrained ViT model weights for binary segmentation of satellite images

## How to Run

This project contains 4 main Jupyter Notebook (`.py`) files, developed and tested on Google Colab:

### 🔹 Model Training
1. **classificationofsatelliteimages.py**  
   - Trains a Vision Transformer (ViT) model for multiclass classification of satellite images using the EuroSAT dataset.
You can run this notebooks in Google Colab to train the models. Trained weights are saved as `vit_eurosat.pth`.
2. **binarysegmentationofsatelliteimages.py**  
   - Performs class-wise binary segmentation on satellite images using a segmentation model.
You can run these notebooks in Google Colab to train the models.Datasets should be placed in drive and image and mask paths should be changed while training according to actual location. Trained weights are saved  in `binary_seg_models` folder.

---

### 🔹 User Interface (UI)

3. **CSIUserInterface.py**  
   - Loads the trained classification model (`vit_eurosat.pth`) and allows the user to upload an image for land cover classification.

4. **BSSIUserInterface.py**  
   - Loads the trained classification models (`binary_seg_models`) and allows the user to upload an image for land cover classification.

---

### 💻 How to Execute in Google Colab
1. Open the notebook in Google Colab.
2. For testing binary segmentation of satellite images upload trained models folder location in this manner ->"/content/drive/MyDrive/binary_seg_models"
3. For testing classification of satellite images upload trained model location in this manner ->"/content/drive/MyDrive/vit_eurosat.pth"
4. Run the cell to get interface to upload images.
5. These steps are for testing both classification and binary segmentation.
---
## Drivelink for DeepGlobeLandCover dataset
https://drive.google.com/drive/folders/1DtZD0Ttb6RnHKC-XYdqG0iUHkagLt3ns

## Drivelink for Eurosat dataset
https://drive.google.com/file/d/1OcQoPnkezlKd0RHkruL3vgJ6Lw4BSDmR/view?usp=sharing

## Drivelink for vit_eurosat.ph
https://drive.google.com/file/d/1x3aL6wmhfufo9m-x9dLHEY4rmPLUNCi_/view?usp=sharing

## Drivelink for binary_seg_models folder
https://drive.google.com/drive/folders/1m-32qJXYRcSpyov0W4Tg0mGRNiHTuDxG?usp=sharing

## Drivelink for Video
https://drive.google.com/file/d/1hzJMaAxY7UlTZlE2zUGFFsVTZpD2SRal/view?usp=drivesdk
