# Random-Forest-Classification
## Pneumonia Detection using Chest X-ray Images

### Overview
This repository contains code for building a machine learning model that can classify chest X-ray images as either `NORMAL` or having `PNEUMONIA`.

### Workflow
The process can be broken down into two main steps:
1. Preprocessing and saving the X-ray image data.
2. Training a RandomForest classifier and evaluating its performance.

### 1. Preprocessing and Saving X-ray Image Data
This part of the code is responsible for loading, preprocessing, and saving chest X-ray image data. 

#### Steps:
- **Directories Setup:**  
  We define two directories containing JPG images for `NORMAL` and `PNEUMONIA` cases.
  
- **Resizing Dimensions:**  
  The desired dimensions for the images are set to 256x256 pixels.
  
- **Data Loading & Preprocessing:**  
  For each class label (`NORMAL` and `PNEUMONIA`):
  - List all `.jpeg` image files in the directory.
  - Load each image using OpenCV.
  - Convert the image to grayscale.
  - Resize the image to the desired dimensions.
  - Store the preprocessed image data and its corresponding label.

- **Saving Processed Data:**  
  The processed image data and labels are saved to `.joblib` files for future use.

### 2. Training and Evaluating RandomForest Classifier
In this section, we train a RandomForest classifier on the preprocessed image data and evaluate its performance on a test set.

#### Steps:
- **Data Loading:**  
  Load the preprocessed data and labels from the `.joblib` files.
  
- **Data Flattening:**  
  The 2D image data is flattened to 1D to make it suitable for the classifier.
  
- **Data Splitting:**  
  Split the data into training and testing sets.
  
- **RandomForest Classifier Setup:**  
  A RandomForest classifier is defined with certain hyperparameters (number of trees, max depth, min samples split, etc.). 

- **Model Training:**  
  The RandomForest classifier is trained on the training data.
  
- **Model Saving:**  
  The trained model is saved to a `.joblib` file for deployment or future use.
  
- **Performance Evaluation:**  
  The trained model's performance is evaluated on the test set in terms of F1 score and test accuracy. A confusion matrix is also generated to understand the model's classification behavior visually.

### Libraries Used
- **pandas**: Data manipulation.
- **numpy**: Numerical operations.
- **sklearn**: Machine learning model creation, training, and evaluation.
- **cv2 (OpenCV)**: Image loading and preprocessing.
- **joblib**: Saving and loading data/models.
- **seaborn & matplotlib**: Data visualization.

### Note
Make sure to adjust the directories in the code to match your local dataset paths.
