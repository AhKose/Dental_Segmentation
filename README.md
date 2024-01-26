# Tooth Image Segmentation Using U-Net

## Overview

This repository contains the implementation of a U-Net-based deep learning model to segment tooth images. U-Net is a convolutional neural network that excels in biomedical image segmentation tasks. This project demonstrates the entire pipeline from data preprocessing and augmentation to model training and evaluation.

## Features

Data Preprocessing: Load and preprocess images and masks.
U-Net Model: Implement the U-Net architecture for segmentation.
Training and Evaluation: Train the model on tooth images and evaluate its performance, visualize the original, masked, and predicted segmented images.

## Project Structure

data_preprocessing.py: Contains functions for loading and preprocessing the images and masks.

model.py: Defines the U-Net architecture.

train.py: Contains the training loop for the model, evaluating and visualizing the model's performance.

## Usage

Data Source
This project utilizes the Tufts Dental Database, available on Kaggle: Tufts Dental Database[https://www.kaggle.com/datasets/deepologylab/tufts-dental-database]. Before running the project, please download this dataset and ensure the paths in the data_preprocessing.py file are correctly set to where you've stored the data.
To define the model, execute the following command:
python model.py
To evaluate the model and visualize the results, run:
python evaluate.py

## Dependencies

TensorFlow
NumPy
scikit-learn
PIL
matplotlib

## Results

![image](https://github.com/ahk19/Dental_Segmentation/assets/48156018/2a5b5653-afdc-4df6-8bb9-3afc96be6f5b)
![image](https://github.com/ahk19/Dental_Segmentation/assets/48156018/a4b546df-1dd8-4206-9746-461cf9071972)
![image](https://github.com/ahk19/Dental_Segmentation/assets/48156018/26f283d5-358a-4857-a624-623cd1f3bba4)

