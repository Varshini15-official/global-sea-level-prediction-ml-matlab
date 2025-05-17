# Global Sea Level Rise Prediction Using Machine Learning
This MATLAB project analyzes historical global mean sea level (GMSL) data and predicts future sea level rise from 2014 to 2100 using three machine learning models: Linear Regression, Support Vector Machine (SVM), and a Neural Network.

## Project Overview

- Dataset: Monthly global sea level data (GMSL) with uncertainty, covering April 1983 to September 2013.
- Goal: Train and test models on the last 30 years of observed data to forecast future sea level changes.
- Models Implemented:
  - Linear Regression
  - Support Vector Machine (SVM) with a linear kernel
  - Feedforward Neural Network with two hidden layers
- Uncertainty: Incorporates uncertainty in measurements during visualization and future predictions.
- Evaluation: Calculates Mean Absolute Error (MAE) on test data for each model.
- Visualization: Includes plots comparing original data, training/testing splits, model predictions, and uncertainty bounds.
- User Interaction: Allows user to input a future year (2014–2100) to get sea level rise predictions from all models.

## Features
- Data cleaning and preprocessing (handling missing values)
- Conversion of time labels to numeric years
- Model training with performance timing
- Prediction for both testing period and extended future years
- Visual comparison of model accuracy and predictions
- Interactive prediction for any specified year within the forecast range

## How to Run
1. Place the dataset file `globalsealevelnew.csv` in a known location.
2. Update the file path in the MATLAB script accordingly:
   data = readtable('C:\Users\varsh\OneDrive\Documents\globalsealevelnew.csv');
3. Run the MATLAB script.
4. Follow the prompt to enter a year between 2014 and 2100 for a sea level rise prediction.

## Dataset Description
- The dataset contains the following columns:
- Time — Monthly time stamps (e.g., "1983-Apr")
- GMSL — Global Mean Sea Level values (in millimeters)
- GMSLuncertainty — Uncertainty in sea level measurements

## Dependencies
- MATLAB (tested on R2018a or later recommended)
- Statistics and Machine Learning Toolbox and deep learning (for Linear Regression and SVM)
- Neural Network Toolbox (for training feedforward neural networks)
- And other required tools after u run the code

## Results
- The project outputs:
- Training time for each model
- Mean Absolute Error (MAE) on test data
- Visual plots showing:
- Original data with uncertainty shading
- Predictions on test data
- Future forecasts (2014–2100)
- Comparative forecast of all models

## Author
Developed by Varshini15-official

