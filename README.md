# 🌊 Global Sea Level Rise Prediction Using Machine Learning

A MATLAB project analyzing historical global mean sea level (GMSL) data and forecasting future sea level rise from 2014 to 2100 using **Linear Regression**, **Support Vector Machine (SVM)**, and **Neural Network models**.

---

## 🚀 Project Overview

- Dataset: Monthly global sea level data (1983–2013) with uncertainty values  
- Objective: Train ML models on past 30 years’ data to predict future sea level rise  

## 📈 Models Used

- Linear Regression  
- Support Vector Machine (SVM) with linear kernel  
- Feedforward Neural Network with 2 hidden layers  
- Visualizes measurement uncertainty  
- Evaluation through Mean Absolute Error (MAE) on test data  
- Allows user to input any year between 2014–2100 for prediction  

## ✨ Features

- Cleans and preprocesses data, handles missing values  
- Converts time labels to numeric year format  
- Trains models with execution time tracking  
- Forecasts both test range and future data  
- Displays plots comparing original data, predictions, and uncertainty  
- Interactive input: user provides a year to get model forecasts  

## ▶️ How to Run

Run the MATLAB script

When prompted, enter a year between 2014 and 2100 to view predictions

---

## 📂 Dataset Description

- Time: Monthly timestamps (e.g., "1983-Apr")  
- GMSL: Global Mean Sea Level in millimeters  
- GMSLuncertainty: Measurement uncertainty values

---

## 🛠 Dependencies

- MATLAB (R2018a or later recommended)  
- Statistics and Machine Learning Toolbox  
- Neural Network Toolbox  
- Deep Learning Toolbox

---

## 📊 Results

- Reports model training time  
- Calculates Mean Absolute Error (MAE) for testing data  
- Generates the following plots:  
  - Original sea level with uncertainty shading  
  - Predictions vs. test data  
  - Forecasts from 2014 to 2100  
  - Comparison plot of all three models’ predictions

---

## 👩‍💻 Author

Developed by Varshini15-official
