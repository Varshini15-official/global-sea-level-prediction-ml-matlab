🌊 Global Sea Level Rise Prediction Using Machine Learning
A MATLAB project analyzing historical global mean sea level (GMSL) data and forecasting future sea level rise from 2014 to 2100 using Linear Regression, Support Vector Machine (SVM), and Neural Network models.

🚀 Project Overview
Dataset: Monthly global sea level data (1983–2013) with uncertainty values

Objective: Train ML models on past 30 years’ data to predict future sea level rise

Models:

Linear Regression

Support Vector Machine (SVM) with linear kernel

Feedforward Neural Network (2 hidden layers)

Includes measurement uncertainty visualization

Evaluation using Mean Absolute Error (MAE) on test data

Interactive user input to predict sea level for any year (2014–2100)

✨ Features
Data preprocessing and cleaning (missing values handled)

Numeric conversion of time stamps for modeling

Model training with execution time measurement

Prediction on test data and extended future range

Graphical comparison of original data, predictions, and uncertainty

User-friendly input for customized year prediction

▶️ How to Run
Place the dataset globalsealevelnew.csv in your desired folder.

Update the dataset path in the MATLAB script:

matlab
Copy
Edit
data = readtable('C:\Your\Path\Here\globalsealevelnew.csv');
Run the MATLAB script.

Enter a year between 2014 and 2100 when prompted to get sea level rise forecasts from all models.

📂 Dataset Description
Time: Monthly timestamps (e.g., "1983-Apr")

GMSL: Global Mean Sea Level (in millimeters)

GMSLuncertainty: Measurement uncertainty in sea level

🛠 Dependencies
MATLAB (R2018a or later recommended)

Statistics and Machine Learning Toolbox

Neural Network Toolbox

Deep Learning Toolbox (for SVM and Neural Network training)

📊 Results
Training duration for each model

Mean Absolute Error (MAE) on testing data

Visual plots including:

Original sea level data with uncertainty shading

Model predictions on test data

Future forecasts (2014–2100)

Comparative forecast of all three models

👩‍💻 Author
Developed by Varshini15-official

