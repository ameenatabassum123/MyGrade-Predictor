# SmartGrade Predictor

> Predict final exam scores for students using machine learning and interactive inputs — powered by Streamlit!

## Overview

MyGrade Predictor is a web-based app that estimates students' final exam scores (G3) based on their study behavior and academic performance throughout the year. This project leverages Python, pandas, scikit-learn, and Streamlit to provide a user-friendly demo for data science in education.

## Features

- Interactive sidebar for student feature inputs
- Predicts final exam score using a trained regression model
- Displays model accuracy and coefficients
- Visualizes actual vs predicted scores
- Clean and modern interface with customizable look

## How to Use

1. Clone this repo and place `student-mat.csv` in the root folder.
2. Install requirements with:  
   `pip install -r requirements.txt`
3. Run the app:  
   `streamlit run app.py`
4. Enter data in the sidebar and click "Predict Final Exam Score" to see predictions.

## Tech Stack

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- Streamlit

## Dataset

Uses the [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/student+performance).

## Author

Made by [Ameena Tabassum]
