# Telco Customer Churn Prediction App

## Overview

This is a Streamlit-based web application for predicting customer churn using the Telco Customer Churn dataset. The application allows users to input customer details and select from various pre-trained machine learning models to predict whether a customer will stay or leave the service.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)

## Introduction

The **Telco Customer Churn** dataset is widely used in the telecommunications industry to predict customer churn. Churn refers to the phenomenon where customers discontinue their subscription to a service. Understanding and predicting customer churn is crucial as retaining existing customers is more cost-effective than acquiring new ones.

## Features

- **Customer Demographics:** Gender, Senior Citizen status, Partner, Dependents.
- **Services Subscribed:** Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
- **Account Information:** Tenure, Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges.
- **Machine Learning Models:** Logistic Regression, K-Nearest Neighbors, Decision Tree, Support Vector Machine, Random Forest, XGBoost.

## Dataset

The dataset includes various features that capture the demographics, services, and account information of customers. This data helps in understanding the factors contributing to customer churn.

## Installation

To run this application locally, please follow these steps:


1-Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

2-Install the required packages:
pip install -r requirements.txt

3-Download the pre-trained models:

Place the following model files in the root directory of the project:

LogisticRegression_best_model.joblib
KNeighborsClassifier_best_model.joblib
DecisionTreeClassifier_best_model.joblib
SVC_best_model.joblib
RandomForestClassifier_best_model.joblib
XGBClassifier_best_model.joblib

5-Run the Streamlit app:


Usage
Load the application:
The app will be accessible in your browser after running the Streamlit command.

Input Features:

Select the customer features using the provided options.
Choose a machine learning model for prediction.
Predict Churn:

Click the "Predict" button to see whether the customer is predicted to stay or leave.
Model Details
The app supports the following machine learning models:

Logistic Regression
K-Nearest Neighbors
Decision Tree
Support Vector Machine
Random Forest
XGBoost
These models were pre-trained on the Telco Customer Churn dataset and saved using the joblib library.

Contributing
If you wish to contribute to this project, feel free to submit a pull request. Please ensure your code is well-documented and adheres to the existing code style.
