# file modelling_tuning.py
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)

from preprocessing import automate_Azizah

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Online Training")

df = pd.read_csv("../PCOS_raw.csv")
save_path = "preprocessing.joblib"
file_path_header = "PCOS_preprocessing_header.csv"
file_path_data = "PCOS_preprocessing.csv"
X_train, X_test, y_train, y_test = automate_Azizah(df, save_path, file_path_header, file_path_data)

input_example = X_train[0:5]

# Parameter Grid untuk GridSearchCV
param_grid = {
    'n_estimators': [50, 700],
    'max_depth': [5,50],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}

with mlflow.start_run():
    # Inisialisasi dan Grid Search
    rf = RandomForestClassifier()

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    # Mendapatkan Parameter dan Model Terbaik
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Log best parameters
    mlflow.log_params(best_params)

    # Train the best model on the training set
    best_model.fit(X_train, y_train)

    # Logging model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example
    )

    # Evaluate the model on the test set and log accuracy
    accuracy = best_model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    
    
    