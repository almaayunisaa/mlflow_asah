# file modelling_tuning.py
from preprocess import preprocess_data
import pandas as pd
import mlflow
from preprocessing.automated_AzizahSalmaAyunisaPurnomo import automate_Azizah
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("Eksperimen_SML_Azizah_Salma_Ayunisa_Purnomo/PCOS_raw.csv")
save_path = "Eksperimen_SML_Azizah_Salma_Ayunisa_Purnomo/preprocessing/preprocessing.joblib"
file_path = "Eksperimen_SML_Azizah_Salma_Ayunisa_Purnomo/preprocessing/PCOS_preprocessing.csv"
X_train, X_test, y_train, y_test = automate_Azizah(df, save_path, file_path)
input_example = df[0:5]

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Online Training")

# Parameter Grid untuk GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300, 505, 700],
    'max_depth': [5, 10, 15, 20, 25, 37, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
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