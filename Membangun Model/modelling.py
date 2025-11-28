# file modelling.py
import pandas as pd
import mlflow
from preprocessing.automated_AzizahSalmaAyunisaPurnomo import automate_Azizah
from sklearn.ensemble import RandomForestClassifier

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Model Statis")

df = pd.read_csv("Eksperimen_SML_Azizah_Salma_Ayunisa_Purnomo/PCOS_raw.csv")
save_path = "Eksperimen_SML_Azizah_Salma_Ayunisa_Purnomo/preprocessing/preprocessing.joblib"
file_path = "Eksperimen_SML_Azizah_Salma_Ayunisa_Purnomo/preprocessing/PCOS_preprocessing.csv"
X_train, X_test, y_train, y_test = automate_Azizah(df, save_path, file_path)

input_example = X_train[0:5]

with mlflow.start_run():
    n_estimators = 500
    max_depth = 30
    min_samples_split = 5
    min_samples_leaf=2
    
    mlflow.autolog()

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    # Logging model
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        input_example=input_example
    )

    rf.fit(X_train, y_train)

    accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
