import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import mlflow
import mlflow.sklearn

scaler = None  # global

def prepare_data(data_path: str, target: str, test_size=0.2, random_state=42):
    global scaler
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler.feature_columns = X.columns.tolist()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test





def select_and_train_model(X_train, y_train, model_type=None):
    

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    best_model, best_score = None, float("inf")
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("insurance_charges")

    if model_type:
        if model_type not in models:
            raise ValueError(f"Invalid model_type. Choose from {list(models.keys())}")
        model = models[model_type]
   
        with mlflow.start_run(run_name=model_type):
            # Log parameters if provided
            if params_dict:
                mlflow.log_params(params_dict)
            # Log tags if provided
            if tags_dict:
                mlflow.set_tags(tags_dict)
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)

            # Calcul des métriques
            mse = mean_squared_error(y_train, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_train, y_pred)
            r2 = r2_score(y_train, y_pred)

            # Log metrics dans MLflow
            mlflow.log_metric("train_mse", mse)
            mlflow.log_metric("train_rmse", rmse)
            mlflow.log_metric("train_mae", mae)
            mlflow.log_metric("train_r2", r2)

            # Log du modèle
            mlflow.sklearn.log_model(model, "model")

            print(f"Trained {model_type} | MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

        return model

    # Pour la sélection automatique de modèle, on peut aussi logguer chaque run
    params = {
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
        "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name in params:
                from sklearn.model_selection import GridSearchCV
                grid = GridSearchCV(model, params[name], cv=3, scoring="neg_mean_squared_error")
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_train)

            mse = mean_squared_error(y_train, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_train, y_pred)
            r2 = r2_score(y_train, y_pred)

            mlflow.log_metric("train_mse", mse)
            mlflow.log_metric("train_rmse", rmse)
            mlflow.log_metric("train_mae", mae)
            mlflow.log_metric("train_r2", r2)
            mlflow.sklearn.log_model(model, "model")

            if mse < best_score:
                best_model, best_score = model, mse

            print(f"{name} | MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

    print(f"Best model: {type(best_model).__name__} with train MSE: {best_score:.2f}")
    return best_model


def evaluate_model(model, X_test, y_test, save_plots=True):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    print("Evaluation metrics:", metrics)
    return metrics


def save_model(model, model_path="model.pkl"):
    global scaler
    feature_columns = getattr(scaler, "feature_columns", None)
    joblib.dump((model, scaler, feature_columns), model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="model.pkl"):
    model, scaler, feature_columns = joblib.load(model_path)
    scaler.feature_columns = feature_columns
    return model, scaler

