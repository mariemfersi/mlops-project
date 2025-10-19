from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from model_pipeline import load_model, save_model, prepare_data, select_and_train_model

app = FastAPI(title="Insurance Charges Predictor")

# Load model and scaler at startup
model, scaler = load_model("model.pkl")


class InputData(BaseModel):
    features: list


class RetrainData(BaseModel):
    data_path: str
    target: str
    model_type: str = None


@app.get("/columns")
def get_columns():
    if scaler is None or not hasattr(scaler, "feature_columns"):
        return {"error": "Le scaler n'a pas encore été initialisé."}
    return {"feature_columns": scaler.feature_columns}


@app.post("/predict")
def predict(data: InputData):
    try:
        df_input = pd.DataFrame([data.features], columns=scaler.feature_columns)
        X_scaled = scaler.transform(df_input)
        prediction = model.predict(X_scaled)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain")
def retrain(data: RetrainData):
    global model, scaler
    try:
        X_train, X_test, y_train, y_test = prepare_data(
            data.data_path, data.target
        )

        params_dict = {"n_estimators": 100, "max_depth": 10}
        tags_dict = {"author": "Mariem", "version": "v1.0"}

        if data.model_type:
            model = select_and_train_model(
                X_train,
                y_train,
                model_type=data.model_type,
                params_dict=params_dict,
                tags_dict=tags_dict,
            )
        else:
            model = select_and_train_model(
                X_train, y_train, params_dict=params_dict, tags_dict=tags_dict
            )

        save_model(model, "model.pkl")
        _, scaler = load_model("model.pkl")

        return {"message": f"Model retrained successfully with type: {type(model).__name__}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
