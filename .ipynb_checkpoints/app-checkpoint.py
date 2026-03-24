from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -------------------------
# 1. Wczytanie modelu (pipeline + label encoding w środku)
# -------------------------
model = joblib.load("models/xgboost_pipeline.pkl")  # Twój zapisany pipeline

# -------------------------
# 2. FastAPI app
# -------------------------
app = FastAPI(
    title="Sleep Disorder Prediction API",
    description="Predicts Sleep Disorder based on user data",
    version="1.0"
)

# -------------------------
# 3. Schemat danych wejściowych
# -------------------------
class SleepData(BaseModel):
    Age: int
    BMI: float
    Systolic: int
    Diastolic: int
    Gender: str
    Occupation: str
    Sleep_Duration: float
    Physical_Activity_Level: int
    Stress_Level: int
    Heart_Rate: int
    Daily_Steps: int
    BMI_Category: str

# -------------------------
# 4. Endpoint predykcji
# -------------------------
@app.post("/predict")
def predict_sleep_disorder(data: SleepData):
    # zamiana danych na DataFrame
    input_df = pd.DataFrame([data.dict()])

    # predykcja (pipeline zwraca już oryginalne etykiety)
    pred_label = model.predict(input_df)

    return {"predicted_sleep_disorder": pred_label[0]}