from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.staticfiles import StaticFiles

# -------------------------
# 1. Wczytanie modelu (pipeline + preprocessing)
# -------------------------
model = joblib.load("xgboost_pipeline.pkl")  # Twój zapisany pipeline

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
    # Zamiana Pydantic -> nazwy kolumn ze spacją
    input_dict = data.dict()
    mapping = {
        "Sleep_Duration": "Sleep Duration",
        "Physical_Activity_Level": "Physical Activity Level",
        "Stress_Level": "Stress Level",
        "BMI_Category": "BMI Category",
        "Heart_Rate": "Heart Rate",
        "Daily_Steps": "Daily Steps"
    }
    for k, v in mapping.items():
        input_dict[v] = input_dict.pop(k)
    
    # Zamiana danych na DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predykcja (pipeline zwraca liczby 0,1,2)
    pred_label_encoded = model.predict(input_df)

    # Ręczne mapowanie klas
    label_map = {0: "None", 1: "Insomnia", 2: "Sleep Apnea"}
    pred_label_name = label_map[pred_label_encoded[0]]

    return {"predicted_sleep_disorder": pred_label_name}

# -------------------------
# 5. Mount frontend na /static
# -------------------------
app.mount("/static", StaticFiles(directory=".", html=True), name="frontend")