from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.staticfiles import StaticFiles
# -------------------------
# 1. Wczytanie modelu (pipeline + label encoding w środku)
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

    # Predykcja (pipeline zwraca już oryginalne etykiety)
    pred_label = model.predict(input_df)
    
    # Konwersja predykcji na string, żeby JSON przeszedł poprawnie
    pred_label_py = str(pred_label[0])

    return {"predicted_sleep_disorder": pred_label_py}


app.mount("/", StaticFiles(directory=".", html=True), name="frontend")