import joblib
import pandas as pd

# Wczytaj zapisany model
model = joblib.load("xgboost_pipeline.pkl")

# Przykładowy wiersz danych – wszystkie cechy użyte w modelu
sample_data = pd.DataFrame([{
    "Age": 30,
    "BMI": 22.5,
    "Systolic": 120,
    "Diastolic": 80,
    "Gender": "Male",
    "Occupation": "Engineer",
    "Sleep Duration": 7.0,
    "Physical Activity Level": 60,
    "Stress Level": 4,
    "Heart Rate": 70,
    "Daily Steps": 7000,
    "BMI Category": "Normal"
}])

# Predykcja
pred = model.predict(sample_data)
print("Predicted Sleep Disorder:", pred[0])
