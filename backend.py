from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
MODEL_PATH = "best_rf_model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None

@app.on_event("startup")
def load_models():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Model and Scaler loaded successfully.")
    else:
        print("⚠️ Model or Scaler not found! Ensure training script has finished.")

# Pydantic schema matching the dataset features
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/predict")
def predict_risk(data: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    patient_dict = data.dict()
    input_arr = np.array([[patient_dict.get(f, 0) for f in feature_names]])
    
    try:
        input_sc = scaler.transform(input_arr)
        pred = model.predict(input_sc)[0]
        prob = model.predict_proba(input_sc)[0][1]
        
        risk = "HIGH RISK 🔴" if prob >= 0.6 else ("MODERATE RISK 🟡" if prob >= 0.4 else "LOW RISK 🟢")
        
        return {
            "Prediction": "Heart Disease" if pred == 1 else "No Heart Disease",
            "Probability": round(float(prob), 4),
            "Risk_Level": risk
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory=".", html=True), name="static")
