from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.sav")

app = FastAPI(title="Diabetes Prediction API")

# Input schema
class InputData(BaseModel):
    Pregnancies: int = Field(ge=0)
    Glucose: int = Field(gt=0)
    BloodPressure: int = Field(gt=0)
    SkinThickness: int = Field(ge=0)
    Insulin: int = Field(ge=0)
    BMI: float = Field(gt=0)
    DiabetesPedigreeFunction: float = Field(gt=0)
    Age: int = Field(gt=0)

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        response = {"prediction": result}

        # Optional probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_array)[0][1]
            response["probability"] = round(float(prob), 3)

        return response

    except Exception as e:
        return {"error": str(e)}
