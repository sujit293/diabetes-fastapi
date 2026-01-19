import json
from unittest import result
import requests

url = "http://127.0.0.1:8000 / dibaetes_predict"

input_data = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 25.6,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}

input_json = json.dumps(input_data)

response = requests.post(url, data=input_json)

print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())
print("Prediction Result:", response.json().get("prediction"))
print(response.text)

