import requests

url = "http://127.0.0.1:8000/predict"

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

response = requests.post(url, json=input_data)

print("Status Code:", response.status_code)

try:
    print("Response:", response.json())
except Exception:
    print("Raw Response:", response.text)
