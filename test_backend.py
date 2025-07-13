import requests

url = "http://127.0.0.1:5000/predict-json"

data = {
  "Gender": "Male",
  "Age": 23,
  "Height": 1.75,
  "Weight": 70,
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "FCVC": 2,
  "NCP": 3,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2,
  "SCC": "no",
  "FAF": 1,
  "TUE": 3,
  "CALC": "Sometimes",
  "MTRANS": "Walking"
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
