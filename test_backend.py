import requests
url = "http://192.168.33.77:5000/predict-json"


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

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Raw Response Text:")
    print(response.text)  # Print full raw response from server
except Exception as e:
    print("❌ Error:", e)
