from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("trained_model")
label_encoder = joblib.load("label_encoder")

feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC',
                 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    for feature in feature_names:
        value = request.form.get(feature)
        try:
            val = float(value)
        except:
            val = value
        input_data.append(val)

    df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f"Predicted Obesity Level: {predicted_label}")

if __name__ == '__main__':
    app.run(debug=True)
