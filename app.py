from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("trained_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Predefined feature list (very important!)
EXPECTED_FEATURES = [
    # Replace these with the actual features used in training
    # Example:
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
    'Gender_Female', 'Gender_Male',
    'family_history_with_overweight_no', 'family_history_with_overweight_yes',
    'FAVC_no', 'FAVC_yes',
    'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
    'SMOKE_no', 'SMOKE_yes',
    'SCC_no', 'SCC_yes',
    'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Public_Transportation', 'MTRANS_Sedentary', 'MTRANS_Walking'
]

@app.route('/')
def home():
    return jsonify({"message": "Obesity Predictor Backend is running!"})

@app.route('/predict-json', methods=['POST'])
def predict_json():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "❌ No input data provided."}), 400

        df = pd.DataFrame([data])

        # One-hot encode input and align columns
        df_encoded = pd.get_dummies(df)
        for col in EXPECTED_FEATURES:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[EXPECTED_FEATURES]

        prediction = model.predict(df_encoded)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": f"❌ Error during prediction: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
