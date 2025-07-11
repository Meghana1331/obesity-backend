from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoder
model = joblib.load("trained_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

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

        # One-hot encode input to match training
        df_encoded = pd.get_dummies(df)

        # Align columns with training data
        model_features = model.feature_name_
        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_features]

        prediction = model.predict(df_encoded)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": f"❌ Error during prediction: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
