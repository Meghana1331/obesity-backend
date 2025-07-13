import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("train.csv")

# Encode the target
label_encoder = LabelEncoder()
data["NObeyesdad"] = label_encoder.fit_transform(data["NObeyesdad"])

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# One-hot encode categorical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC',
                        'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
data = pd.get_dummies(data, columns=categorical_features)

# Split features and target
X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

# Train the model using sklearn wrapper
model = LGBMClassifier()
model.fit(X, y)

# Save the model using joblib
joblib.dump(model, "trained_model.pkl")
