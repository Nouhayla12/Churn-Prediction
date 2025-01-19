from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model("trained_model.keras")
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from the form
    data = request.form.to_dict()
    
    # Convert data to the correct format
    features = [
        float(data["CreditScore"]),
        int(data["Gender"] == "Male"),
        float(data["Age"]),
        float(data["Tenure"]),
        float(data["Balance"]),
        float(data["NumOfProducts"]),
        int(data["HasCrCard"] == "Yes"),
        int(data["IsActiveMember"] == "Yes"),
        float(data["EstimatedSalary"]),
        int(data["Geography"] == "France"),
        int(data["Geography"] == "Germany"),
        int(data["Geography"] == "Spain"),
    ]
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    result = "Customer will leave the bank in the future" if prediction > 0.5 else "Customer will not leave the bank in the future"
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
