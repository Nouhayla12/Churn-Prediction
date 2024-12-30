from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
import pickle
import os

app = Flask(__name__)

try:
    # Load the trained model and scaler
    model = load_model("trained_model.keras")
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    error = None
    
    if request.method == 'POST':
        try:
            # Get values from the form
            credit_score = float(request.form.get('credit_score', 0))
            gender = 1 if request.form.get('gender') == 'Male' else 0
            age = float(request.form.get('age', 0))
            tenure = float(request.form.get('tenure', 0))
            balance = float(request.form.get('balance', 0))
            products_number = float(request.form.get('products_number', 0))
            has_card = 1 if request.form.get('has_card') == 'Yes' else 0
            is_active = 1 if request.form.get('is_active') == 'Yes' else 0
            salary = float(request.form.get('salary', 0))
            
            # Handle geography one-hot encoding
            geography = request.form.get('geography', 'France')
            geo_france = 1 if geography == 'France' else 0
            geo_germany = 1 if geography == 'Germany' else 0
            geo_spain = 1 if geography == 'Spain' else 0
            
            # Create input array
            input_data = [[credit_score, gender, age, tenure, balance, products_number, 
                          has_card, is_active, salary, geo_france, geo_germany, geo_spain]]
            
            # Validate model and scaler
            if model is None or scaler is None:
                raise Exception("Model or scaler not properly loaded")
            
            # Scale the input
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prob = model.predict(scaled_data)[0][0]
            prediction = "Will Leave" if prob > 0.5 else "Will Stay"
            probability = round(float(prob) * 100, 2)
            
        except Exception as e:
            error = f"Error during prediction: {str(e)}"
            print(error)
    
    return render_template('index.html', 
                         prediction=prediction, 
                         probability=probability,
                         error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
