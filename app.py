from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
import pickle

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = load_model("trained_model.keras")
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    
    if request.method == 'POST':
        try:
            # Get values from the form
            credit_score = float(request.form['credit_score'])
            gender = 1 if request.form['gender'] == 'Male' else 0
            age = float(request.form['age'])
            tenure = float(request.form['tenure'])
            balance = float(request.form['balance'])
            products_number = float(request.form['products_number'])
            has_card = 1 if request.form['has_card'] == 'Yes' else 0
            is_active = 1 if request.form['is_active'] == 'Yes' else 0
            salary = float(request.form['salary'])
            
            # Handle geography one-hot encoding
            geography = request.form['geography']
            geo_france = 1 if geography == 'France' else 0
            geo_germany = 1 if geography == 'Germany' else 0
            geo_spain = 1 if geography == 'Spain' else 0
            
            # Create input array
            input_data = [[credit_score, gender, age, tenure, balance, products_number, 
                          has_card, is_active, salary, geo_france, geo_germany, geo_spain]]
            
            # Scale the input
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prob = model.predict(scaled_data)[0][0]
            prediction = "Will Leave" if prob > 0.5 else "Will Stay"
            probability = round(float(prob) * 100, 2)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction = "Error in prediction"
            probability = 0
    
    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
