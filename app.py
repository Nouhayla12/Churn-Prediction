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
    error = None
    form_submitted = False
    
    if request.method == 'POST':
        form_submitted = True
        try:
            # Add debug prints
            print("Form data:", request.form)
            
            # Get values from the form with validation
            credit_score = request.form.get('credit_score')
            if not credit_score:
                raise ValueError("Credit score is required")
            credit_score = float(credit_score)
            
            gender = 1 if request.form.get('gender') == 'Male' else 0
            
            age = request.form.get('age')
            if not age:
                raise ValueError("Age is required")
            age = float(age)
            
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
            
            print("Input data:", input_data)  # Debug print
            
            # Scale the input
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prob = model.predict(scaled_data)[0][0]
            prediction = "Will Leave" if prob > 0.5 else "Will Stay"
            probability = round(float(prob) * 100, 2)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")  # Debug print
            error = f"Error during prediction: {str(e)}"
            prediction = None
            probability = None
    
    return render_template('index.html', 
                         prediction=prediction, 
                         probability=probability,
                         error=error,
                         form_submitted=form_submitted)
