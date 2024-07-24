import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
import joblib
import os
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = joblib.load(file)

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Retrieve features from the form
    features = {
        'manufactor': request.form['manufactor'],
        'Year': int(request.form['Year']),
        'model': request.form['model'],
        'Hand': request.form['Hand'],
        'Gear': request.form['Gear'],
        'capacity_Engine': float(request.form['capacity_Engine']),
        'Engine_type': request.form['Engine_type'],
        'Prev_ownership': request.form['Prev_ownership'],
        'Curr_ownership': request.form['Curr_ownership'],
        'Color': request.form['Color'],
        'Km': int(request.form['Km'])
    }

    # Convert features into a DataFrame
    feature_names = ['manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Color', 'Km']
    input_data = pd.DataFrame([features], columns=feature_names)

    # Apply the prepare_data function to the input data
    input_data = prepare_data(input_data)

    # Transform the input data using the preprocessor
    input_processed = preprocessor.transform(input_data)

    # Predict the price using the trained model
    predicted_price = model.predict(input_processed)[0]

    return render_template('index.html', prediction_text='Predicted Price: {:.2f}'.format(predicted_price))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
