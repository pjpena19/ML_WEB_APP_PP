from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("mymodel.joblib")

# Define expected model features (adjust as needed)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  try:
    # Get data from JSON request
    data = request.get_json()

    # Create a DataFrame
    input_data = pd.