from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carga el modelo entrenado
model = joblib.load("mymodel.joblib")

# Define las columnas que espera el modelo (ajusta según tu dataset)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Obtén los datos del formulario
        data = request.get_json()

        # Crea un DataFrame con los datos recibidos
        input_data = pd.DataFrame([data], columns=features)

        # Realiza la predicción
        prediction = model.predict(input_data)[0]

        # Devuelve la predicción como un string
        if prediction == 1:
            result = "La persona tiene un alto riesgo de desarrollar diabetes."
        else:
            result = "La persona tiene un bajo riesgo de desarrollar diabetes."

        return jsonify({'prediction': result})

    except Exception as e:
        # Maneja errores de manera más robusta
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)