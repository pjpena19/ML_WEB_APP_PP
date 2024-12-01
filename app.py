from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carga el modelo entrenado (reemplaza "mymodel.joblib" con el nombre de tu archivo)
model = joblib.load("mymodel.joblib")

# Define las características esperadas por el modelo (ajusta según tus datos)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Obtén los datos del formulario
        data = request.get_json()

        # Crea un DataFrame con las características en el orden correcto
        input_data = pd.DataFrame([data], columns=features)

        # Realiza la predicción
        prediction = model.predict(input_data)[0]

        # Prepara el mensaje de resultado
        if prediction == 1:
            result = "Alto riesgo de diabetes. Se recomienda consultar con un médico de inmediato."
        else:
            result = "Bajo riesgo de diabetes. Continúe manteniendo un estilo de vida saludable."

        return jsonify({'prediction': result})

    except Exception as e:
        print(f"Error durante la predicción: {e}")  # Registra el error para depuración
        return jsonify({'error': 'Ocurrió un error. Por favor, verifica los datos ingresados.'}), 500

if __name__ == '__main__':
    app.run(debug=True)