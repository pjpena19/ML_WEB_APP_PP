from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carga el modelo entrenado
model = joblib.load("mymodel.joblib")

# Define las columnas que espera el modelo (ajusta según tu dataset)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    return render_template('index.html') 1 

@app.route('/', methods=['POST'])
def predict():
    try:
        # Obtén los datos del formulario
        data = request.get_json()

        # Crea un DataFrame con los datos recibidos
        input_data = pd.DataFrame([data], columns=features)

        # Realiza la predicción
        prediction = model.predict(input_data)[0]

        # Prepara el mensaje de resultado
        if prediction == 1:
            result = "¡Alto riesgo de diabetes! Recomendamos consultar a un médico."
        else:
            result = "Bajo riesgo de diabetes. Sin embargo, es importante mantener un estilo de vida saludable."

        # Devuelve la predicción como un diccionario con un formato más claro
        return jsonify({'prediction': result})

    except Exception as e:
        # Maneja errores de manera más robusta
        return jsonify({'error': 'Ocurrió un error al realizar la predicción. Por favor, verifica los datos ingresados.'}), 500

if __name__ == '__main__':
    app.run(debug=True)