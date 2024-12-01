from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model (replace "mymodel.joblib" with your actual filename)
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
        input_data = pd.DataFrame([data], columns=features)

        # Perform prediction
        prediction = model.predict(input_data)[0]

        # Prepare informative prediction messages (adjust threshold as needed)
        if prediction == 1:
            result = "Alto riesgo de diabetes. Se recomienda consultar con un médico de inmediato."
        else:
            result = "Bajo riesgo de diabetes. Continúe manteniendo un estilo de vida saludable."

        return jsonify({'prediction': result})

    except Exception as e:
        # Handle errors gracefully (log or display a generic message)
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Ocurrió un error. Intente nuevamente.'}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production