from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carga el modelo entrenado
model = joblib.load("mymodel.joblib")

# Define las características esperadas por el modelo
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

app = Flask(__name__) # __name__ = app
model_class = {
    "0":"Bajo riesgo de diabetes. Continúe manteniendo un estilo de vida saludable.",
    "1":"Alto riesgo de diabetes. Se recomienda consultar con un médico de inmediato." 
}

@app.route("/", methods = ["GET", "POST"])
def predict():

    if request.method == "POST":
        # Obtén los datos del formulario

        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        val5 = float(request.form["val8"])
        val6 = float(request.form["val6"])
        val7 = float(request.form["val7"])
        val8 = float(request.form["val8"])

        # Crea un DataFrame con las características en el orden correcto
        my_df = pd.DataFrame([[val1, val2, val3,val5,val6,val6,val7,val8]],columns=['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Realiza la predicción
        prediction = str(model.predict(my_df)[0])
        pred_class = model_class[prediction]

    else:

        pred_class = None

    return render_template("index.html", prediction = pred_class)