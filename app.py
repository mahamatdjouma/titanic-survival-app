from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('titanic_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = model.predict(final_features)[0]
    return render_template('index.html', prediction_text=f'Survival Prediction: {"Survived" if prediction==1 else "Did Not Survive"}')

if __name__ == "__main__":
    app.run(debug=True)
