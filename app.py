from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('titanic_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and convert to float
        features = [float(x) for x in request.form.values()]
        
        # Scale the features
        final_features = scaler.transform([features])
        
        # Predict survival
        prediction = model.predict(final_features)[0]
        
        result = "Survived ✅" if prediction == 1 else "Did Not Survive ❌"
        return render_template('index.html', prediction_text=f'Survival Prediction: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    # Required for deployment on Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
