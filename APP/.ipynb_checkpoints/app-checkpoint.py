from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app) # This allows your AI website to talk to this Python script

# --- Load Models ---
print("Loading models...")
rfc_model = joblib.load('rfc_model.pkl')
le_encoder = joblib.load('label_encoder.pkl')
xgbr_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
autoencoder = load_model('lstm_autoencoder.h5')
SEQ_LENGTH = 24
print("Models loaded!")

# --- The Smart Alert Engine Logic ---
def run_smart_engine(data):
    # Extract data from the website's JSON
    # Note: In a real app, you'd parse the 24hr series data here. 
    # For this demo, we will simulate the series based on the current value 
    # to keep the input simple for the user.
    
    current_moisture = data.get('soil_moisture', 5)
    # Simulate a 24hr series based on current reading (Simplified for demo)
    soil_moisture_series = np.linspace(current_moisture, current_moisture, SEQ_LENGTH)
    
    # 1. Anomaly Detection
    scaled_series = scaler.transform(soil_moisture_series.reshape(-1, 1))
    sequence = scaled_series.reshape(1, SEQ_LENGTH, 1)
    reconstruction = autoencoder.predict(sequence, verbose=0)
    error = np.mean(np.abs(sequence - reconstruction))
    
    # Hardcoded threshold from your notebook results
    THRESHOLD = 0.107 
    
    if error > THRESHOLD:
        return {
            "type": "URGENT",
            "message": "🚨 URGENT ANOMALY DETECTED!",
            "details": f"Unusual soil moisture pattern detected. Error: {error:.3f}"
        }

    # 2. Contextual Warning
    nitrogen = data.get('soil_nitrogen_ppm', 0)
    if nitrogen < 15:
        return {
            "type": "WARNING",
            "message": "⚠️ CONTEXTUAL WARNING",
            "details": f"Nitrogen levels are critically low ({nitrogen} ppm). Yield may drop by ~20%."
        }

    # 3. Recommendation
    # Prepare input for ML models
    input_data = pd.DataFrame([{
        'soil_nitrogen_balance': data.get('soil_nitrogen_balance', 30),
        'avg_jan_temp_change': data.get('temp_change', 0.5),
        'pesticide_use_kg_ha': data.get('pesticide', 0.2),
        'nitrogen_fertilizer_tonnes': data.get('fertilizer', 100)
    }])
    
    crop_encoded = rfc_model.predict(input_data)[0]
    crop_name = le_encoder.inverse_transform([crop_encoded])[0]
    
    # Prepare regression input
    reg_input = input_data.copy()
    reg_input['crop_name_encoded'] = crop_encoded
    yield_pred = xgbr_model.predict(reg_input)[0]
    
    return {
        "type": "RECOMMENDATION",
        "message": "🌱 PROACTIVE RECOMMENDATION",
        "details": f"System Normal. Recommended Crop: {crop_name}. Predicted Yield: {yield_pred:.2f} hg/ha"
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = run_smart_engine(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)