import os
import mlflow.pyfunc
import pandas as pd
import joblib
import sys
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./experiments")
MODEL_NAME = os.getenv("MODEL_NAME", "ClassificationModel")
MODEL_VERSION = os.getenv("MODEL_VERSION", "3")

# Global variables
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    
    print(f"Connecting to MLflow at: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)
    
    try:
        # 1. Load the Model from Registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        print(f"Loading model from: {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 2. Load the Scaler Artifact
        client = mlflow.tracking.MlflowClient()
        # Get the run_id that generated this specific model version
        mv = client.get_model_version(MODEL_NAME, MODEL_VERSION)
        run_id = mv.run_id
        
        print(f"Downloading scaler from run_id: {run_id}...")
        scaler_path = client.download_artifacts(run_id, "preprocessing/scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        print("Model and Scaler loaded successfully!")
        
    except Exception as e:
        print(f"CRITICAL ERROR loading model/scaler: {e}")
        sys.stderr.write(f"Error: {e}\n")

# === CRITICAL FIX: LOAD GLOBALLY ===
# This ensures Gunicorn runs this automatically when it starts
load_model_and_scaler()

@app.route('/health', methods=['GET'])
def health():
    if model is not None and scaler is not None:
        return jsonify({"status": "healthy", "model_version": MODEL_VERSION}), 200
    return jsonify({"status": "unhealthy", "reason": "Model or Scaler not loaded"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
             return jsonify({"error": "Model/Scaler not loaded. Check server logs."}), 503

        # 1. Parse Input
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400
        
        # 2. Preprocess
        input_data = pd.DataFrame(data['features'])
        scaled_data = scaler.transform(input_data)
        
        # 3. Predict
        predictions = model.predict(scaled_data)
        
        # 4. Return Result
        return jsonify({"predictions": predictions.tolist()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)