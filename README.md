
# End-to-End MLOps Pipeline: Breast Cancer Classification

## 📌 Project Overview
This project implements a production-ready Machine Learning Operations (MLOps) pipeline for classifying breast cancer tumors. It leverages **Docker** for containerization, **MLflow** for experiment tracking and model management, and **Flask** for serving real-time predictions.

## 🏗️ System Architecture

The project follows a decoupled Microservices architecture:

1.  **Training Service (`model_trainer.py`)**:
    * Runs inside a Docker container for environment consistency.
    * Fetches data, preprocesses it using `StandardScaler`, and trains a `RandomForestClassifier`.
    * Logs artifacts (model binary, scaler, confusion matrix) to the **MLflow Tracking Server**.

2.  **Model Registry (MLflow)**:
    * Acts as the central source of truth.
    * Stores versioned models (e.g., v1, v2, v6).
    * Allows the API to dynamically fetch specific model versions without rebuilding the image.

3.  **Inference Service (`inference_api.py`)**:
    * A **Flask** REST API running on **Gunicorn**.
    * Loads the production model (Version 6) and Scaler at startup.
    * Exposes endpoints for `/health` checks and `/predict` requests.

## 🚀 Key Features
* **Reproducible Environment:** Fully containerized training and inference services using Docker.
* **Experiment Tracking:** Logs model parameters, metrics (Accuracy, F1-Score), and artifacts to an MLflow server.
* **Model Registry:** Automatically versions and manages trained models.
* **Robust Serving:** A Flask API serves predictions with input validation and error handling.
* **Automated Testing:** Includes a suite of unit tests verified with `pytest`.

## 📂 Project Structure
```text
ml_project/
├── src/                  # Source code
│   ├── data_processor.py # Data loading and scaling
│   ├── model_trainer.py  # Model training and MLflow logging
│   └── inference_api.py  # Flask API for serving predictions
├── tests/                # Automated tests
│   └── test_api.py       # Unit tests for API endpoints
├── experiments/          # MLflow artifact store (models & metrics)
├── Dockerfile            # Definition for the Python environment
├── docker-compose.yml    # Orchestration for API and MLflow services
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```

## 🛠️ Setup & Installation

**Prerequisites:** Ensure you have Docker Desktop installed and running.

### 1. Build and Start the Services

Run the following command to build the Docker images and start the MLflow server and API:

```bash
docker-compose up --build -d

```

* **MLflow UI:** Access at [http://localhost:5000](https://www.google.com/search?q=http://localhost:5000)
* **API Health Check:** [http://localhost:8000/health](https://www.google.com/search?q=http://localhost:8000/health)

---

## 🏃‍♂️ Usage Guide

### 2. Train the Model

To ensure consistency across operating systems (Windows/Linux), training is executed **inside the container**. We run as `root` to ensure file permissions are handled correctly.

```bash
docker-compose exec -u root model_api python -m src.model_trainer

```

*This command will train the model, log metrics to MLflow, and register a new version of the `ClassificationModel`.*

### 3. Update Deployment

After training a new model version, update the `MODEL_VERSION` in `docker-compose.yml` to the latest version (e.g., `6`) and restart the API:

```bash
docker-compose restart model_api

```

### 4. Run Unit Tests

Verify the system integrity by running the test suite inside the container:

```bash
docker-compose exec -u root model_api python -m pytest tests/

```

### 5. Make a Prediction

You can test the API using the provided python script or `curl`.

**Option A: Using Python Script**

```bash
python test_request.py

```

**Option B: Using CURL**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]}'

```

---

## 📡 API Documentation

### **Health Check**

* **Endpoint:** `/health`
* **Method:** `GET`
* **Response:** `200 OK`
```json
{
  "status": "healthy",
  "model_version": "6"
}

```



### **Predict**

* **Endpoint:** `/predict`
* **Method:** `POST`
* **Payload:**
```json
{
  "features": [[...list of 30 numerical features...]]
}

```


* **Response:** `200 OK`
```json
{
  "predictions": [0]
}

```



## 🔧 Troubleshooting

**Permission Errors:**
If you encounter "Permission denied" errors when running tests or training, it is due to user conflicts between the host and container. Use the `-u root` flag in your docker commands as shown in the "Usage Guide" above.

**Path Errors (Windows):**
If Git Bash converts paths incorrectly (e.g., `C:/Program Files/Git/...`), prepend a double slash `//` to paths in your Docker commands.

```

```