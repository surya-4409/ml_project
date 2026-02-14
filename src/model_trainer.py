import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.data_processor import load_and_preprocess_data

def train_model(n_estimators=100, max_depth=None, run_name="Default_Run"):
    """
    Trains a Random Forest model, logs metrics/artifacts to MLflow, 
    and registers the model.
    """
    # 1. Set up MLflow
    # We use a local tracking URI for this phase. 
    # Docker will handle the server later, but this allows local testing.
    mlflow.set_tracking_uri("file:./experiments") 
    mlflow.set_experiment("BreastCancer_Classification")

    with mlflow.start_run(run_name=run_name):
        # 2. Load Data
        X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

        # 3. Log Hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForest")

        # 4. Train Model
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        clf.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 6. Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        print(f"Run '{run_name}' - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # 7. Log Artifacts (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Acc: {acc:.2f})")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 8. Log Artifacts (Scaler)
        # We need the scaler for the API later, so we save it as an artifact
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")

        # 9. Log Model
        # This saves the model in a format MLflow can serve/load later
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name="ClassificationModel",
            input_example=X_train[:1] # Optional: provides schema to MLflow
        )

if __name__ == "__main__":
    # We will simulate 3 runs with different parameters as requested
    print("Starting MLflow runs...")
    train_model(n_estimators=50, max_depth=3, run_name="Run_1_Simple")
    train_model(n_estimators=100, max_depth=10, run_name="Run_2_Complex")
    train_model(n_estimators=200, max_depth=None, run_name="Run_3_Full")
    print("Experiments completed. Check ./experiments folder.")