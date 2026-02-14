import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Loads the Breast Cancer dataset, splits it into training and testing sets,
    and scales features using StandardScaler.

    Returns:
        X_train_scaled (ndarray): Scaled training features.
        X_test_scaled (ndarray): Scaled testing features.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
        scaler (StandardScaler): Fitted scaler object (needed for inference).
        feature_names (list): List of feature names for reference.
    """
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Split data
    # Stratify ensures the class distribution is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    # CRITICAL: Fit ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, data.feature_names

if __name__ == "__main__":
    # verification block
    X_tr, X_te, y_tr, y_te, s, f = load_and_preprocess_data()
    print(f"Data Loaded Successfully.")
    print(f"Training set shape: {X_tr.shape}")
    print(f"Test set shape: {X_te.shape}")