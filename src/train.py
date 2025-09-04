import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle
import os
import mlflow
import mlflow.sklearn

def train(input_path: str, model_path: str, n_estimators=200):
    # Load processed data
    df = pd.read_csv(input_path)

    # Separate features and target automatically
    if "Target" not in df.columns:
        raise ValueError("Target column not found in processed dataset!")

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Sanity check
    assert len(X) == len(y), f"Length mismatch: X={len(X)}, y={len(y)}"

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Save model as artifact
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained and saved to {model_path}")
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train("data/Nvidia_stock_processed.csv", "models/model.pkl")
