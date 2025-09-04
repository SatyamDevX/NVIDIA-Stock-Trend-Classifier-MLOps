import pandas as pd
import pickle
import shap
import mlflow
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("data/Nvidia_stock_processed.csv")
X = df.drop(columns=["Target"])
y = df["Target"]

# Load model
model_path = "models/model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("reports/shap_summary.png")

# Optional: log in MLflow
with mlflow.start_run():
    mlflow.log_artifact("reports/shap_summary.png")
