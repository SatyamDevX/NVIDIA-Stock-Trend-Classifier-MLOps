from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

# Load trained model
model_path = "models/model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": pred.tolist()}
