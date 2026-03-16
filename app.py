from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# load model
model = joblib.load("model/churn_model.pkl")

# create feature template (same as training)
feature_columns = model.feature_names_in_

@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API Running"}

@app.post("/predict-churn")
def predict_churn(data: dict):

    # create empty dataframe with all features
    df = pd.DataFrame(columns=feature_columns)

    # add new row
    df.loc[0] = 0

    # fill provided values
    for key, value in data.items():
        if key in df.columns:
            df.at[0, key] = value

    # predict probability
    prob = model.predict_proba(df)[0][1]

    # risk category
    if prob > 0.7:
        risk = "High Risk"
    elif prob > 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "churn_probability": float(prob),
        "risk_category": risk
    }