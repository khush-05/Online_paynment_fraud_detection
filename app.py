from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib

model = tf.keras.models.load_model("fraud_ann_model.h5")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class Transaction(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int

@app.post("/predict")
def predict_fraud(tx: Transaction):
     input_data = np.array([[
        tx.step,
        tx.amount,
        tx.oldbalanceOrg,
        tx.newbalanceOrig,
        tx.oldbalanceDest,
        tx.newbalanceDest,
        tx.type_CASH_OUT,
        tx.type_DEBIT,
        tx.type_PAYMENT,
        tx.type_TRANSFER,
        tx.oldbalanceOrg - tx.newbalanceOrig,      # balance_diff_orig
        tx.newbalanceDest - tx.oldbalanceDest      # balance_diff_dest
    ]])

     input_scaled = scaler.transform(input_data)
     prediction = model.predict(input_scaled)[0][0]
     is_fraud = prediction >= 0.5    
     return {
        "fraud_probability": float(prediction),
        "is_fraud": bool(is_fraud)
    }
