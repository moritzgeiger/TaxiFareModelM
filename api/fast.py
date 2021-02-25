from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.externals import joblib
import pandas as pd

PATH_TO_LOCAL_MODEL = "model.joblib"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare")
def predict_fare(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    X_pred = pd.DataFrame({
                           "key": '2012-10-06 12:10:20.0000001',
                           "pickup_datetime": [float(pickup_datetime)],
                           "pickup_longitude": [float(pickup_longitude)],
                           "pickup_latitude": [float(pickup_latitude)], 
                           "dropoff_longitude": float(dropoff_longitude)], 
                           "dropoff_latitude": [float(dropoff_latitude)], 
                            "passenger_count": [int(passenger_count)]
                            })
    y_pred = pipeline.predict(X_pred)[0]
    return {
            "prediction": y_pred
            }