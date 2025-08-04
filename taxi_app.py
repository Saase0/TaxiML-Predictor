import pandas as pd
import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI()

with open("taxi_price.pkl", "rb") as f:
    model, model_columns = pickle.load(f)

class Taxi_price(BaseModel):
    Trip_Distance_km: float
    Time_of_Day: str
    Day_of_Week: str
    Passenger_Count: float
    Traffic_Conditions: str
    Weather: str
    Base_Fare: float
    Per_Km_Rate: float
    Per_Minute_Rate: float
    Trip_Duration_Minutes: float

templates = Jinja2Templates(directory=".")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(features: Taxi_price):
    input_df = pd.DataFrame([features.dict()])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    return {"predicted_price": float(prediction)}

