from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


# Load model and preprocessor
model = joblib.load("RF_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Define request schema using Pydantic
class ClientData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend's Render URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/")
def predict(data: ClientData):
    input_df = pd.DataFrame([data.dict()])
    
    # Feature engineering (as done in training)
    month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                   'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    input_df['month'] = input_df['month'].map(month_order)
    input_df['contacted_before'] = input_df['pdays'].apply(lambda x: 0 if x == -1 else 1)

    # Transform with preprocessor
    input_processed = preprocessor.transform(input_df)
    
    # Predict
    prediction = model.predict(input_processed)[0]
    proba = model.predict_proba(input_processed)[0, 1]

    return {
        "subscribed": bool(prediction),
        "probability": round(proba, 4)
    }