# pipeline/backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------
# Load model and scaler
# ------------------------------
MODEL_PATH = "Backend/house_price_model.pkl"
SCALER_PATH = "Backend/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Columns used during training
COLUMNS = [
    'UNDER_CONSTRUCTION','RERA','BHK_NO.','SQUARE_FT','READY_TO_MOVE','RESALE',
    'LONGITUDE','LATITUDE',
    'Bangalore','Chennai','Ghaziabad','Jaipur','Kolkata','Lalitpur','Maharashtra','Mumbai','Noida','Other','Pune',
    'Builder','Dealer','Owner'
]

# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess_input(user_input: dict):
    # Base numeric features
    row = {
        'UNDER_CONSTRUCTION': user_input.get('UNDER_CONSTRUCTION', 0),
        'RERA': user_input.get('RERA', 1),
        'BHK_NO.': user_input.get('BHK_NO', 3),
        'SQUARE_FT': user_input.get('SQUARE_FT', 1000),
        'READY_TO_MOVE': user_input.get('READY_TO_MOVE', 1),
        'RESALE': user_input.get('RESALE', 1),
        'LONGITUDE': user_input.get('LONGITUDE', 12.9716),
        'LATITUDE': user_input.get('LATITUDE', 77.5946)
    }

    # Initialize one-hot columns
    for col in COLUMNS[8:]:
        row[col] = 0

    # One-hot encode city
    city = user_input.get('city', 'Bangalore')
    row[city if city in COLUMNS[8:19] else 'Other'] = 1

    # One-hot encode seller
    seller = user_input.get('seller_type', 'Builder')
    row[seller if seller in COLUMNS[19:] else 'Owner'] = 1

    df = pd.DataFrame([row], columns=COLUMNS)
    return scaler.transform(df)

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="House Price Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------
# Request model
# ------------------------------
class HouseInput(BaseModel):
    BHK_NO: int
    SQUARE_FT: float
    city: str
    seller_type: str
    UNDER_CONSTRUCTION: int = 0
    RERA: int = 1
    READY_TO_MOVE: int = 1
    RESALE: int = 1
    LONGITUDE: float = 12.9716
    LATITUDE: float = 77.5946

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
def predict_price(user_input: HouseInput):
    processed_input = preprocess_input(user_input.dict())
    prediction = model.predict(processed_input)
    return {"predicted_price_lacs": float(prediction[0])}
