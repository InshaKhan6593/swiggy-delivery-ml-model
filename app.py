from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set sklearn output as pandas
set_config(transform_output='pandas')

# Initialize Dagshub tracking
import dagshub
import mlflow.client

try:
    dagshub.init(repo_owner='speedyskill', repo_name='swiggy-delivery-ml-model', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/speedyskill/swiggy-delivery-ml-model.mlflow")
    logger.info("Successfully initialized DagHub and MLflow")
except Exception as e:
    logger.error(f"Failed to initialize DagHub/MLflow: {e}")
    sys.exit(1)


class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


def load_model_information(file_path):
    try:
        with open(file_path) as f:
            run_info = json.load(f)
        logger.info(f"Successfully loaded model information from {file_path}")
        return run_info
    except FileNotFoundError:
        logger.error(f"Model information file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model information: {e}")
        raise


def load_transformer(transformer_path):
    try:
        transformer = joblib.load(transformer_path)
        logger.info(f"Successfully loaded transformer from {transformer_path}")
        return transformer
    except FileNotFoundError:
        logger.error(f"Transformer file not found: {transformer_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load transformer: {e}")
        raise


# Columns to preprocess
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]
nominal_cat_cols = ['weather', 'type_of_order', 'type_of_vehicle', "festival", "city_type", "is_weekend", "order_time_of_day"]
ordinal_cat_cols = ["traffic", "distance_type"]

# Initialize MLflow client
client = MlflowClient()

# Load model name from run info
logger.info("Loading model information...")
try:
    model_info = load_model_information("run_information.json")
    model_name = model_info['model_name']
    logger.info(f"Model name: {model_name}")
except Exception as e:
    logger.error(f"Failed to load model information: {e}")
    sys.exit(1)

# Define stage
stage = "Staging"

# Load model from MLflow registry
model_path = f"models:/{model_name}/{stage}"
logger.info(f"Loading model from: {model_path}")
try:
    model = mlflow.sklearn.load_model(model_path)
    logger.info("Successfully loaded model from MLflow registry")
except Exception as e:
    logger.error(f"Failed to load model from path '{model_path}': {e}")
    logger.error("This might be due to network issues, authentication, or the model not existing in the registry")
    sys.exit(1)

# Load preprocessor
preprocessor_path = "models/preprocessor.joblib"
logger.info(f"Loading preprocessor from: {preprocessor_path}")
try:
    preprocessor = load_transformer(preprocessor_path)
    logger.info("Successfully loaded preprocessor")
except Exception as e:
    logger.error(f"Failed to load preprocessor from path '{preprocessor_path}': {e}")
    sys.exit(1)

# Build the prediction pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ("regressor", model)
])
logger.info("Successfully built prediction pipeline")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

@app.post("/predict")
def do_predictions(data: Data):
    try:
        # Convert input to DataFrame
        pred_data = pd.DataFrame({
            'ID': data.ID,
            'Delivery_person_ID': data.Delivery_person_ID,
            'Delivery_person_Age': data.Delivery_person_Age,
            'Delivery_person_Ratings': data.Delivery_person_Ratings,
            'Restaurant_latitude': data.Restaurant_latitude,
            'Restaurant_longitude': data.Restaurant_longitude,
            'Delivery_location_latitude': data.Delivery_location_latitude,
            'Delivery_location_longitude': data.Delivery_location_longitude,
            'Order_Date': data.Order_Date,
            'Time_Orderd': data.Time_Orderd,
            'Time_Order_picked': data.Time_Order_picked,
            'Weatherconditions': data.Weatherconditions,
            'Road_traffic_density': data.Road_traffic_density,
            'Vehicle_condition': data.Vehicle_condition,
            'Type_of_order': data.Type_of_order,
            'Type_of_vehicle': data.Type_of_vehicle,
            'multiple_deliveries': data.multiple_deliveries,
            'Festival': data.Festival,
            'City': data.City
        }, index=[0])

        # Clean the raw input data
        cleaned_data = perform_data_cleaning(pred_data)

        # Get prediction
        prediction = model_pipe.predict(cleaned_data)[0]
        
        return {"Predicted_Delivery_Time": prediction}
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": f"Failed to generate prediction: {str(e)}"}


if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)