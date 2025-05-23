from fastapi import FastAPI
from pydantic import BaseModel, validator, Field
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub
import dagshub
import mlflow.client

dagshub.init(repo_owner='speedyskill', repo_name='swiggy-delivery-ml-model', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/speedyskill/swiggy-delivery-ml-model.mlflow")


class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float = Field(..., description="Must be between 8.0 and 37.0 (India)")
    Restaurant_longitude: float = Field(..., description="Must be between 68.0 and 97.0 (India)")
    Delivery_location_latitude: float = Field(..., description="Must be between 8.0 and 37.0 (India)")
    Delivery_location_longitude: float = Field(..., description="Must be between 68.0 and 97.0 (India)")
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

    @validator('Restaurant_latitude', 'Delivery_location_latitude')
    def validate_latitude(cls, v):
        if not (8.0 <= v <= 37.0):
            raise ValueError(f"Latitude must be between 8.0 and 37.0 for India (got {v})")
        return v

    @validator('Restaurant_longitude', 'Delivery_location_longitude')
    def validate_longitude(cls, v):
        if not (68.0 <= v <= 97.0):
            raise ValueError(f"Longitude must be between 68.0 and 97.0 for India (got {v})")
        return v

    
    
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer


from scripts.data_clean_utils import (
    change_column_names, data_cleaning, clean_lat_long,
    calculate_haversine_distance, create_distance_type, drop_columns, columns_to_drop
)

def perform_data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Modified version of perform_data_cleaning that returns the cleaned DataFrame
    instead of saving it to a file
    """
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns, columns=columns_to_drop)
    )
    
    return cleaned_data


# columns to preprocess in data
num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

#mlflow client
client = MlflowClient()

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

# stage of the model
stage = "Production"

# get the latest model version
latest_model_ver = client.get_latest_versions(name=model_name,stages=[stage])[0].version

# load model path
model_path = f"models:/{model_name}/{latest_model_ver}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_uri=model_path)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
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
        },index=[0]
    )

    
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return {
    "prediction": round(predictions, 2),
    "distance": round(float(cleaned_data["distance"].iloc[0]), 2)
    }

   
   
if __name__ == "__main__":
    uvicorn.run(app="app:app",host='0.0.0.0',port=8000)