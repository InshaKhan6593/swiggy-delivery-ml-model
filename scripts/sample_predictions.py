import pandas as pd
import requests
from pathlib import Path

# path for data
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

# prediction endpoint
predict_url = "http://ec2-13-51-177-101.eu-north-1.compute.amazonaws.com/predict"

for i in range(3):

    # sample row for testing the endpoint
    sample_row = pd.read_csv(data_path).dropna().sample(1)
    print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
    # remove the target column
    data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()
    print(data)

    # get the response from API
    response = requests.post(url=predict_url,json=data)

    print("The status code for response is", response.status_code)

if response.status_code == 200:
    response_json = response.json()
    print("Full response JSON:", response_json)  # << Add this line to debug

    if "Predicted_Delivery_Time" in response_json:
        prediction = response_json["Predicted_Delivery_Time"]
        print(f"The prediction value by the API is {float(prediction):.2f} min")
    else:
        print("Prediction key missing in response:", response_json)
else:
    print("Error:", response.status_code)