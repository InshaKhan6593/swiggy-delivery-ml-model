import streamlit as st
from streamlit_folium import st_folium
import folium
import overpy
import requests
import json
from datetime import datetime
from decimal import Decimal
import pandas as pd

# Custom JSON encoder to handle Decimal objects
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# API endpoint - configurable
API_URL = st.sidebar.text_input("API Endpoint", "http://localhost:8000/predict")

# Initialize Overpass API
overpass_api = overpy.Overpass()

# Fetch restaurants from OpenStreetMap (Mumbai only)
@st.cache_data(show_spinner=False)
def fetch_restaurants():
    try:
        query = """
        [out:json][timeout:25];
        node["amenity"="restaurant"](18.88,72.77,19.30,73.05);  // Mumbai bounding box
        out body;
        """
        result = overpass_api.query(query)
        return [
            {
                "name": node.tags.get("name", "Unnamed Restaurant"),
                "lat": node.lat,
                "lon": node.lon
            }
            for node in result.nodes
        ][:100]
    except overpy.exception.OverpassRuntimeError:
        st.error("Overpass API timed out. Please try again later.")
        return []
    except Exception as e:
        st.error(f"Failed to load restaurants: {e}")
        return []

# Validate pickup time
def validate_pickup_time(order_time, pickup_time):
    if order_time is None or pickup_time is None:
        st.error("Please provide both Order Time and Pickup Time.")
        return False

    order_dt = datetime.combine(datetime.today(), order_time)
    pickup_dt = datetime.combine(datetime.today(), pickup_time)

    # Allow next-day pickup (after midnight)
    if pickup_dt < order_dt:
        pickup_dt += pd.Timedelta(days=1)

    return True

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "select"
if "selected_restaurant" not in st.session_state:
    st.session_state.selected_restaurant = None
if "delivery_coords" not in st.session_state:
    st.session_state.delivery_coords = None
if "order_time" not in st.session_state:
    st.session_state.order_time = datetime.now().time()
if "pickup_time" not in st.session_state:
    st.session_state.pickup_time = datetime.now().time()

# Main navigation
if st.session_state.page == "select":
    st.title("Select a Restaurant")
    restaurants = fetch_restaurants()
    for idx, r in enumerate(restaurants):
        if st.button(r["name"], key=f"restaurant_{idx}"):
            st.session_state.selected_restaurant = r
            st.session_state.page = "order"
            st.rerun()

elif st.session_state.page == "order":
    r = st.session_state.selected_restaurant
    st.title("Place Order")
    st.markdown(f"*Restaurant:* {r['name']}")

    # Delivery map
    m = folium.Map(location=[r["lat"], r["lon"]], zoom_start=13)
    folium.Marker(
        location=[r["lat"], r["lon"]],
        tooltip=r["name"],
        icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
    ).add_to(m)

    if st.session_state.delivery_coords:
        folium.Marker(
            location=st.session_state.delivery_coords,
            tooltip="Delivery Location",
            icon=folium.Icon(color="blue")
        ).add_to(m)

    map_state = st_folium(m, height=500, width=700)

    if map_state and map_state.get("last_clicked"):
        st.session_state.delivery_coords = (
            map_state["last_clicked"]["lat"],
            map_state["last_clicked"]["lng"]
        )
        st.success(f"Selected Delivery Location: {st.session_state.delivery_coords}")

    # Form
    st.markdown("### Order Details")
    with st.form("order_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input("Delivery Person Age", "30")
            ratings = st.text_input("Delivery Person Ratings", "4.5")
            weather = st.selectbox("Weather Conditions", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog"])
            traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])

        with col2:
            order_type = st.selectbox("Type of Order", ["Snack", "Meal", "Drinks", "Buffet"])
            vehicle = st.selectbox("Type of Vehicle", ["bike", "scooter"])
            vehicle_condition = st.slider("Vehicle Condition", 0, 3, 2)
            multiple_deliveries = st.selectbox("Multiple Deliveries", ["0", "1", "2", "3"])

        with col3:
            festival = st.selectbox("Festival", ["No", "Yes"])
            city = st.selectbox("City", ["Urban", "Semi-Urban", "Metropolitian"])

            order_time = st.time_input("Time Ordered",
                                       value=st.session_state.get("order_time", datetime.now().time()),
                                       key="order_time")
            pickup_time = st.time_input("Time Picked",
                                        value=st.session_state.get("pickup_time", datetime.now().time()),
                                        key="pickup_time")

        submit = st.form_submit_button("Predict Delivery Time")

    if submit:
        if not validate_pickup_time(order_time, pickup_time):
            st.stop()  # Prevent further execution

        if not st.session_state.delivery_coords:
            st.error("Please select a delivery location on the map.")
            st.stop()

        # Payload
        payload = {
            "ID": "TEST123",
            "Delivery_person_ID": "DP001",
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": ratings,
            "Restaurant_latitude": r["lat"],
            "Restaurant_longitude": r["lon"],
            "Delivery_location_latitude": st.session_state.delivery_coords[0],
            "Delivery_location_longitude": st.session_state.delivery_coords[1],
            "Order_Date": datetime.today().strftime('%d-%m-%Y'),
            "Time_Orderd": order_time.strftime('%H:%M'),
            "Time_Order_picked": pickup_time.strftime('%H:%M'),
            "Weatherconditions": weather,
            "Road_traffic_density": traffic,
            "Vehicle_condition": vehicle_condition,
            "Type_of_order": order_type,
            "Type_of_vehicle": vehicle,
            "multiple_deliveries": multiple_deliveries,
            "Festival": festival,
            "City": city
        }

        # API call
        try:
            headers = {"Content-Type": "application/json"}
            json_payload = json.dumps(payload, cls=DecimalEncoder)
            res = requests.post(API_URL, data=json_payload, headers=headers)

            try:
                result = res.json()
                if isinstance(result, dict) and "prediction" in result and "distance" in result:
                    prediction = float(result["prediction"])
                    distance = float(result["distance"])
                    st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")
                    st.info(f"Delivery Distance: {distance:.2f} km")
                else:
                    st.error("API response format is invalid or incomplete.")
            except json.JSONDecodeError:
                st.error(f"Invalid JSON response: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

    st.markdown("---")
    if st.button("Back to Restaurant List"):
        st.session_state.page = "select"
        st.session_state.delivery_coords = None
        st.rerun()