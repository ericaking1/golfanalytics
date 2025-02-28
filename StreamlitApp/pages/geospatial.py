import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="Golf Shot Predictions", page_icon="⛳")
st.markdown("# Golf Shot Predictions")

option = st.selectbox(
    "Select a Golf Course:",
    ("Clifton Park", "Elkridge", "Pine Ridge"),
    index=0,
)

course_coords = {
    "Clifton Park": (39.3202193906142, -76.57967450138088),
    "Elkridge": (39.372440602200875, -76.63036689675737),
    "Pine Ridge": (39.444025295839374, -76.58027871650481),
}

st.write("Get your location:")
user_coords = streamlit_geolocation()
st.write(user_coords)

def generate_golf_data(n=10):
    shot_types = ["Drive", "Iron Shot", "Approach", "Chip", "Putt"]
    lat_start, lon_start = course_coords[option]
    shots = []
    
    for _ in range(n):
        lat_end = lat_start + random.uniform(-0.0005, 0.0005)
        lon_end = lon_start + random.uniform(-0.0005, 0.0005)
        shots.append({
            "lat": lat_start,
            "lon": lon_start,
            "lat2": lat_end,
            "lon2": lon_end,
            "shot_type": random.choice(shot_types),
            "distance": round(random.uniform(10, 300), 1),
        })
        lat_start, lon_start = lat_end, lon_end
    
    return pd.DataFrame(shots)

golf_data = generate_golf_data()

le_shot = LabelEncoder()
golf_data["shot_type_encoded"] = le_shot.fit_transform(golf_data["shot_type"])

X = golf_data[["distance", "lat", "lon"]]
y = golf_data["shot_type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_next_shot(distance, lat, lon):
    features = np.array([[distance, lat, lon]])
    prediction = model.predict(features)
    return le_shot.inverse_transform(prediction)[0]

latest_shot = golf_data.iloc[-1]
predicted_shot = predict_next_shot(latest_shot["distance"], latest_shot["lat2"], latest_shot["lon2"])

st.write(f"Predicted Next Shot: {predicted_shot}")

layers = [
    pdk.Layer(
        "ArcLayer",
        data=golf_data,
        get_source_position=["lon", "lat"],
        get_target_position=["lon2", "lat2"],
        get_source_color=[0, 200, 0, 160],
        get_target_color=[200, 0, 0, 160],
        auto_highlight=True,
        width_scale=0.0001,
        get_width="distance / 50",
        width_min_pixels=2,
        width_max_pixels=5,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=golf_data,
        get_position=["lon2", "lat2"],
        get_color=[255, 255, 0, 160],
        get_radius=5,
    ),
    pdk.Layer(
        "TextLayer",
        data=golf_data,
        get_position=["lon2", "lat2"],
        get_text="shot_type",
        get_color=[0, 0, 0, 200],
        get_size=12,
        get_alignment_baseline="'bottom'",
    )
]

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        initial_view_state={
            "latitude": course_coords[option][0],
            "longitude": course_coords[option][1],
            "zoom": 18,
            "pitch": 60,
        },
        layers=layers,
    )
)
