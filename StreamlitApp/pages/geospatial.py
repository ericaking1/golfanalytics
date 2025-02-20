import streamlit as st
import pandas as pd
import pydeck as pdk
import random

st.set_page_config(page_title="Golf Shot Predictions", page_icon="⛳")

st.markdown("# Golf Shot Predictions")
option = st.selectbox(
    "Select a Golf Course:",
    (["Clifton Park", "Elkridge", "Pine Ridge"]),
    index=0,
    placeholder="Select a feature...",
)
course_coords = {"Clifton Park": (39.3202193906142, -76.57967450138088), "Elkridge": (39.372440602200875, -76.63036689675737), "Pine Ridge": (39.444025295839374, -76.58027871650481)}

def generate_golf_data():
    shots = []
    shot_types = ["Drive", "Iron Shot", "Approach", "Chip", "Putt"]
    lat_start, lon_start = course_coords[option]
    
    for i in range(10):
        lat_end = lat_start + random.uniform(-0.001, 0.001)
        lon_end = lon_start + random.uniform(-0.001, 0.001)
        shots.append({
            "lat": lat_start,
            "lon": lon_start,
            "lat2": lat_end,
            "lon2": lon_end,
            "shot_type": random.choice(shot_types),
            "distance": round(random.uniform(50, 100), 1)
        })
        lat_start, lon_start = lat_end, lon_end 
    
    return pd.DataFrame(shots)

golf_data = generate_golf_data()

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
        get_width="distance / 10",
        width_min_pixels=3,
        width_max_pixels=10,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=golf_data,
        get_position=["lon2", "lat2"],
        get_color=[255, 255, 0, 160], 
        get_radius=10,
    ),
    pdk.Layer(
        "TextLayer",
        data=golf_data,
        get_position=["lon2", "lat2"],
        get_text="shot_type",
        get_color=[0, 0, 0, 200],
        get_size=15,
        get_alignment_baseline="'bottom'",
    )
]

# Display map
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        initial_view_state={
            "latitude": course_coords[option][0],
            "longitude": course_coords[option][1], 
            "zoom": 15,
            "pitch": 50,
        },
        layers=layers,
    )
)
