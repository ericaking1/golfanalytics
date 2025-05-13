import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from auth import get_user_shots

st.set_page_config(page_title="Golf Course Simulation", page_icon="⛳")
st.markdown("# Golf Course Simulation")

# Check if user is logged in
if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.warning("Please login to access this page.")
    st.stop()

# # Course selection
# course = st.selectbox(
#     "Select a Golf Course:",
#     ("Clifton Park", "Elkridge", "Pine Ridge"),
#     index=0,
# )

# # Hole selection
# hole_number = st.number_input("Select Hole Number:", min_value=1, max_value=18, step=1)

# Simulation parameters
num_simulations = 1
show_hazards = st.checkbox("Show Hazards", value=True)

# Course coordinates and hole layouts
course_data = {
    "Clifton Park": {
        "center": (39.32318903778765, -76.58700524418536),
        "holes": {
            1: {
                "tee": (39.32592099564656, -76.5843713165854),
                "green": (39.32237917682804, -76.58753409610676),
                "fairway": [
                    (39.3202193906142, -76.57967450138088),
                    (39.3212193906142, -76.58067450138088),
                    (39.3222193906142, -76.58567450138088),
                    (39.32318903778765, -76.58700524418536),
                    (39.3222193906142, -76.58567450138088),
                    (39.3212193906142, -76.58067450138088),
                    (39.3202193906142, -76.57967450138088)
                ],
                "hazards": [
                    {
                        "type": "bunker",
                        "coordinates": [
                            (39.32218903778765, -76.58600524418536),
                            (39.32228903778765, -76.58600524418536),
                            (39.32228903778765, -76.58610524418536),
                            (39.32218903778765, -76.58610524418536)
                        ]
                    },
                    {
                        "type": "water",
                        "coordinates": [
                            (39.32118903778765, -76.58300524418536),
                            (39.32138903778765, -76.58300524418536),
                            (39.32138903778765, -76.58320524418536),
                            (39.32118903778765, -76.58320524418536)
                        ]
                    }
                ]
            }
        }
    }
}

# Get user's historical shots
user_shots = get_user_shots(st.session_state.user_id)
if not user_shots:
    st.warning("No shot data available. Please log some shots first.")
    st.stop()

# Convert shots to DataFrame
shots_df = pd.DataFrame(user_shots, columns=[
    'id', 'user_id', 'Shot Type', 'Carry (yards)', 'Club Speed (MPH)',
    'Ball Speed (MPH)', 'Launch Angle (Deg)', 'Spin Rate (RPM)',
    'Face Angle (Deg)', 'Face to Path (Deg)', 'Club Path (Deg)',
    'Attack Angle (Deg)', 'Launch Direction (Deg)', 'timestamp'
])

# Convert numeric columns
numeric_columns = [
    'Carry (yards)', 'Club Speed (MPH)', 'Ball Speed (MPH)',
    'Launch Angle (Deg)', 'Spin Rate (RPM)', 'Face Angle (Deg)',
    'Face to Path (Deg)', 'Club Path (Deg)', 'Attack Angle (Deg)',
    'Launch Direction (Deg)'
]

for col in numeric_columns:
    shots_df[col] = pd.to_numeric(shots_df[col], errors='coerce')

# Prepare data for prediction
le_shot = LabelEncoder()
shots_df["shot_type_encoded"] = le_shot.fit_transform(shots_df["Shot Type"])

# Use all numeric columns for training
X = shots_df[numeric_columns]
y = shots_df["shot_type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_next_shot(distance, lat, lon, shot_number):
    # Force shot type based on distance and shot sequence
    if shot_number == 1:  # First shot must be a drive
        return "Drive"
    elif distance > 50:  # Mid-range
        return "Iron Shot" if distance > 100 else "Approach"
    elif distance > 20:  # Close to green
        return "Chip"
    else:  # On or near green
        return "Putt"

def calculate_shot_endpoint(start_lat, start_lon, distance, angle):
    # Convert distance from yards to degrees (approximate)
    distance_deg = distance * 0.00001
    end_lat = start_lat + (distance_deg * np.sin(np.radians(angle)))
    end_lon = start_lon + (distance_deg * np.cos(np.radians(angle)))
    return end_lat, end_lon

def simulate_round(hole_data):
    shots = []
    current_lat, current_lon = hole_data["tee"]
    green_lat, green_lon = hole_data["green"]
    
    # Track what shot number we're on
    shot_number = 1
    
    while True:
        # Calculate distance to hole
        distance = np.sqrt(
            (current_lat - green_lat)**2 + 
            (current_lon - green_lon)**2
        ) * 111000  # Convert to meters
        distance_yards = distance * 1.09361  # Convert to yards
        
        if distance_yards < 2:  # Hole reached (within 2 yards)
            break
            
        # Get shot type based on distance and shot sequence
        shot_type = predict_next_shot(distance_yards, current_lat, current_lon, shot_number)
        
        # Get average carry distance for this specific shot type
        shot_type_stats = shots_df[shots_df['Shot Type'] == shot_type]['Carry (yards)'].mean()
        carry_distance = float(shot_type_stats)
        
        # Adjust carry distance based on shot type
        if shot_type == "Putt":
            carry_distance = min(distance_yards, 20)  # Putts shouldn't go past the hole
        elif shot_type == "Chip":
            carry_distance = min(distance_yards, 50)
        
        # Calculate shot endpoint
        angle = np.arctan2(green_lat - current_lat, green_lon - current_lon)
        next_lat, next_lon = calculate_shot_endpoint(
            current_lat, current_lon, carry_distance, np.degrees(angle)
        )
        
        shots.append({
            "lat": current_lat,
            "lon": current_lon,
            "lat2": next_lat,
            "lon2": next_lon,
            "shot_type": shot_type,
            "shot_number": shot_number,
            "distance_to_hole": round(distance_yards, 1),
            "carry": round(carry_distance, 1)
        })
        
        current_lat, current_lon = next_lat, next_lon
        shot_number += 1
    
    return pd.DataFrame(shots)

# Get hole data (TEMPORARILY hole 1 at clifton)
hole_data = course_data["Clifton Park"]["holes"][1]

# Generate simulations
all_simulations = []
for _ in range(num_simulations):
    simulation = simulate_round(hole_data)
    all_simulations.append(simulation)

# Create layers for visualization
layers = [
    # Fairway layer
    pdk.Layer(
        "PolygonLayer",
        data=[{
            "coordinates": hole_data["fairway"],
            "color": [0, 128, 0, 100]
        }],
        get_polygon="coordinates",
        get_fill_color="color",
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
    ),
    # Green layer
    pdk.Layer(
        "PolygonLayer",
        data=[{
            "coordinates": [
                (hole_data["green"][0] - 0.00005, hole_data["green"][1] - 0.00005),
                (hole_data["green"][0] + 0.00005, hole_data["green"][1] - 0.00005),
                (hole_data["green"][0] + 0.00005, hole_data["green"][1] + 0.00005),
                (hole_data["green"][0] - 0.00005, hole_data["green"][1] + 0.00005)
            ],
            "color": [0, 255, 0, 100]
        }],
        get_polygon="coordinates",
        get_fill_color="color",
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
    )
]

# Add hazard layers if enabled
if show_hazards:
    for hazard in hole_data["hazards"]:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=[{
                    "coordinates": hazard["coordinates"],
                    "color": [139, 69, 19, 100] if hazard["type"] == "bunker" else [0, 0, 255, 100]
                }],
                get_polygon="coordinates",
                get_fill_color="color",
                get_line_color=[0, 0, 0],
                line_width_min_pixels=2,
            )
        )

# Add shot layers for each simulation
# Define colors for each shot type
shot_colors = {
    "Drive": [255, 0, 0, 160],      # Red
    "Iron Shot": [255, 165, 0, 160],  # Orange
    "Approach": [255, 255, 0, 160],   # Yellow
    "Chip": [0, 255, 0, 160],        # Green
    "Putt": [0, 0, 255, 160]         # Blue
}

for i, simulation in enumerate(all_simulations):
    # Add color column to the simulation dataframe
    simulation['color'] = simulation['shot_type'].map(shot_colors)
    
    layers.extend([
        pdk.Layer(
            "ArcLayer",
            data=simulation,
            get_source_position=["lon", "lat"],
            get_target_position=["lon2", "lat2"],
            get_source_color="color",
            get_target_color="color",
            auto_highlight=True,
            width_scale=0.0001,
            get_width="distance / 50",
            width_min_pixels=2,
            width_max_pixels=5,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=simulation,
            get_position=["lon2", "lat2"],
            get_color="color",
            get_radius=5,
        ),
        pdk.Layer(
            "TextLayer",
            data=simulation,
            get_position=["lon2", "lat2"],
            get_text="shot_type",
            get_color=[255, 255, 255, 200],  # White text for better visibility
            get_size=12,
            get_alignment_baseline="'bottom'",
        )
    ])

# Display the map
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        initial_view_state={
            "latitude": hole_data["tee"][0],
            "longitude": hole_data["tee"][1],
            "zoom": 18,
            "pitch": 60,
        },
        layers=layers,
    )
)

# Display simulation results
st.markdown("### Simulation Results")
for i, simulation in enumerate(all_simulations):
    # st.markdown(f"#### Simulation {i+1}")
    st.dataframe(simulation[['shot_type', 'distance_to_hole']], hide_index=True)
    st.write(f"Total Shots: {len(simulation)}")
