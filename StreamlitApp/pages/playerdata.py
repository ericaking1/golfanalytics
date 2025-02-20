import streamlit as st
import pandas as pd

st.set_page_config(page_title="Golf Shot Logger", page_icon="⛳")

st.markdown("# Golf Shot Logger")

if "golf_data" not in st.session_state:
    st.session_state["golf_data"] = pd.DataFrame(columns=[ 
        "Shot Type", 
        "Carry (yards)", 
        "Club Speed (MPH)", 
        "Ball Speed (MPH)", 
        "Launch Angle (Deg)", 
        "Spin Rate (RPM)", 
        "Face Angle (Deg)", 
        "Face to Path (Deg)", 
        "Club Path (Deg)", 
        "Attack Angle (Deg)", 
        "Launch Direction (Deg)"
    ])

csv_file = st.file_uploader("Upload a Golf Shot Log (CSV)", type=["csv"])

if csv_file:
    st.session_state["golf_data"] = pd.read_csv(csv_file)

# User input fields
shot_type = st.selectbox("Shot Type:", ["Drive", "Iron Shot", "Approach", "Chip", "Putt"])
carry = st.number_input("Carry (yards):", min_value=1, max_value=400, step=1)
club_speed = st.number_input("Club Speed (MPH):", min_value=30, max_value=150, step=1)
ball_speed = st.number_input("Ball Speed (MPH):", min_value=30, max_value=220, step=1)
launch_angle = st.number_input("Launch Angle (Deg):", min_value=-10, max_value=45, step=1)
spin_rate = st.number_input("Spin Rate (RPM):", min_value=500, max_value=12000, step=100)
face_angle = st.number_input("Face Angle (Deg):", min_value=-10.0, max_value=10.0, step=0.1)
face_to_path = st.number_input("Face to Path (Deg):", min_value=-10.0, max_value=10.0, step=0.1)
club_path = st.number_input("Club Path (Deg):", min_value=-10.0, max_value=10.0, step=0.1)
attack_angle = st.number_input("Attack Angle (Deg):", min_value=-10.0, max_value=10.0, step=0.1)
launch_direction = st.number_input("Launch Direction (Deg):", min_value=-20.0, max_value=20.0, step=0.1)

# Add shot to DataFrame
if st.button("Add Shot"):
    new_shot = {
        "Shot Type": shot_type,
        "Carry (yards)": carry,
        "Club Speed (MPH)": club_speed,
        "Ball Speed (MPH)": ball_speed,
        "Launch Angle (Deg)": launch_angle,
        "Spin Rate (RPM)": spin_rate,
        "Face Angle (Deg)": face_angle,
        "Face to Path (Deg)": face_to_path,
        "Club Path (Deg)": club_path,
        "Attack Angle (Deg)": attack_angle,
        "Launch Direction (Deg)": launch_direction,
    }

    st.session_state["golf_data"] = pd.concat(
        [st.session_state["golf_data"], pd.DataFrame([new_shot])], ignore_index=True
    )

    st.success("Shot saved!")

st.markdown("### Saved Shots")
st.dataframe(st.session_state["golf_data"])

csv_data = st.session_state["golf_data"].to_csv(index=False)
st.download_button("Download CSV", csv_data, file_name="golf_shots.csv", mime="text/csv")
