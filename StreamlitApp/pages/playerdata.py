import streamlit as st
import pandas as pd
from auth import get_user_shots, save_user_shot, delete_user_shot

st.set_page_config(page_title="Golf Shot Logger", page_icon="⛳")

# Check if user is logged in
if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.warning("Please login to access this page.")
    st.stop()

st.markdown("# Golf Shot Logger")

# Load user's shots from database
shots = get_user_shots(st.session_state.user_id)
if shots:
    # Convert shots to DataFrame
    df = pd.DataFrame(shots, columns=[
        'id', 'user_id', 'Shot Type', 'Carry (yards)', 'Club Speed (MPH)',
        'Ball Speed (MPH)', 'Launch Angle (Deg)', 'Spin Rate (RPM)',
        'Face Angle (Deg)', 'Face to Path (Deg)', 'Club Path (Deg)',
        'Attack Angle (Deg)', 'Launch Direction (Deg)', 'timestamp'
    ])
    df = df.drop(['user_id', 'timestamp'], axis=1)
    st.session_state["golf_data"] = df
else:
    st.session_state["golf_data"] = pd.DataFrame(columns=[ 
        "id",
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

# Add shot to database
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

    # Save shot to database
    save_user_shot(st.session_state.user_id, new_shot)
    
    # Update session state
    st.session_state["golf_data"] = pd.concat(
        [st.session_state["golf_data"], pd.DataFrame([new_shot])], ignore_index=True
    )

    st.success("Shot saved!")
    st.rerun()

st.markdown("### Saved Shots")

# Display shots with delete buttons
if not st.session_state["golf_data"].empty:
    # Create a copy of the DataFrame without the id column for display
    display_df = st.session_state["golf_data"].drop('id', axis=1)
    
    # Add delete buttons for each shot
    for idx, row in st.session_state["golf_data"].iterrows():
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.dataframe(display_df.iloc[[idx]], hide_index=True)
        with col2:
            if st.button("🗑️", key=f"delete_{idx}"):
                shot_id = row['id']
                delete_user_shot(shot_id)
                st.success("Shot deleted!")
                st.rerun()
else:
    st.info("No shots recorded yet. Add your first shot above!")

# Export to CSV
if not st.session_state["golf_data"].empty:
    csv_data = display_df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, file_name="golf_shots.csv", mime="text/csv")
