import streamlit as st
import math

def distance(ball_speed, launch_angle, g=9.8):
    angle = math.radians(launch_angle)
    speed = ball_speed * 0.488889  # Convert to yards per second
    return (speed**2) * math.sin(2*angle) / g

def direction(face_angle, face_to_path):
    return face_angle + face_to_path

def final(dist, degrees):
    return dist / math.cos(math.radians(degrees))

# Prediction Example
st.subheader("Predict Carry Distance")
ball_speed = st.slider("Ball Speed (MPH)", 100, 200, 150)
launch_angle = st.slider("Launch Angle (Deg)", 0, 20, 10)
face_angle = st.slider("Face Angle (Deg)", -5, 5, 0)
face_to_path = st.slider("Face to Path (Deg)", -5, 5, 0)

pred_dist = distance(ball_speed, launch_angle)
pred_dir = direction(face_angle, face_to_path)
pred_final = final(pred_dist, pred_dir)
st.write(f"Predicted Carry: {pred_final:.2f} yards")
