import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai.chat_models import ChatOpenAI

st.title("Virtual Golf Coach")

openai_api_key = st.text_input("OpenAI API Key", type="password")

optimal_ranges = {
    "Club Speed (MPH)": (80, 120),
    "Ball Speed (MPH)": (100, 180),
    "Launch Angle (Deg)": (10, 15),
    "Spin Rate (RPM)": (2000, 4000),
    "Face Angle (Deg)": (-2, 2),
    "Face to Path (Deg)": (-2, 2),
    "Club Path (Deg)": (-5, 5),
    "Attack Angle (Deg)": (-3, 3),
    "Launch Direction (Deg)": (-5, 5),
}

if "golf_data" in st.session_state and not st.session_state["golf_data"].empty:
    df = st.session_state["golf_data"]
    st.success("Loaded user data from session state! 🟢")
elif "golf_shots.csv":
    try:
        df = pd.read_csv("golf_shots.csv")
        st.info("Loaded user data from CSV.")
    except FileNotFoundError:
        df = pd.DataFrame()
        st.warning("No shot data found! Please log some shots first.")

if not df.empty:
    latest_shot = df.iloc[-1].to_dict()
    st.write("### Your Most Recent Shot Data:")
    st.json(latest_shot)

    feedback = []
    for metric, (low, high) in optimal_ranges.items():
        if metric in latest_shot:
            value = latest_shot[metric]
            if value < low:
                feedback.append(f"- Your {metric} is **too low** ({value}). Consider increasing it.")
            elif value > high:
                feedback.append(f"- Your {metric} is **too high** ({value}). Consider adjusting.")

    feedback_text = "\n".join(feedback) if feedback else "Your shot data is within optimal ranges."

else:
    latest_shot = None
    feedback_text = "No recent shots available."

def generate_response(input_text):
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter a valid OpenAI API key!", icon="⚠")
        return

    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)

    coaching_prompt = f"""
    You are a professional golf coach. A golfer has provided their most recent shot data.
    
    Their shot statistics:
    {latest_shot if latest_shot else "No data available"}

    Key areas for improvement:
    {feedback_text}

    Based on this, answer the golfer's question of: {input_text}
    """

    response = model.invoke(coaching_prompt)
    st.info(response)

with st.form("coaching_form"):
    user_question = st.text_area("Ask your Virtual Coach:", "How can I improve my accuracy?")
    submitted = st.form_submit_button("Get Advice")

    if submitted and latest_shot:
        generate_response(user_question)
    elif submitted:
        st.warning("No shot data available for analysis.")
