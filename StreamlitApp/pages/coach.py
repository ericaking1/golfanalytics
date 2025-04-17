import streamlit as st
import pandas as pd
from langchain_deepseek import ChatDeepSeek
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error as mse
import os
from auth import get_user_shots

# Check if user is logged in
if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.warning("Please login to access this page.")
    st.stop()

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Virtual Golf Coach")
st.title("Virtual Golf Coach")

# ---- API Key Input ----
deepseek_api_key = st.text_input("DeepSeek API Key", type="password", help="Enter your DeepSeek API key to enable the virtual coach. You can get one at https://platform.deepseek.com/")

# ---- Load Research Data ----
def load_research_data():
    """Calculate research data directly from the CSV file."""
    try:
        # Read and preprocess data
        df = pd.read_csv('GGXY.csv')
        df.replace('-', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Prepare features and target
        X = df.drop(['Club', 'Carry'], axis=1).astype(float)
        y = df['Carry'].astype(float)
        
        # Train Random Forest for feature importance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=500)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Calculate feature importances
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Calculate optimal ranges using SVR
        optimal_ranges = {}
        for feature in X.columns:
            X_feature = X[feature].to_numpy().reshape(-1, 1)
            
            # Scale features
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_feature)
            y_train_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
            
            # Train SVR
            svr = SVR(kernel='rbf')
            svr.fit(X_train_scaled, y_train_scaled)
            
            # Find optimal range
            X_range = np.linspace(X_feature.min(), X_feature.max(), 50000).reshape(-1, 1)
            X_range_scaled = scaler_X.transform(X_range)
            y_range_pred_scaled = svr.predict(X_range_scaled)
            y_range_pred = scaler_y.inverse_transform(y_range_pred_scaled.reshape(-1, 1)).ravel()
            
            # Identify optimal ranges
            max_y = np.max(y_range_pred)
            threshold = 0.98 * max_y
            optimal_x_values = X_range[y_range_pred >= threshold].flatten()
            
            if len(optimal_x_values) > 0:
                optimal_ranges[feature] = (float(optimal_x_values[0]), float(optimal_x_values[-1]))
        
        return feature_importances, optimal_ranges
        
    except Exception as e:
        st.warning(f"Error calculating research data: {str(e)}")
        return None, None

# Load research data
feature_importances, optimal_ranges = load_research_data()

# ---- Load Golf Shot Data ----
shots = get_user_shots(st.session_state.user_id)
if shots:
    # Convert shots to DataFrame
    df = pd.DataFrame(shots, columns=[
        'id', 'user_id', 'Shot Type', 'Carry (yards)', 'Club Speed (MPH)',
        'Ball Speed (MPH)', 'Launch Angle (Deg)', 'Spin Rate (RPM)',
        'Face Angle (Deg)', 'Face to Path (Deg)', 'Club Path (Deg)',
        'Attack Angle (Deg)', 'Launch Direction (Deg)', 'timestamp'
    ])
    df = df.drop(['id', 'user_id', 'timestamp'], axis=1)
    st.success("Loaded user data from database! 🟢")
else:
    df = pd.DataFrame()
    st.warning("No shot data found! Please log some shots first.")

# ---- Process Shot Data ----
if not df.empty:
    # Store all shot data
    all_shots = df.to_dict('records')
    latest_shot = all_shots[-1]
    
    st.write("### Your Shot History")
    st.dataframe(df)
    
    # Calculate shot statistics
    shot_stats = {}
    for metric in optimal_ranges.keys():
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce')
            values = values.dropna()
            
            if not values.empty:
                # Calculate weighted average
                weights = np.exp(np.linspace(0, 1, len(values)))
                weights = weights / weights.sum()
                weighted_mean = np.average(values, weights=weights)
                
                # Calculate weighted standard deviation
                weighted_std = np.sqrt(np.average((values - weighted_mean)**2, weights=weights))
                
                # Determine recent trend
                recent_value = values.iloc[-1]
                if recent_value > weighted_mean + weighted_std:
                    recent_trend = "above average"
                elif recent_value < weighted_mean - weighted_std:
                    recent_trend = "below average"
                else:
                    recent_trend = "within average range"
                
                shot_stats[metric] = {
                    'mean': weighted_mean,
                    'std': weighted_std,
                    'min': values.min(),
                    'max': values.max(),
                    'count': len(values),
                    'recent_trend': recent_trend
                }
    
    # Generate feedback based on weighted statistics
    feedback = []
    for metric, (low, high) in optimal_ranges.items():
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce')
            values = values.dropna()
            
            if not values.empty:
                # Calculate weighted average
                weights = np.exp(np.linspace(0, 1, len(values)))
                weights = weights / weights.sum()
                weighted_mean = np.average(values, weights=weights)
                
                # Check if weighted mean is outside optimal range
                if weighted_mean < low:
                    feedback.append(f"- Your recent {metric} is **too low** ({weighted_mean:.1f}). Consider increasing it.")
                elif weighted_mean > high:
                    feedback.append(f"- Your recent {metric} is **too high** ({weighted_mean:.1f}). Consider adjusting.")
                
                # Check consistency (weighted standard deviation)
                weighted_std = np.sqrt(np.average((values - weighted_mean)**2, weights=weights))
                if weighted_std > (high - low) * 0.2:  # If standard deviation is more than 20% of the optimal range
                    feedback.append(f"- Your {metric} is **inconsistent** (std: {weighted_std:.1f}). Work on consistency.")
    
    # Add research-based insights with weighted averages
    if feature_importances is not None:
        feedback.append("\n--- Research-Based Insights ---")
        for _, row in feature_importances.iterrows():
            feature = row['Feature']
            if feature in df.columns:
                values = pd.to_numeric(df[feature], errors='coerce')
                values = values.dropna()
                
                if not values.empty:
                    # Calculate weighted average
                    weights = np.exp(np.linspace(0, 1, len(values)))
                    weights = weights / weights.sum()
                    weighted_mean = np.average(values, weights=weights)
                    
                    if feature in optimal_ranges:
                        research_low, research_high = optimal_ranges[feature]
                        if weighted_mean < research_low:
                            feedback.append(f"- Research shows your recent {feature} is below optimal range for maximizing carry distance.")
                        elif weighted_mean > research_high:
                            feedback.append(f"- Research shows your recent {feature} is above optimal range for maximizing carry distance.")
    
    feedback_text = "\n".join(feedback) if feedback else "Your shot data is within optimal ranges and shows good consistency."

else:
    all_shots = None
    latest_shot = None
    shot_stats = {}
    feedback_text = "No shot data available."

# ---- Function to Generate AI Coaching Response ----
def generate_response(input_text):
    if not deepseek_api_key:
        st.warning("Please enter a valid DeepSeek API key!", icon="⚠")
        return

    try:
        # Initialize DeepSeek Model
        model = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            api_key=deepseek_api_key,
        )

        # Test the API key with a simple request
        test_response = model.invoke([("system", "Test"), ("human", "Hello")])
        if not test_response:
            st.error("Invalid API key. Please check your DeepSeek API key and try again.")
            return

    except Exception as e:
        if "Authentication" in str(e) or "401" in str(e):
            st.error("Invalid API key. Please check your DeepSeek API key and try again.")
        else:
            st.error(f"Error connecting to DeepSeek API: {str(e)}")
        return

    # Golf knowledge for structured responses
    golf_knowledge = """
    - Ball Speed, Launch Angle, and Spin Rate are key factors for optimizing carry distance.
    - Face to Path and Attack Angle should be within ±2 degrees to reduce shot curvature.
    - A negative Attack Angle increases spin rate, reducing efficiency for long shots.
    - Club Path should remain within ±5 degrees for straighter shots.
    """

    # Add research findings to golf knowledge
    if feature_importances is not None:
        golf_knowledge += "\n--- Research Findings ---\n"
        golf_knowledge += "Feature importance analysis shows the following order of impact on carry distance:\n"
        for _, row in feature_importances.iterrows():
            golf_knowledge += f"- {row['Feature']}: {row['Importance']:.2f}\n"
        
        golf_knowledge += "\nOptimal ranges based on data analysis:\n"
        for feature, (low, high) in optimal_ranges.items():
            golf_knowledge += f"- {feature}: {low:.1f} to {high:.1f}\n"

    # Add user's shot statistics to the prompt
    user_stats = ""
    if all_shots is not None:
        user_stats = "\n--- User's Shot Statistics ---\n"
        for metric, stats in shot_stats.items():
            user_stats += f"{metric}:\n"
            user_stats += f"- Average: {stats['mean']:.1f}\n"
            user_stats += f"- Standard Deviation: {stats['std']:.1f}\n"
            user_stats += f"- Range: {stats['min']:.1f} to {stats['max']:.1f}\n"
            user_stats += f"- Number of Shots: {stats['count']}\n"
            user_stats += f"- Recent Trend: {stats['recent_trend']}\n\n"

    # Example prompts for few-shot learning
    example_prompts = """
    --- Example 1: Driver Analysis ---
    User: "How can I improve my driver distance?"
    Coach: "Based on your data, I notice your ball speed is averaging 145 mph, which is good, but your launch angle of 8.5° is too low for optimal distance. The research shows that for your ball speed, a launch angle between 12-15° would be ideal. Your spin rate is also a bit high at 2800 RPM, which is reducing your carry distance. To improve:
    1. Focus on increasing your launch angle by adjusting your tee height and ball position
    2. Work on reducing spin by ensuring your attack angle is more positive
    3. Your face-to-path consistency is good at ±1.5°, so maintain that while making these adjustments"

    --- Example 2: Iron Consistency ---
    User: "Why am I so inconsistent with my 7-iron?"
    Coach: "Looking at your 7-iron data, I see several areas affecting consistency:
    1. Your face-to-path variance is high (±3.5°), which explains the directional inconsistency
    2. The research shows that for irons, face-to-path should be within ±2° for consistent ball flight
    3. Your attack angle is varying between -2° and -6°, which is causing inconsistent spin rates
    To improve:
    1. Focus on maintaining a consistent face angle through impact
    2. Work on keeping your attack angle between -3° and -4° for more predictable ball flight
    3. Your ball speed consistency is good, so maintain that while working on these other aspects"

    --- Example 3: General Improvement ---
    User: "What should I focus on first to improve my game?"
    Coach: "Based on the research data and your shot statistics, here's your priority list:
    1. Face-to-Path: This has the highest impact on shot quality. Your current range of ±3° is too wide. Work on getting this within ±2°
    2. Launch Angle: Your average is 10° but varies significantly. The research shows optimal ranges between 12-15° for your club speed
    3. Spin Rate: Your current average of 2500 RPM is good, but the variance is high. Focus on consistent contact
    Start with face-to-path control, as this will have the biggest immediate impact on your game."
    """

    # Construct AI Prompt
    coaching_prompt = f"""
    You are a professional golf coach specializing in shot analysis.
    
    --- Golf Knowledge ---
    {golf_knowledge}

    --- User's Shot Data ---
    {user_stats}

    --- Example Coaching Responses ---
    {example_prompts}

    --- Key Areas for Improvement ---
    {feedback_text}

    --- User's Question ---
    {input_text}

    Provide an answer that is **specific, actionable, and data-driven**, following the style and structure of the example responses above. Your response should:
    1. Reference specific metrics from the user's data
    2. Compare their performance to the research findings
    3. Provide clear, actionable steps for improvement
    4. Prioritize the most impactful changes first
    5. Use a similar structure to the example responses
    """

    # Generate Response
    response = model.invoke([("system", "You are a golf coach with deep understanding of shot data and research findings."),
                             ("human", coaching_prompt)])
    
    # Clean up the response to show only the coach's message
    if response:
        # Convert response to string and remove common artifacts
        clean_response = str(response)
        
        # Remove content= prefix if present
        if clean_response.startswith("content="):
            clean_response = clean_response[8:]
        
        # Remove any quotes at the start and end
        clean_response = clean_response.strip('"\'')
        
        # Remove "Coach's Response:" prefix if present
        clean_response = clean_response.replace("Coach's Response:", "").strip()
        
        # Remove any additional metadata or notes at the end
        if "*(Note:" in clean_response:
            clean_response = clean_response.split("*(Note:")[0].strip()
        
        # Remove any additional_kwargs or response_metadata
        if "additional_kwargs" in clean_response:
            clean_response = clean_response.split("additional_kwargs")[0].strip()
        
        # Remove any trailing quotes or whitespace
        clean_response = clean_response.strip('"\' \n')
        
        # Convert \n to actual newlines
        clean_response = clean_response.replace('\\n', '\n')
        
        # Display the markdown content directly
        st.markdown(clean_response)

# ---- User Input Form ----
with st.form("coaching_form"):
    user_question = st.text_area("Ask your Virtual Coach:", "How can I improve my accuracy?")
    submitted = st.form_submit_button("Get Advice")

    if submitted and latest_shot:
        generate_response(user_question)
    elif submitted:
        st.warning("No shot data available for analysis.")