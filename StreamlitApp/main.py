import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error as mse
import math

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.sidebar.header('Golf Analytics Dashboard')
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

def distance(ball_speed, launch_angle, g=9.8):
    angle = math.radians(launch_angle)
    speed = ball_speed * 0.488889  # Convert to yards per second
    return (speed**2) * math.sin(2*angle) / g

def direction(face_angle, face_to_path):
    return face_angle + face_to_path

def final(dist, degrees):
    return dist / math.cos(math.radians(degrees))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace('-', np.nan, inplace=True)
    df.dropna(inplace=True)
    st.write("### Uploaded Data", df.head())
    
    X = df.drop(['Club', 'Carry (Yds)'], axis=1).astype(float)
    y = df['Carry (Yds)'].astype(float)
    
    # Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = mse(y_test, y_pred_rf)
    
    st.write("### Random Forest Results")
    st.write(f"R² Score: {r2_rf:.2f}, RMSE: {rmse_rf:.2f}")
    
    # Feature Importance
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    st.write("### Feature Importance in Random Forest")
    fig, ax = plt.subplots()
    ax.barh(feature_importances['Feature'], feature_importances['Importance'], color='lightsteelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance in Random Forest Regression')
    ax.invert_yaxis()
    st.pyplot(fig)
    
    # SVR Analysis
    r2scores = []
    for i in range(len(X.columns)):
        X_feature = X[X.columns[i]].to_numpy().reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=500)
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        svr = SVR(kernel='rbf')
        svr.fit(X_train_scaled, y_train_scaled) 
        
        y_pred_scaled = svr.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        r2 = r2_score(y_test, y_pred)
        r2scores.append(r2)
    
    scores = pd.DataFrame({'Feature': X.columns, 'R-squared': r2scores}).sort_values(by='R-squared', ascending=False)
    st.write("### Feature Importance in SVR")
    fig, ax = plt.subplots()
    ax.barh(scores['Feature'], scores['R-squared'], color='palevioletred')
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance in SVR')
    ax.invert_yaxis()
    st.pyplot(fig)

    # SVR Graphs
    st.write("### SVR Graphs")
    option = st.selectbox(
        "Select a feature:",
        (X.columns),
        index=0,
        placeholder="Select a feature...",
    )

    X_feature = X[option].to_numpy().reshape(-1, 1)
    X_range = np.linspace(X_feature.min(), X_feature.max(), 500).reshape(-1, 1)
    X_range_scaled = scaler_X.transform(X_range)
    y_range_pred_scaled = svr.predict(X_range_scaled)
    y_range_pred = scaler_y.inverse_transform(y_range_pred_scaled.reshape(-1, 1)).ravel()

    fig, ax = plt.subplots()
    ax.scatter(X_feature, y, color='darkslateblue', alpha=0.5, label='Actual Data')
    ax.plot(X_range, y_range_pred, color='palevioletred', lw=2, label='SVR Regression Curve')
    ax.set_xlabel(option)
    ax.set_ylabel('Carry (Yds)')
    ax.set_title(f'SVR on {option}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    
    # Heatmaps
    st.write("### Heatmaps")
    def create_heatmap(x, y, z, xlabel, ylabel, title):
        x = pd.to_numeric(x, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        z = pd.to_numeric(z, errors='coerce')

        valid_data = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
        
        heatmap_data, xedges, yedges = np.histogram2d(valid_data['x'], valid_data['y'], bins=50, weights=valid_data['z'])
        counts, _, _ = np.histogram2d(valid_data['x'], valid_data['y'], bins=50)
        heatmap_avg = np.divide(heatmap_data, counts, out=np.zeros_like(heatmap_data), where=counts != 0)
        
        fig, ax = plt.subplots()
        c = ax.imshow(heatmap_avg.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='BuPu')
        fig.colorbar(c, label='Carry (yards)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        st.pyplot(fig)

    
    options = st.multiselect(
        "Features for Heatmap",
        X.columns,
        ["Ball Speed (MPH)", "Launch Angle (Deg)"],
        max_selections=2,
    )
    
    create_heatmap(df[options[0]], df[options[1]], df['Carry (Yds)'], 'Ball Speed', 'Launch Angle', 'Ball Speed and Launch Angle')    


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
