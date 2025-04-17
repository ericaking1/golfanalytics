import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error as mse
import scipy
from auth import get_user_shots
import seaborn as sns

# Check if user is logged in
if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.warning("Please login to access this page.")
    st.stop()

# Initialize session state for research data if not exists
if 'research_data' not in st.session_state:
    st.session_state.research_data = {
        'optimal_ranges': {},
        'feature_importances': None,
        'r2': None,
        'r2_rf': None,
        'rmse_rf': None,
        'r2_svr': None,
        'rmse_svr': None
    }

df = pd.read_csv('GGXY.csv')
df.replace('-', np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(['Club', 'Carry'], axis=1).astype(float)
y = df['Carry'].astype(float)

# Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=500)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse(y_test, y_pred_rf))

st.write("### Random Forest Results")
st.write(f"R² Score: {r2_rf:.2f}, RMSE: {rmse_rf:.2f}")

# Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
st.session_state.research_data['feature_importances'] = feature_importances

st.write("### Feature Importance in Random Forest")
fig, ax = plt.subplots()
ax.barh(feature_importances['Feature'], feature_importances['Importance'], color='lightsteelblue')
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importance in Random Forest Regression')
ax.invert_yaxis()
st.pyplot(fig)
st.text('With 9 features on hand, we want to know which ones have the most impact on carry. This is why we need to use random forest to find the order of importance for all the features. This technique considers the total weight of each feature (AKA how much impact it has by itself and in conjunction with other features).')

# SVR Analysis
st.write("### SVR Graphs")
option = st.selectbox(
    "Select a feature:",
    (X.columns),
    index=0,
    placeholder="Select a feature...",
)

X_feature = X[option].to_numpy().reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=500)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Train SVR
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)

# Predictions
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Calculate SVR metrics
r2_svr = r2_score(y_test, y_pred)
rmse_svr = np.sqrt(mse(y_test, y_pred))

X_feature = X[option].to_numpy().reshape(-1, 1)
X_range = np.linspace(X_feature.min(), X_feature.max(), 50000).reshape(-1, 1)
X_range_scaled = scaler_X.transform(X_range)
y_range_pred_scaled = svr.predict(X_range_scaled)
y_range_pred = scaler_y.inverse_transform(y_range_pred_scaled.reshape(-1, 1)).ravel()

# r2 scores
r2 = r2_score(y_test,y_pred)
st.text(f'R-squared for {option}: {r2:.5f}')

# Identify the optimal X range that maximizes Y
optimal_ranges = []
max_y = np.max(y_range_pred) 
threshold = 0.98 * max_y  # threshold for "near max" regions

optimal_x_values = X_range[y_range_pred >= threshold].flatten()

discrete_ranges = []
start = optimal_x_values[0]

for j in range(1, len(optimal_x_values)):
    if optimal_x_values[j] - optimal_x_values[j - 1] > (X_range[1] - X_range[0]):
        discrete_ranges.append((start, optimal_x_values[j - 1]))
        start = optimal_x_values[j] 
discrete_ranges.append((start, optimal_x_values[-1])) 

# Limit to at most 2 optimal ranges based on predicted Y value
discrete_ranges = sorted(
    discrete_ranges, key=lambda r: np.mean(y_range_pred[(X_range.flatten() >= r[0]) & (X_range.flatten() <= r[1])]), reverse=True
)[:2]

fixed_width = 0.1 * (X_feature.max() - X_feature.min()) 

adjusted_ranges = []
for r in discrete_ranges:
    mid_point = (r[0] + r[1]) / 2 
    new_start = max(X_feature.min(), mid_point - fixed_width / 2)
    new_end = min(X_feature.max(), mid_point + fixed_width / 2)
    adjusted_ranges.append((float(new_start), float(new_end)))

optimal_ranges.append((option, adjusted_ranges))
# Update session state with the new optimal range
if adjusted_ranges:  # Only update if we found valid ranges
    st.session_state.research_data['optimal_ranges'][option] = adjusted_ranges[0]

st.text(f"Optimal X ranges for maximizing Y on {option}: {adjusted_ranges}")

fig, ax = plt.subplots()
ax.scatter(X_feature, y, color='darkslateblue', alpha=0.5, label='Actual Data')
ax.plot(X_range, y_range_pred, color='palevioletred', lw=2, label='SVR Regression Curve')
ax.axvspan(discrete_ranges[0][0], discrete_ranges[0][1], alpha=0.3, color='red')   
for r in adjusted_ranges:
    ax.axvspan(r[0], r[1], alpha=0.3, color='red')
    
ax.set_xlabel(option)
ax.set_ylabel('Carry (Yds)')
ax.set_title(f'SVR on {option}')
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.text('On the graph, we can see the regression line that tries to find the relationship between the feature and carry as a modelable function. The region shaded in red shows the optimla range of values for the feature to maximize feature. The optimal range corresponds with the range of values that the festure can take on in order to reflect what is realistic and feasible.')

# Heatmaps
st.write("### Heatmaps")
def create_heatmap(x, y, z, xlabel, ylabel, title):
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    z = pd.to_numeric(z, errors='coerce')

    valid_data = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    
    heatmap_data, xedges, yedges = np.histogram2d(valid_data['x'], valid_data['y'], bins=100, weights=valid_data['z'])
    counts, _, _ = np.histogram2d(valid_data['x'], valid_data['y'], bins=100)
    heatmap_avg = np.divide(heatmap_data, counts, out=np.zeros_like(heatmap_data), where=counts != 0)
    
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    mask = counts == 0
    known_x, known_y = X[~mask], Y[~mask]
    known_z = heatmap_avg[~mask]

    interp_func = scipy.interpolate.griddata(
        (known_x, known_y), known_z, (X, Y), method='linear', fill_value=0
    )
    
    heatmap_smooth = scipy.ndimage.gaussian_filter(interp_func, sigma=2.5)

    fig, ax = plt.subplots()
    c = ax.imshow(heatmap_smooth.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', interpolation="bilinear")
    fig.colorbar(c, label='Carry (yards)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)
    st.text('The heatmaps aim to show pairwise interaction between features. This is helpful for showing the tradeoffs or synergy between features - especially useful for directional features where takeoff and curvature in the shot are both important.')

options = st.multiselect(
    "Features for Heatmap",
    X.columns,
    ["Ball Speed", "Launch Angle"],
    max_selections=2,
)

create_heatmap(df[options[0]], df[options[1]], df['Carry'], options[0], options[1], f'Heatmap of {options[0]} and {options[1]}')    

# Optimal Ranges Visualization
st.write("### Optimal Ranges by Feature Interactions")

# Read and process the optimal ranges data
optimal_df = pd.read_csv('optimal_ranges_combo.csv')
optimal_df = optimal_df.set_index("Unnamed: 0")
labels = optimal_df.columns.tolist()
optimal_df = optimal_df.loc[labels, labels]
data = optimal_df.fillna("").values.tolist()

def get_range_width(cell):
    try:
        nums = eval(cell)
        return nums[1] - nums[0]
    except:
        return np.nan

value_matrix = np.array([
    [get_range_width(cell) for cell in row]
    for row in data
])

fig, ax = plt.subplots(figsize=(30, 20))
diag = np.diag(value_matrix)
diff_from_diag = np.abs(value_matrix - diag[np.newaxis, :])
percent_diff = np.abs(value_matrix - diag[np.newaxis, :]) / np.abs(diag[np.newaxis, :])

masked_array = np.ma.masked_invalid(percent_diff)

c = ax.imshow(masked_array, cmap='coolwarm')

for i in range(len(labels)):
    for j in range(len(labels)):
        if data[i][j] != "":
            if i == j:
                try:
                    nums = eval(data[i][j])
                    rounded_text = f"({nums[0]:.2f}, {nums[1]:.2f})"
                except:
                    rounded_text = data[i][j]
                ax.text(j, i, rounded_text, ha='center', va='center', color='white', fontsize=15, weight='bold')

                ax.add_patch(plt.Rectangle((i-.5, i-.5), 1, 1, fill=True, color='midnightblue', edgecolor='black'))
            else:
                try:
                    nums = eval(data[i][j])
                    rounded_text = f"({nums[0]:.2f}, {nums[1]:.2f})"
                except:
                    rounded_text = data[i][j]
                ax.text(j, i, rounded_text, ha='center', va='center', fontsize=15)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=30, rotation = 90)
ax.set_yticklabels(labels, fontsize=30)

cbar = fig.colorbar(c, ax=ax)
cbar.ax.tick_params(labelsize=20)  
cbar.set_label("Percentage Difference from Diagonal", fontsize = 20)  

st.pyplot(fig)
st.markdown("""
This matrix visualization shows the optimal ranges for different combinations of golf shot features. 
- The diagonal (dark blue) represents the optimal range for each individual feature
- The off-diagonal cells show the optimal ranges when considering pairs of features together
- The color intensity indicates the width of the optimal range (darker = wider range)
- This helps identify which feature combinations have the most flexibility in their optimal values
""")

# Correlation Between Column Feature and Row Feature 
st.write("### Correlation Between Column Feature and Row Feature ")

features = [
    "Club Speed", "Ball Speed", "Launch Angle", "Spin Rate", "Face Angle",
    "Face to Path", "Club Path", "Attack Angle", "Launch Direction"
]

df = pd.DataFrame(value_matrix, index=features, columns=features)

correlation_matrix = df.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
st.pyplot(plt)

# At the end, update the final research data
st.session_state.research_data.update({
    'feature_importances': feature_importances,
    'r2': r2,
    'r2_rf': r2_rf,
    'rmse_rf': rmse_rf,
    'r2_svr': r2_svr,
    'rmse_svr': rmse_svr
})