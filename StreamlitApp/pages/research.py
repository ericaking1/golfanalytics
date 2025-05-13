import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.mixture import GaussianMixture
import scipy
import seaborn as sns
from auth import get_user_shots

# Check if user is logged in
if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.warning("Please login to access this page.")
    st.stop()

# Get user's historical shots
user_shots = get_user_shots(st.session_state.user_id)
if not user_shots:
    st.warning("No shot data available. Please log some shots first.")
    st.stop()

# Convert user shots to DataFrame
user_df = pd.DataFrame(user_shots, columns=[
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
    user_df[col] = pd.to_numeric(user_df[col], errors='coerce')

# Load GGXY data
df = pd.read_csv('GGXY.csv')
df.replace('-', np.nan, inplace=True)
df.dropna(inplace=True)

# Add user shot selection
st.sidebar.markdown("### User Shot Selection")
show_user_shots = st.sidebar.checkbox("Show User Shots", value=True)

if show_user_shots:
    # Add timestamp to make each shot unique
    user_df['timestamp'] = pd.to_datetime(user_df['timestamp'])
    user_df['shot_id'] = user_df.apply(
        lambda row: f"{row['Shot Type']} - {row['timestamp'].strftime('%Y-%m-%d %H:%M')} - {row['Carry (yards)']:.0f}yds",
        axis=1
    )
    
    # Display shot selection interface
    st.sidebar.markdown("#### Select Shots to Display")
    
    # Group shots by type
    for shot_type in user_df['Shot Type'].unique():
        st.sidebar.markdown(f"**{shot_type}**")
        type_shots = user_df[user_df['Shot Type'] == shot_type]
        
        # Create a multiselect for each shot type
        selected_shots = st.sidebar.multiselect(
            f"Select {shot_type} shots",
            options=type_shots['shot_id'].tolist(),
            default=[],
            key=f"select_{shot_type}"
        )
        
        # Add selected shots to the filtered dataframe
        if selected_shots:
            selected_indices = type_shots[type_shots['shot_id'].isin(selected_shots)].index
            if 'filtered_user_df' not in locals():
                filtered_user_df = user_df.loc[selected_indices]
            else:
                filtered_user_df = pd.concat([filtered_user_df, user_df.loc[selected_indices]])
    
    # Use filtered dataframe if shots are selected, otherwise use empty dataframe
    if 'filtered_user_df' in locals():
        user_df = filtered_user_df
    else:
        user_df = pd.DataFrame(columns=user_df.columns)

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

# Add user shots if enabled and shots are selected
if show_user_shots and not user_df.empty:
    # Map column names from GGXY to user data format
    column_mapping = {
        'Ball Speed': 'Ball Speed (MPH)',
        'Launch Angle': 'Launch Angle (Deg)',
        'Spin Rate': 'Spin Rate (RPM)',
        'Face Angle': 'Face Angle (Deg)',
        'Face to Path': 'Face to Path (Deg)',
        'Club Path': 'Club Path (Deg)',
        'Attack Angle': 'Attack Angle (Deg)',
        'Launch Direction': 'Launch Direction (Deg)',
        'Club Speed': 'Club Speed (MPH)'
    }
    
    user_xlabel = column_mapping.get(option, option)
    if user_xlabel in user_df.columns:
        ax.scatter(user_df[user_xlabel], user_df['Carry (yards)'], 
                  c='red', s=100, alpha=0.7, label='Your Selected Shots')
    
ax.set_xlabel(option)
ax.set_ylabel('Carry (Yds)')
ax.set_title(f'SVR on {option}')
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.text('On the graph, we can see the regression line that tries to find the relationship between the feature and carry as a modelable function. The region shaded in red shows the optimal range of values for the feature to maximize carry. The optimal range corresponds with the range of values that the feature can take on in order to reflect what is realistic and feasible.')

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
    
    # Add user shots if enabled and shots are selected
    if show_user_shots and not user_df.empty:
        # Map column names from GGXY to user data format
        column_mapping = {
            'Ball Speed': 'Ball Speed (MPH)',
            'Launch Angle': 'Launch Angle (Deg)',
            'Spin Rate': 'Spin Rate (RPM)',
            'Face Angle': 'Face Angle (Deg)',
            'Face to Path': 'Face to Path (Deg)',
            'Club Path': 'Club Path (Deg)',
            'Attack Angle': 'Attack Angle (Deg)',
            'Launch Direction': 'Launch Direction (Deg)',
            'Club Speed': 'Club Speed (MPH)'
        }
        
        user_xlabel = column_mapping.get(xlabel, xlabel)
        user_ylabel = column_mapping.get(ylabel, ylabel)
        
        if user_xlabel in user_df.columns and user_ylabel in user_df.columns:
            ax.scatter(user_df[user_xlabel], user_df[user_ylabel], 
                      c='red', s=100, alpha=0.7, label='Your Selected Shots')
            ax.legend()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)
    st.text('The heatmaps aim to show pairwise interaction between features. This is helpful for showing the tradeoffs or synergy between features - especially useful for directional features where takeoff and curvature in the shot are both important.')


print(X.columns)
print(df)
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

# GMM Clustering Section
st.write("### Shot Type Clustering Analysis")

df = pd.read_csv('GGXY.csv')
df = df[(df['Launch Angle'] != '-') & (df['Face to Path'] != '-') & (df['Carry'] != '-')]
df['Launch Angle'] = df['Launch Angle'].astype(float)
df['Face to Path'] = df['Face to Path'].astype(float)
df['Carry'] = df['Carry'].astype(float)
df = df.dropna(subset=['Launch Angle', 'Face to Path', 'Carry'])

# Clean and prepare data for GMM
df_gmm = df.copy()
df_gmm = df_gmm[(df_gmm['Launch Angle'] != '-') & (df_gmm['Face to Path'] != '-') & (df_gmm['Carry'] != '-')]
df_gmm['Launch Angle'] = df_gmm['Launch Angle'].astype(float)
df_gmm['Face to Path'] = df_gmm['Face to Path'].astype(float)
df_gmm['Carry'] = df_gmm['Carry'].astype(float)
df_gmm = df_gmm.dropna(subset=['Launch Angle', 'Face to Path', 'Carry'])

# Use 3-cluster GMM on Launch Angle and Face to Path
X = df_gmm[['Launch Angle', 'Face to Path']]
gmm_contact = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
df_gmm['Contact Cluster'] = gmm_contact.fit_predict(X)

# Compute cluster means
cluster_means = df_gmm.groupby('Contact Cluster')['Launch Angle'].mean().sort_values()

# Map: lowest → Fat, middle → Square, highest → Thin
label_map = {
    cluster_means.index[0]: 'Fat',
    cluster_means.index[1]: 'Square',
    cluster_means.index[2]: 'Thin'
}
df_gmm['Contact Type'] = df_gmm['Contact Cluster'].map(label_map)

# Assign direction by Face to Path
def shot_direction(face_to_path):
    if face_to_path < -2:
        return 'Right'
    elif face_to_path > 2:
        return 'Left'
    else:
        return 'Straight'

df_gmm['Direction'] = df_gmm['Face to Path'].apply(shot_direction)

# Color mapping
color_map = {
    ('Thin', 'Right'): '#08306B',
    ('Thin', 'Straight'): '#4292C6',
    ('Thin', 'Left'): '#DEEBF7',
    
    ('Square', 'Right'): '#3E6B0F',
    ('Square', 'Straight'): '#66C24D',
    ('Square', 'Left'): '#CFEBC2',
    
    ('Fat', 'Right'): '#67000D',
    ('Fat', 'Straight'): '#FB6A4A',
    ('Fat', 'Left'): '#FEE0D2'
}

df_gmm['Color'] = df_gmm.apply(lambda row: color_map[(row['Contact Type'], row['Direction'])], axis=1)

# Marker map: shape by Contact Type
marker_map = {
    'Thin': '^',    # triangle up
    'Square': 's',  # square
    'Fat': 'v'      # triangle down
}

# Create the clustering visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each group with color and shape
for (contact, direction), color in color_map.items():
    marker = marker_map[contact]
    subset = df_gmm[(df_gmm['Contact Type'] == contact) & (df_gmm['Direction'] == direction)]
    plt.scatter(subset['Face to Path'], subset['Launch Angle'],
                color=color, edgecolor='k', s=80, marker=marker,
                label=f'{contact} - {direction}')

# Add user shots if enabled and shots are selected
if show_user_shots and not user_df.empty:
    # Predict clusters for user shots
    user_X = user_df[['Launch Angle (Deg)', 'Face to Path (Deg)']].rename(columns={
        'Launch Angle (Deg)': 'Launch Angle',
        'Face to Path (Deg)': 'Face to Path'
    })
    
    # Only predict if we have valid data
    if not user_X.empty and not user_X.isna().any().any():
        user_clusters = gmm_contact.predict(user_X)
        user_df['Contact Cluster'] = user_clusters
        user_df['Contact Type'] = user_df['Contact Cluster'].map(label_map)
        user_df['Direction'] = user_df['Face to Path (Deg)'].apply(shot_direction)
        user_df['Color'] = user_df.apply(lambda row: color_map[(row['Contact Type'], row['Direction'])], axis=1)
        
        # Plot user shots with larger markers and black edges
        for (contact, direction), color in color_map.items():
            marker = marker_map[contact]
            subset = user_df[(user_df['Contact Type'] == contact) & (user_df['Direction'] == direction)]
            if not subset.empty:
                plt.scatter(subset['Face to Path (Deg)'], subset['Launch Angle (Deg)'],
                           color=color, edgecolor='black', s=150, marker=marker,
                           label=f'Your {contact} - {direction} shots', alpha=0.8, linewidth=2)

# Build custom legend with both GGXY and user shots
legend_patches = []
# Add GGXY shots
for key in color_map:
    legend_patches.append(Patch(color=color_map[key], label=f'{key[0]} - {key[1]}'))
# Add user shots if present
if show_user_shots and not user_df.empty:
    for key in color_map:
        if any((user_df['Contact Type'] == key[0]) & (user_df['Direction'] == key[1])):
            legend_patches.append(Patch(color=color_map[key], label=f'Your Shots: {key[0]} - {key[1]}'))

plt.legend(handles=legend_patches, title='Shot Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add shaded hypothesis zone
ftp_center = (-1, 2)
la_center = (11, 13)
ftp_margin = (ftp_center[1] - ftp_center[0]) * 0.3
la_margin = (la_center[1] - la_center[0]) * 0.3
n_fades = 30
alphas = np.linspace(0.3, 0, n_fades)

for i in range(n_fades):
    x0 = ftp_center[0] - ftp_margin * (i + 1) / n_fades
    x1 = ftp_center[1] + ftp_margin * (i + 1) / n_fades
    ax.axvspan(x0, x1, color='gray', alpha=alphas[i], zorder=0)

for i in range(n_fades):
    y0 = la_center[0] - la_margin * (i + 1) / n_fades
    y1 = la_center[1] + la_margin * (i + 1) / n_fades
    ax.axhspan(y0, y1, color='gray', alpha=alphas[i], zorder=0)

plt.title("GMM Clustering of Shot Types (Contact - Direction)")
plt.xlabel("Face to Path (°)")
plt.ylabel("Launch Angle (°)")
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)

# Uncertainty Analysis
st.write("### Shot Classification Uncertainty Analysis")

# Get membership probabilities
probs = gmm_contact.predict_proba(X.values)

# Create probability DataFrame
prob_df = pd.DataFrame(probs, columns=['Cluster 0', 'Cluster 1', 'Cluster 2'])
prob_df['Launch Angle'] = df_gmm['Launch Angle']
prob_df['Face to Path'] = df_gmm['Face to Path']
prob_df['Assigned Cluster'] = gmm_contact.predict(X.values)
prob_df['Contact Type'] = prob_df['Assigned Cluster'].map(label_map)

# Add max probability and filter for uncertainty
prob_df['Max Prob'] = prob_df[['Cluster 0', 'Cluster 1', 'Cluster 2']].max(axis=1)
uncertain_df = prob_df[prob_df['Max Prob'] < 0.7].copy()

# Format probabilities as percentages
for col in ['Cluster 0', 'Cluster 1', 'Cluster 2']:
    uncertain_df[col] = (uncertain_df[col] * 100).round(0).astype(int).astype(str) + '%'

# Round numeric values
uncertain_df['Launch Angle'] = uncertain_df['Launch Angle'].round(2)
uncertain_df['Face to Path'] = uncertain_df['Face to Path'].round(2)
uncertain_df['Max Prob'] = (uncertain_df['Max Prob'] * 100).round(0).astype(int).astype(str) + '%'

# Display uncertain shots
st.write("Shots with classification uncertainty (probability < 70%):")
st.dataframe(uncertain_df.head(10))

# Display user shot statistics
if show_user_shots and not user_df.empty:
    st.write("### Your Shot Statistics")
    user_stats = user_df.groupby(['Contact Type', 'Direction']).agg({
        'Carry (yards)': ['mean', 'std', 'count'],
        'Launch Angle (Deg)': ['mean', 'std'],
        'Face to Path (Deg)': ['mean', 'std']
    }).round(2)
    st.dataframe(user_stats)
