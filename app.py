import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(page_title="CA CRIME MAP", layout="wide")
st.title("📍 CA CRIME MAP")
st.markdown("Explore crime hotspots, top cities, and cluster patterns in California crime data.")

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def load_data():
    crime_url = "https://raw.githubusercontent.com/VictorKilanko/california-crime-dashboard/main/chapter1log.csv"
    df = pd.read_csv(crime_url)
    uscities = pd.read_csv("uscities.csv")
    ca_cities = uscities[uscities['state_name'] == 'California'][['city', 'county_name', 'lat', 'lng']]
    ca_cities.columns = ['City', 'County', 'Latitude', 'Longitude']
    per_capita_cols = [col for col in df.columns if col.endswith('_per_100k')]
    clearance_cols = [col for col in per_capita_cols if 'clr' in col]
    crime_cols = [col for col in per_capita_cols if col not in clearance_cols]

    meta_cols = ['County', 'City']
    filtered_df = df[meta_cols + per_capita_cols]
    city_crime_df = filtered_df.groupby(['County', 'City']).mean(numeric_only=True).reset_index()
    city_crime_df['TotalCrime_per_100k'] = city_crime_df[crime_cols].sum(axis=1)
    threshold = city_crime_df['TotalCrime_per_100k'].quantile(0.75)
    city_crime_df['Hotspot'] = (city_crime_df['TotalCrime_per_100k'] >= threshold).astype(int)

    # Merge coordinates
    city_crime_df['City'] = city_crime_df['City'].str.lower().str.strip()
    ca_cities['City'] = ca_cities['City'].str.lower().str.strip()
    merged_df = pd.merge(city_crime_df, ca_cities, on='City', how='left')
    merged_df = merged_df.dropna(subset=['Latitude', 'Longitude'])
    return merged_df, per_capita_cols, crime_cols

merged_df, per_capita_cols, crime_cols = load_data()

# Label map
crime_label_map = {
    "Violent_per_100k": "Violent Crime",
    "Property_per_100k": "Property Crime",
    "Homicide_per_100k": "Homicide",
    "Forrape_per_100k": "Forcible Rape",
    "Frobact_per_100k": "Robbery (Firearm)",
    "Violentclr_per_100k": "Violent Crime Clearance",
    "Robbery_per_100k": "Robbery",
    "Aggassault_per_100k": "Aggravated Assault",
    "Burglary_per_100k": "Burglary",
    "Fassact_per_100k": "Assault (Firearm)",
    "Propertyclr_per_100k": "Property Crime Clearance",
    "Vehicletheft_per_100k": "Vehicle Theft",
    "Lttotal_per_100k": "Larceny-Theft",
    "Arson_per_100k": "Arson"
}

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
page = st.sidebar.radio("Navigate", ["Hotspot Map", "Top 10 Cities", "Crime Clusters", "Crime Prediction Tool"])

# ---------------------------
# PAGE 1: MAP
# ---------------------------
if page == "Hotspot Map":
    st.subheader("🗺️ California Crime Hotspot Map")
    map_center = [36.5, -119.5]
    m = folium.Map(location=map_center, zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in merged_df.iterrows():
        color = 'red' if row['Hotspot'] else 'green'
        popup = f"""
        <b>City:</b> {row['City'].title()}, {row['County_x']}<br>
        <b>Total Crime Rate:</b> {row['TotalCrime_per_100k']:.1f}/100k<br>
        <b>Hotspot:</b> {'Yes' if row['Hotspot'] else 'No'}
        """
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=300),
            tooltip=row['City'].title()
        ).add_to(marker_cluster)

    st_folium(m, height=500, width=1000)

# ---------------------------
# PAGE 2: TOP 10 & BREAKDOWNS
# ---------------------------
elif page == "Top 10 Cities":
    st.subheader("🔥 Top 10 Hotspot Cities")
    hotspots = merged_df[merged_df['Hotspot'] == 1].sort_values(by='TotalCrime_per_100k', ascending=False).head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hotspots, x='City', y='TotalCrime_per_100k', palette='Reds_r', ax=ax1)
    ax1.set_title("Top 10 Hotspot Cities by Total Crime Rate")
    ax1.set_ylabel("Total Crimes per 100k")
    ax1.set_xlabel("City")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("🧊 Top 10 Safest Cities")
    safest = merged_df[merged_df['Hotspot'] == 0].sort_values(by='TotalCrime_per_100k', ascending=True).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=safest, x='City', y='TotalCrime_per_100k', palette='Greens', ax=ax2)
    ax2.set_title("Top 10 Safest Cities by Total Crime Rate")
    ax2.set_ylabel("Total Crimes per 100k")
    ax2.set_xlabel("City")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("🔎 Crime Breakdown for Hotspot Cities")
    city_options = hotspots['City'].str.title().unique().tolist()
    selected_city = st.selectbox("Select a Hotspot City", city_options)
    row = merged_df[merged_df['City'].str.title() == selected_city].iloc[0]
    breakdown = row[crime_cols].sort_values(ascending=False)

    renamed = [crime_label_map.get(col, col) for col in breakdown.index]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=breakdown.values, y=renamed, palette="Reds", ax=ax3)
    ax3.set_title(f"Crime Breakdown in {selected_city}")
    ax3.set_xlabel("Incidents per 100k")
    st.pyplot(fig3)

# ---------------------------
# PAGE 3: CLUSTER ANALYSIS
# ---------------------------
elif page == "Crime Clusters":
    st.subheader("🔬 Crime Pattern Clustering")

    features = merged_df[crime_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.markdown("We grouped cities into 4 clusters based on similar crime patterns using KMeans machine learning.")

    fig4, ax4 = plt.subplots()
    sns.countplot(x='Cluster', data=merged_df, palette='pastel', ax=ax4)
    ax4.set_title("Number of Cities per Crime Cluster")
    ax4.set_ylabel("City Count")
    st.pyplot(fig4)

    cluster_means = merged_df.groupby('Cluster')[crime_cols].mean().T
    cluster_means.index = [crime_label_map.get(col, col) for col in cluster_means.index]

    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt=".1f", ax=ax5)
    ax5.set_title("Average Crime Rate by Cluster")
    st.pyplot(fig5)

    st.subheader("🏙️ Cities in Selected Crime Cluster")
    selected_cluster = st.selectbox("Choose a cluster number", sorted(merged_df['Cluster'].unique()))
    cluster_cities = merged_df[merged_df['Cluster'] == selected_cluster][['City', 'County']]
    st.dataframe(cluster_cities.sort_values(by='City').reset_index(drop=True))

# ---------------------------
# PAGE 4: CRIME PREDICTION TOOL
# ---------------------------
elif page == "Crime Prediction Tool":
    st.title("🔮 Crime Prediction Tool")
    st.markdown("Use historical patterns to simulate and predict future crime or clearance levels in California cities.")
    
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/VictorKilanko/california-crime-dashboard/main/chapter1log.csv"
        df = pd.read_csv(url)
        df['City'] = df['City'].str.lower().str.strip()
        return df
    
    df = load_data()
    
    crime_label_map = {
        "Violent_per_100k": "Violent Crime",
        "Property_per_100k": "Property Crime",
        "Homicide_per_100k": "Homicide",
        "Forrape_per_100k": "Forcible Rape",
        "Frobact_per_100k": "Robbery (Firearm)",
        "Violentclr_per_100k": "Violent Crime Clearance",
        "Robbery_per_100k": "Robbery",
        "Aggassault_per_100k": "Aggravated Assault",
        "Burglary_per_100k": "Burglary",
        "Fassact_per_100k": "Assault (Firearm)",
        "Propertyclr_per_100k": "Property Crime Clearance",
        "Vehicletheft_per_100k": "Vehicle Theft",
        "Lttotal_per_100k": "Larceny-Theft",
        "Arson_per_100k": "Arson"
    }
    
    reverse_label_map = {v: k for k, v in crime_label_map.items()}
    crime_options = list(crime_label_map.values())
    
    st.subheader("🎯 Select a Target Crime Outcome to Predict")
    target_label = st.selectbox("Select Target Crime", crime_options)
    target_col = reverse_label_map[target_label]
    
    predictors = [col for col in crime_label_map.keys() if col != target_col]
    
    X = df[predictors].fillna(0)
    y = df[target_col].fillna(0)
    
    # Model training
    model = LinearRegression()
    model.fit(X, y)
    
    st.markdown("### 🎛️ Adjust Influencing Variables")
    
    user_inputs = {}
    col1, col2 = st.columns(2)
    for i, predictor in enumerate(predictors):
        readable = crime_label_map[predictor]
        col = col1 if i % 2 == 0 else col2
        min_val = float(df[predictor].min())
        max_val = float(df[predictor].max())
        default_val = float(df[predictor].mean())
        user_inputs[predictor] = col.slider(
            label=readable,
            min_value=round(min_val, 1),
            max_value=round(max_val, 1),
            value=round(default_val, 1),
            step=1.0
        )
    
    input_array = np.array([list(user_inputs.values())])
    prediction = model.predict(input_array)[0]
    
    st.markdown("---")
    st.subheader("📈 Predicted Value")
    st.metric(label=f"Estimated {target_label} per 100,000", value=f"{prediction:.1f}")
