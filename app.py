import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
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
    clearance_cols = [col for col in per_capita_cols if 'Clr' in col]
    crime_cols = [col for col in per_capita_cols if col not in clearance_cols]

    meta_cols = ['County', 'City']
    filtered_df = df[meta_cols + per_capita_cols]
    city_crime_df = filtered_df.groupby(['County', 'City']).mean(numeric_only=True).reset_index()
    city_crime_df['TotalCrime_per_100k'] = city_crime_df[crime_cols].sum(axis=1)
    threshold = city_crime_df['TotalCrime_per_100k'].quantile(0.75)
    city_crime_df['Hotspot'] = (city_crime_df['TotalCrime_per_100k'] >= threshold).astype(int)

    city_crime_df['City'] = city_crime_df['City'].str.lower().str.strip()
    ca_cities['City'] = ca_cities['City'].str.lower().str.strip()
    merged_df = pd.merge(city_crime_df, ca_cities, on='City', how='left')
    merged_df = merged_df.dropna(subset=['Latitude', 'Longitude'])
    return merged_df, per_capita_cols, crime_cols

merged_df, per_capita_cols, crime_cols = load_data()

crime_label_map = {
    "Violent_per_100k": "Violent Crime",
    "Property_per_100k": "Property Crime",
    "Homicide_per_100k": "Homicide",
    "ForRape_per_100k": "Forcible Rape",
    "FROBact_per_100k": "Robbery (Firearm)",
    "ViolentClr_per_100k": "Violent Crime Clearance",
    "Robbery_per_100k": "Robbery",
    "AggAssault_per_100k": "Aggravated Assault",
    "Burglary_per_100k": "Burglary",
    "FASSact_per_100k": "Assault (Firearm)",
    "PropertyClr_per_100k": "Property Crime Clearance",
    "VehicleTheft_per_100k": "Vehicle Theft",
    "LTtotal_per_100k": "Larceny-Theft",
    "Arson_per_100k": "Arson"
}

page = st.sidebar.radio("Navigate", [
    "Hotspot Map", 
    "Top 10 Cities", 
    "Crime Clusters", 
    "Crime Prediction Tool", 
    "LA Violent Crime Prediction"  # 👈 Add this
])


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

    st.subheader("🧫 Top 10 Safest Cities")
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
    st.subheader("🏩 Cities in Selected Crime Cluster")
    selected_cluster = st.selectbox("Choose a cluster number", sorted(merged_df['Cluster'].unique()))
    cluster_cities = merged_df[merged_df['Cluster'] == selected_cluster][['City', 'County']]
    st.dataframe(cluster_cities.sort_values(by='City').reset_index(drop=True))


# ---------------------------
# PAGE 4: CRIME PREDICTION TOOL (Updated with Correlation-Based Feature Selection)
# ---------------------------
elif page == "Crime Prediction Tool":
    st.title("🔮 Crime Prediction Tool")
    st.markdown("Use machine learning to predict city-specific crime or clearance rates based on demographic and economic indicators. Adjust inputs to simulate different conditions.")

    @st.cache_data
    def load_prediction_data():
        df = pd.read_csv("https://raw.githubusercontent.com/VictorKilanko/california-crime-dashboard/main/chapter1log.csv")
        df['City'] = df['City'].str.lower().str.strip()
        return df

    df = load_prediction_data()
    df = df.dropna(subset=['City'])

    crime_label_map = {
        "Violent_per_100k": "Violent Crime",
        "Property_per_100k": "Property Crime",
        "Homicide_per_100k": "Homicide",
        "ForRape_per_100k": "Forcible Rape",
        "FROBact_per_100k": "Robbery (Firearm)",
        "ViolentClr_per_100k": "Violent Crime Clearance",
        "Robbery_per_100k": "Robbery",
        "AggAssault_per_100k": "Aggravated Assault",
        "Burglary_per_100k": "Burglary",
        "FASSact_per_100k": "Assault (Firearm)",
        "PropertyClr_per_100k": "Property Crime Clearance",
        "VehicleTheft_per_100k": "Vehicle Theft",
        "LTtotal_per_100k": "Larceny-Theft",
        "Arson_per_100k": "Arson"
    }
    reverse_label_map = {v: k for k, v in crime_label_map.items()}
    crime_options = list(crime_label_map.values())

    st.subheader("🎯 Step 1: Choose Crime Outcome")
    target_label = st.selectbox("Select crime to predict", crime_options)
    target_col = reverse_label_map[target_label]

    st.subheader("🏙️ Step 2: Choose City")
    city_options = sorted(df['City'].str.title().unique())
    selected_city = st.selectbox("Select a city", city_options)
    city_data = df[df['City'].str.title() == selected_city]

    if city_data.empty:
        st.warning("No data for selected city.")
    else:
        full_df = df.dropna(subset=[target_col])
        numeric_cols = full_df.select_dtypes(include='number').columns
        excluded_cols = ['City', 'County', 'Year', 'Month'] + list(crime_label_map.keys())
        predictors = [col for col in numeric_cols if col not in excluded_cols]
        corr_values = full_df[predictors].corrwith(full_df[target_col]).abs().sort_values(ascending=False)
        selected_features = corr_values.head(6).index.tolist()

        X = full_df[selected_features].fillna(0)
        y = full_df[target_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression().fit(X_scaled, y)
        r2 = model.score(X_scaled, y)

        st.subheader("🎛️ Step 3: Adjust Influencing Variables")
        st.markdown("Tweak variables to simulate how changing city conditions affect predicted crime outcomes.")

        user_inputs = {}
        col1, col2 = st.columns(2)
        for i, feat in enumerate(selected_features):
            default = city_data[feat].mean() if feat in city_data.columns else X[feat].mean()
            min_val = float(X[feat].min())
            max_val = float(X[feat].max())
            col = col1 if i % 2 == 0 else col2
            user_inputs[feat] = col.slider(
                feat.replace('_', ' ').title(),
                min_value=round(min_val, 2),
                max_value=round(max_val, 2),
                value=round(default, 2),
                step=1.0
            )

        input_array = np.array([list(user_inputs.values())])
        predicted = model.predict(scaler.transform(input_array))[0]

        st.markdown("---")
        st.subheader("📈 Predicted Crime Rate")
        st.metric(f"{target_label} in {selected_city}", f"{predicted:.1f} per 100,000")

        st.markdown("### 🧠 Model Formula")
        formula = f"{target_label} = " + " + ".join([f"{coef:.2f}×{feat}" for coef, feat in zip(model.coef_, selected_features)])
        st.code(formula, language="python")

        st.markdown(f"**Model R² score:** `{r2:.3f}` — higher means better fit")

# ---------------------------
# PAGE 5: LA VIOLENT CRIME API INTEGRATION
# ---------------------------
elif page == "LA Violent Crime Prediction":
    import requests

    st.subheader("🚨 LA Violent Crime Prediction API")
    st.markdown("Use our deployed API to predict the violent crime rate in Los Angeles based on core indicators.")

    # Input fields
    homicide = st.number_input("Homicide per 100k", 0.0, 100.0, 5.0)
    rape = st.number_input("Forcible Rape per 100k", 0.0, 100.0, 40.2)
    robbery = st.number_input("Robbery per 100k", 0.0, 300.0, 110.5)
    assault = st.number_input("Aggravated Assault per 100k", 0.0, 500.0, 210.0)
    truck_drivers = st.number_input("Truck Drivers (Heavy & Tractor-Trailer)", 0.0, 100.0, 50.0)
    vets = st.number_input("Male Vietnam Veterans", 0.0, 100.0, 25.0)

    if st.button("Predict Violent Crime Rate"):
        payload = {
            "Homicide_per_100k": homicide,
            "ForRape_per_100k": rape,
            "Robbery_per_100k": robbery,
            "AggAssault_per_100k": assault,
            "TruckDrivers": truck_drivers,
            "MaleVietnamVeterans": vets
        }

        with st.spinner("Sending request to API..."):
            try:
                response = requests.post("https://la-crime-api.onrender.com/predict", json=payload)
                if response.status_code == 200:
                    prediction = response.json()["Violent_per_100k_Predicted"]
                    st.success(f"Predicted Violent Crime Rate: {prediction} per 100,000")
                else:
                    st.error("API error. Please check input or try again later.")
            except Exception as e:
                st.error(f"Request failed: {e}")

