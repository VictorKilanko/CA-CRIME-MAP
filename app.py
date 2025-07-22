
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------------------------- SETUP ---------------------------
st.set_page_config(page_title="CA CRIME MAP", layout="wide")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Crime Hotspot Map", "Top 10 Cities & Breakdown", "Crime Cluster Patterns"])
st.title("📍 CA CRIME MAP")

# --------------------------- DATA LOADING ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/VictorKilanko/california-crime-dashboard/main/chapter1log.csv"
    df = pd.read_csv(url)

    cities = pd.read_csv("uscities.csv")
    ca_cities = cities[cities['state_name'] == 'California'][['city', 'county_name', 'lat', 'lng']]
    ca_cities.columns = ['City', 'County', 'Latitude', 'Longitude']

    df['City'] = df['City'].str.lower().str.strip()
    ca_cities['City'] = ca_cities['City'].str.lower().str.strip()

    per_capita_cols = [col for col in df.columns if col.endswith('_per_100k') and 'clr' not in col]
    rename_dict = {
        'Aggassault_per_100k': 'Aggravated Assault',
        'Burglary_per_100k': 'Burglary',
        'Fassact_per_100k': 'Firearm Assault',
        'Frobact_per_100k': 'Firearm Robbery',
        'Forrape_per_100k': 'Forcible Rape',
        'Homicide_per_100k': 'Homicide',
        'Lttotal_per_100k': 'Larceny-Theft',
        'Property_per_100k': 'Property Crime',
        'Robbery_per_100k': 'Robbery',
        'Vehicletheft_per_100k': 'Vehicle Theft',
        'Violent_per_100k': 'Violent Crime',
        'Arson_per_100k': 'Arson'
    }
    filtered_rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=filtered_rename_dict)
    readable_cols = list(filtered_rename_dict.values())

    filtered_df = df[['County', 'City'] + readable_cols]
    city_crime_df = filtered_df.groupby(['County', 'City']).mean(numeric_only=True).reset_index()
    city_crime_df['Total Crime Rate'] = city_crime_df[readable_cols].sum(axis=1)

    threshold = city_crime_df['Total Crime Rate'].quantile(0.75)
    city_crime_df['Hotspot'] = (city_crime_df['Total Crime Rate'] >= threshold).astype(int)

    merged_df = pd.merge(city_crime_df, ca_cities, on='City', how='left')
    merged_df = merged_df.dropna(subset=['Latitude', 'Longitude'])

    return merged_df, readable_cols

merged_df, crime_cols = load_data()

# --------------------------- PAGE 1 ---------------------------
if page == "Crime Hotspot Map":
    st.subheader("🗺️ California Crime Hotspot Map")
    map_center = [36.5, -119.5]
    m = folium.Map(location=map_center, zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in merged_df.iterrows():
        color = 'red' if row['Hotspot'] else 'green'
        popup = f"<b>City:</b> {row['City'].title()}, {row['County']}<br><b>Total Crime Rate:</b> {row['Total Crime Rate']:.1f}/100k"
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=300),
            tooltip=row['City'].title()
        ).add_to(marker_cluster)

    st_data = st_folium(m, width=1000, height=600)

# --------------------------- PAGE 2 ---------------------------
elif page == "Top 10 Cities & Breakdown":
    st.subheader("🔥 Top 10 Hotspot Cities in California")
    top_hotspots = merged_df[merged_df['Hotspot'] == 1].sort_values(by='Total Crime Rate', ascending=False).head(10)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_hotspots, x='City', y='Total Crime Rate', palette='Reds_r', ax=ax1)
    ax1.set_title("Top 10 Hotspot Cities by Total Crime Rate")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("🧊 Top 10 Safest Cities in California")
    safest = merged_df[merged_df['Hotspot'] == 0].sort_values(by='Total Crime Rate', ascending=True).head(10)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=safest, x='City', y='Total Crime Rate', palette='Greens', ax=ax2)
    ax2.set_title("Top 10 Safest Cities by Total Crime Rate")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("🔎 Crime Type Breakdown in Hotspot Cities")
    city_options = top_hotspots['City'].str.title().tolist()
    selected_city = st.selectbox("Select a City", city_options)
    row = merged_df[merged_df['City'].str.title() == selected_city].iloc[0]
    breakdown = row[crime_cols].sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=breakdown.values, y=breakdown.index, palette="Reds", ax=ax3)
    ax3.set_title(f"Crime Breakdown in {selected_city}")
    ax3.set_xlabel("Incidents per 100k")
    st.pyplot(fig3)

# --------------------------- PAGE 3 ---------------------------
elif page == "Crime Cluster Patterns":
    st.subheader("🧬 Clustering Cities by Crime Pattern")

    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_data = merged_df[crime_cols]
    merged_df['Cluster'] = kmeans.fit_predict(cluster_data)

    cluster_summary = merged_df.groupby('Cluster')[crime_cols].mean().T

    st.markdown("### 🔢 Cities per Cluster")
    cluster_counts = merged_df['Cluster'].value_counts().sort_index()
    fig4, ax4 = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='Blues', ax=ax4)
    ax4.set_title("Number of Cities in Each Crime Cluster")
    ax4.set_xlabel("Cluster")
    ax4.set_ylabel("City Count")
    st.pyplot(fig4)

    st.markdown("### 🔥 Cluster Crime Profiles (Heatmap)")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_summary, cmap="Reds", annot=True, fmt=".1f", ax=ax5)
    ax5.set_title("Average Crime Rates per Cluster")
    st.pyplot(fig5)

    st.markdown("### 📍 Cities in Selected Cluster")
    selected_cluster = st.selectbox("Choose a Cluster to View Cities", cluster_counts.index.tolist())
    cluster_cities = merged_df[merged_df['Cluster'] == selected_cluster][['City', 'County']]
    st.dataframe(cluster_cities.sort_values(by='City').reset_index(drop=True))
