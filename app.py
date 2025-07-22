import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# PAGE CONFIG & TITLE
# ---------------------------
st.set_page_config(page_title="CA CRIME MAP", layout="wide")
st.title("📍 California Crime Map Dashboard")

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def load_data():
    # Load main crime data
    url = "https://raw.githubusercontent.com/VictorKilanko/california-crime-dashboard/main/chapter1log.csv"
    df = pd.read_csv(url)

    # Load city coordinates
    uscities = pd.read_csv("uscities.csv")
    ca_cities = uscities[uscities['state_name'] == 'California'][['city', 'county_name', 'lat', 'lng']]
    ca_cities.columns = ['City', 'County', 'Latitude', 'Longitude']

    # Clean + enrich
    per_capita_cols = [col for col in df.columns if col.endswith('_per_100k')]
    meta_cols = ['County', 'City']
    filtered_df = df[meta_cols + per_capita_cols]
    city_crime_df = filtered_df.groupby(['County', 'City']).mean(numeric_only=True).reset_index()
    city_crime_df['TotalCrime_per_100k'] = city_crime_df[per_capita_cols].sum(axis=1)
    threshold = city_crime_df['TotalCrime_per_100k'].quantile(0.75)
    city_crime_df['Hotspot'] = (city_crime_df['TotalCrime_per_100k'] >= threshold).astype(int)

    city_crime_df['City'] = city_crime_df['City'].str.lower().str.strip()
    ca_cities['City'] = ca_cities['City'].str.lower().str.strip()
    merged_df = pd.merge(city_crime_df, ca_cities, on='City', how='left')
    merged_df = merged_df.dropna(subset=['Latitude', 'Longitude'])

    return merged_df, per_capita_cols

# Load once
merged_df, per_capita_cols = load_data()

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🗺️ Hotspot Map", "📊 Top 10 & Breakdown", "🧬 Crime Clustering"])

# ---------------------------
# PAGE 1: INTERACTIVE MAP
# ---------------------------
if page == "🗺️ Hotspot Map":
    st.header("🗺️ California Crime Hotspot Map")

    m = folium.Map(location=[36.5, -119.5], zoom_start=6)
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

    st_folium(m, height=600, width=1100)

# ---------------------------
# PAGE 2: TOP 10 + BREAKDOWN
# ---------------------------
elif page == "📊 Top 10 & Breakdown":
    st.header("📊 Top 10 Cities by Crime Rate")

    hotspots = merged_df[merged_df['Hotspot'] == 1].sort_values(by='TotalCrime_per_100k', ascending=False).head(10)
    safest = merged_df[merged_df['Hotspot'] == 0].sort_values(by='TotalCrime_per_100k', ascending=True).head(10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Top 10 Hotspot Cities")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=hotspots, x='TotalCrime_per_100k', y='City', palette='Reds_r', ax=ax1)
        ax1.set_title("Top Crime Hotspots")
        ax1.set_xlabel("Crimes per 100k")
        ax1.set_ylabel("City")
        st.pyplot(fig1)

    with col2:
        st.subheader("🧊 Top 10 Safest Cities")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=safest, x='TotalCrime_per_100k', y='City', palette='Greens', ax=ax2)
        ax2.set_title("Lowest Crime Rates")
        ax2.set_xlabel("Crimes per 100k")
        ax2.set_ylabel("City")
        st.pyplot(fig2)

    st.subheader("🔎 Breakdown of Crime Types in a Hotspot City")
    city_options = hotspots['City'].str.title().unique().tolist()
    selected_city = st.selectbox("Select a City", city_options)

    row = merged_df[merged_df['City'].str.title() == selected_city].iloc[0]
    breakdown = row[per_capita_cols].sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=breakdown.values,
                y=breakdown.index.str.replace("_per_100k", "").str.replace("_", " ").str.title(),
                palette="Reds", ax=ax3)
    ax3.set_title(f"Crime Type Breakdown in {selected_city}")
    ax3.set_xlabel("Incidents per 100k")
    ax3.set_ylabel("Crime Type")
    st.pyplot(fig3)

# ---------------------------
# PAGE 3: CLUSTER ANALYSIS
# ---------------------------
elif page == "🧬 Crime Clustering":
    st.header("🧬 Crime Pattern Clustering Across Cities")

    X = merged_df[per_capita_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("🔢 Cluster Distribution")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Cluster', data=merged_df, palette='Set2', ax=ax4)
    ax4.set_title("Number of Cities per Crime Cluster")
    ax4.set_xlabel("Cluster")
    ax4.set_ylabel("City Count")
    st.pyplot(fig4)

    st.subheader("🌐 Crime Pattern by Cluster (Heatmap)")
    cluster_summary = merged_df.groupby('Cluster')[per_capita_cols].mean().T
    cluster_summary.index = cluster_summary.index.str.replace("_per_100k", "").str.replace("_", " ").str.title()

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_summary, cmap="coolwarm", annot=True, fmt=".1f", ax=ax5)
    ax5.set_title("Average Crime Rate by Cluster")
    st.pyplot(fig5)

# ---------------------------
# END OF FILE
# ---------------------------
