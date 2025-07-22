import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(page_title="CA CRIME MAP", layout="wide")
st.title("📍 CA CRIME MAP")
st.markdown("Explore crime hotspots and patterns across California cities.")

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def load_data():
    # Load main crime dataset from your GitHub repo
    url = "https://raw.githubusercontent.com/VictorKilanko/california-crime-dashboard/main/chapter1log.csv"
    df = pd.read_csv(url)

    # Load city coordinates
    uscities = pd.read_csv("uscities.csv")
    ca_cities = uscities[uscities['state_name'] == 'California'][['city', 'county_name', 'lat', 'lng']]
    ca_cities.columns = ['City', 'County', 'Latitude', 'Longitude']

    # Clean and process
    per_capita_cols = [col for col in df.columns if col.endswith('_per_100k')]
    meta_cols = ['County', 'City']
    filtered_df = df[meta_cols + per_capita_cols]
    city_crime_df = filtered_df.groupby(['County', 'City']).mean(numeric_only=True).reset_index()
    city_crime_df['TotalCrime_per_100k'] = city_crime_df[per_capita_cols].sum(axis=1)
    threshold = city_crime_df['TotalCrime_per_100k'].quantile(0.75)
    city_crime_df['Hotspot'] = (city_crime_df['TotalCrime_per_100k'] >= threshold).astype(int)

    # Merge coordinates
    city_crime_df['City'] = city_crime_df['City'].str.lower().str.strip()
    ca_cities['City'] = ca_cities['City'].str.lower().str.strip()
    merged_df = pd.merge(city_crime_df, ca_cities, on='City', how='left')
    merged_df = merged_df.dropna(subset=['Latitude', 'Longitude'])

    return merged_df, per_capita_cols

merged_df, per_capita_cols = load_data()

# ---------------------------
# INTERACTIVE MAP
# ---------------------------
st.subheader("🗺️ California Crime Hotspot Map")

map_center = [36.5, -119.5]
m = folium.Map(location=map_center, zoom_start=6)
marker_cluster = MarkerCluster().add_to(m)

for _, row in merged_df.iterrows():
    color = 'red' if row['Hotspot'] else 'green'
    popup = f"""
    <b>City:</b> {row['City'].title()}, {row['County']}<br>
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
# TOP 10 HOTSPOTS
# ---------------------------
st.subheader("🔥 Top 10 Hotspot Cities in California")
hotspots = merged_df[merged_df['Hotspot'] == 1].sort_values(by='TotalCrime_per_100k', ascending=False).head(10)

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(data=hotspots, x='City', y='TotalCrime_per_100k', palette='Reds_r', ax=ax1)
ax1.set_title("Top 10 Hotspot Cities by Total Crime Rate")
ax1.set_ylabel("Total Crimes per 100k")
ax1.set_xlabel("City")
plt.xticks(rotation=45)
st.pyplot(fig1)

# ---------------------------
# TOP 10 SAFEST CITIES
# ---------------------------
st.subheader("🧊 Top 10 Safest Cities in California")
safest = merged_df[merged_df['Hotspot'] == 0].sort_values(by='TotalCrime_per_100k', ascending=True).head(10)

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(data=safest, x='City', y='TotalCrime_per_100k', palette='Greens', ax=ax2)
ax2.set_title("Top 10 Safest Cities by Total Crime Rate")
ax2.set_ylabel("Total Crimes per 100k")
ax2.set_xlabel("City")
plt.xticks(rotation=45)
st.pyplot(fig2)

# ---------------------------
# BREAKDOWN OF CRIME TYPES
# ---------------------------
st.subheader("🔎 Crime Type Breakdown in Hotspot Cities")

city_options = hotspots['City'].str.title().unique().tolist()
selected_city = st.selectbox("Select a City", city_options)

row = merged_df[merged_df['City'].str.title() == selected_city].iloc[0]
breakdown = row[per_capita_cols].sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x=breakdown.values, y=breakdown.index.str.replace("_per_100k", "").str.replace("_", " ").str.title(), palette="Reds", ax=ax3)
ax3.set_title(f"Crime Breakdown in {selected_city}")
ax3.set_xlabel("Incidents per 100k")
st.pyplot(fig3)

