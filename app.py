import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------
# PAGE SETUP
# --------------------------------
st.set_page_config(page_title="CA CRIME MAP", layout="wide")
st.title("📍 CA CRIME MAP")
st.markdown("An interactive dashboard to explore California crime hotspots, breakdowns, and clusters.")

# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/VictorKilanko/california-crime-dashboard/main/chapter1log.csv"
    df = pd.read_csv(url)

    cities = pd.read_csv("uscities.csv")
    ca_cities = cities[cities['state_name'] == 'California'][['city', 'county_name', 'lat', 'lng']]
    ca_cities.columns = ['City', 'County', 'Latitude', 'Longitude']

    df['City'] = df['City'].str.lower().str.strip()
    ca_cities['City'] = ca_cities['City'].str.lower().str.strip()

    # Only keep columns that exist and end with _per_100k, and are not clearance metrics
    per_capita_cols = [col for col in df.columns if col.endswith('_per_100k') and 'clr' not in col]

    # Define renaming dict (will be filtered based on what exists)
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

    # Only rename columns that exist in the dataset
    filtered_rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=filtered_rename_dict)

    # Now build list of readable names that exist
    readable_cols = list(filtered_rename_dict.values())

    filtered_df = df[['County', 'City'] + readable_cols]
    city_crime_df = filtered_df.groupby(['County', 'City']).mean(numeric_only=True).reset_index()
    city_crime_df['Total Crime Rate'] = city_crime_df[readable_cols].sum(axis=1)

    threshold = city_crime_df['Total Crime Rate'].quantile(0.75)
    city_crime_df['Hotspot'] = (city_crime_df['Total Crime Rate'] >= threshold).astype(int)

    merged_df = pd.merge(city_crime_df, ca_cities, on='City', how='left')
    merged_df = merged_df.dropna(subset=['Latitude', 'Longitude'])

    return merged_df, readable_cols

# --------------------------------
# PAGE NAVIGATION
# --------------------------------
page = st.sidebar.selectbox("Navigate", ["📍 Hotspot Map", "📊 Top 10 Cities", "🔎 Crime Clusters"])

# --------------------------------
# PAGE 1: HOTSPOT MAP
# --------------------------------
if page == "📍 Hotspot Map":
    st.header("🗺️ California Crime Hotspot Map")

    m = folium.Map(location=[36.5, -119.5], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in merged_df.iterrows():
        color = 'red' if row['Hotspot'] else 'green'
        popup = f"""
        <b>City:</b> {row['City'].title()}, {row['County']}<br>
        <b>Total Crime Rate:</b> {row['Total Crime Rate']:.1f}/100k<br>
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

# --------------------------------
# PAGE 2: TOP 10s
# --------------------------------
elif page == "📊 Top 10 Cities":
    st.header("🔥 Top 10 Hotspot Cities")
    hotspots = merged_df[merged_df['Hotspot'] == 1].sort_values(by='Total Crime Rate', ascending=False).head(10)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hotspots, x='City', y='Total Crime Rate', palette='Reds_r', ax=ax1)
    ax1.set_title("Top 10 Crime Hotspot Cities")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.header("🧊 Top 10 Safest Cities")
    safest = merged_df[merged_df['Hotspot'] == 0].sort_values(by='Total Crime Rate').head(10)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=safest, x='City', y='Total Crime Rate', palette='Greens', ax=ax2)
    ax2.set_title("Top 10 Safest Cities in CA")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.header("🔎 Breakdown of Crimes in Hotspot Cities")
    selected_city = st.selectbox("Choose a hotspot city", hotspots['City'].str.title().tolist())
    row = merged_df[merged_df['City'].str.title() == selected_city].iloc[0]
    breakdown = row[crime_cols].sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=breakdown.values, y=breakdown.index, palette="Reds", ax=ax3)
    ax3.set_title(f"Crime Breakdown in {selected_city}")
    ax3.set_xlabel("Incidents per 100k")
    st.pyplot(fig3)

# --------------------------------
# PAGE 3: CLUSTER ANALYSIS
# --------------------------------
elif page == "🔎 Crime Clusters":
    st.header("🔎 Crime Pattern Clustering")

    # Perform clustering
    X = merged_df[crime_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Plot cluster counts
    st.subheader("📊 Number of Cities in Each Cluster")
    fig4, ax4 = plt.subplots()
    sns.countplot(x='Cluster', data=merged_df, palette='Set2', ax=ax4)
    ax4.set_ylabel("City Count")
    st.pyplot(fig4)

    # Plot heatmap
    st.subheader("🧭 Crime Pattern by Cluster (Heatmap)")
    cluster_profiles = merged_df.groupby('Cluster')[crime_cols].mean()

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_profiles.T, cmap='coolwarm', annot=True, fmt=".1f", ax=ax5)
    ax5.set_title("Average Crime Rate by Cluster")
    st.pyplot(fig5)

    # Cluster membership
    st.subheader("📍 Cities in Selected Cluster")
    selected_cluster = st.selectbox("Select Cluster", sorted(merged_df['Cluster'].unique()))
    cluster_cities = merged_df[merged_df['Cluster'] == selected_cluster][['City', 'County']].sort_values('City')

    st.dataframe(cluster_cities.rename(columns={'City': 'City', 'County': 'County'}), use_container_width=True)

    st.markdown("""
    💡 **How to interpret clusters**:
    - Cluster 0: Moderate mix of violent/property crimes.
    - Cluster 1: Lower crime rates overall.
    - Cluster 2: High larceny and vehicle theft.
    - Cluster 3: High violent crimes like assault and robbery.
    """)

