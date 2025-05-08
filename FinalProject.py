"""
Name:  Lylah Tun
CS230: Section 2
Data:  McDonalds Store Reviews
URL:   https://cs230mcdonalds-8rkvwzn3cbxkyt6dft8gvg.streamlit.app/

Description:

This program is an interactive web app that lets users explore McDonald's customer reviews
across the U.S. Users can filter reviews by state and rating,
view charts showing review trends, and see store locations on a map.
The app includes bar charts, a pie chart, a pivot table, and sample customer comments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk


# Page config
st.set_page_config(page_title="McDonald's Review Explorer", layout="wide")


# [PY1], [PY2], [PY3] Load and clean data with multiple returns and error handling
@st.cache_data
def load_data(file_path="McDonald_s_Reviews(in).csv"):
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip().str.lower()
        df['review'] = df['review'].astype(str).str.replace('ï¿½', '', regex=False)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
        df['rating'] = pd.to_numeric(df['rating'].astype(str).str.extract(r'(\d)')[0], errors='coerce')
        return df, df.columns.tolist()  # [PY2] returns more than one value
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame(), []

# Load data
# [PY1] Function called with default value
data, data_columns = load_data()
st.title("🌟 McDonald's Review Explorer")


# Sidebar filters
st.sidebar.header("Filter Reviews")
unique_states = data['store_address'].str.extract(r', (\w{2}) ')[0].dropna().unique()
selected_state = st.sidebar.selectbox("Select a State:", options=np.append("All", np.sort(unique_states)))
selected_rating = st.sidebar.slider("Select minimum star rating:", 1, 5, 1)


# Navigation menu
st.sidebar.header("Navigate to Section")
section = st.sidebar.radio(
   "Go to:",
   ["Overview", "Rating Distribution", "Store Map", "Reviews", "Average by State", "Rating Pie by State", "Pivot Table"]
)


# Apply filters
filtered = data.copy()
if selected_state != "All":
   filtered = filtered[filtered['store_address'].str.contains(f", {selected_state} ", na=False)]

if selected_rating > 1:
   filtered = filtered[filtered['rating'] >= selected_rating]


# Group by location and compute average rating
grouped = filtered.groupby(['store_name', 'store_address', 'latitude', 'longitude'], dropna=True).agg(
   avg_rating=('rating', 'mean'),
   review_count=('rating', 'count')
).reset_index()

# Drop duplicates [DA1]
grouped = grouped.drop_duplicates(subset=['store_name', 'latitude', 'longitude', 'avg_rating'])

# Add derived column [DA9]
grouped['avg_rating_display'] = grouped['avg_rating'].round(2).astype(str)

# [PY4] List comprehension to flag highly rated
grouped['is_high_rated'] = ["Yes" if r >= 4 else "No" for r in grouped['avg_rating']]


# Color by average rating
def rating_to_color(rating):
   if pd.isna(rating):
       return [128, 128, 128]  # gray for missing
   return [255, 0, 0] if rating < 4 else [255, 215, 0]  # red or yellow


grouped['color'] = grouped['avg_rating'].apply(rating_to_color)


# ---- Overview ----
if section == "Overview":
   st.markdown("### Welcome to the McDonald's Review Explorer")
   st.markdown("Use the sidebar to explore the data through different visualizations including ratings distribution, geographic store map, sample customer reviews, and state-by-state comparisons.")
   st.image("MDlogo.png", width=200)
   st.markdown(f"**Filtered data contains:** {len(filtered)} reviews across {len(grouped)} unique locations.")


# ---- Section 1: Rating Distribution ----
elif section == "Rating Distribution":
   st.subheader("Rating Distribution")
   fig, ax = plt.subplots()
   rating_counts = filtered['rating'].value_counts().sort_index()
   ax.bar(rating_counts.index, rating_counts.values, color='orange')
   ax.set_xlabel("Star Rating")
   ax.set_ylabel("Number of Reviews")
   ax.set_title("Review Count by Star Rating")
   st.pyplot(fig)


# ---- Section 2: Map of McDonald's Locations ----
elif section == "Store Map":
   st.subheader("Map of McDonald's Review Locations")


   tooltip = {
       "html": "<b>Store:</b> McDonald's<br/>"
               "<b>Address:</b> {store_address}<br/>"
               "<b>Avg Rating:</b> {avg_rating_display}<br/>"
               "<b>Review Count:</b> {review_count}",
       "style": {"backgroundColor": "white", "color": "black"}
   }
   st.write(f"Number of points shown: {len(grouped)}")
   st.pydeck_chart(pdk.Deck(
       map_style='mapbox://styles/mapbox/light-v9',
       initial_view_state=pdk.ViewState(
           latitude=grouped['latitude'].mean(),
           longitude=grouped['longitude'].mean(),
           zoom=4,
           pitch=0,
       ),
       layers=[
           pdk.Layer(
               'ScatterplotLayer',
               data=grouped,
               get_position='[longitude, latitude]',
               get_color='color',
               get_radius=175,
               pickable=True,
               auto_highlight=True
           )
       ],
       tooltip=tooltip
   ))


# ---- Section 3: Reviews ----
elif section == "Reviews":
   st.subheader("Reviews")
   selected_review_rating = st.selectbox(
       "Select a rating to view reviews:",
       sorted(filtered['rating'].dropna().unique())
   )
   filtered_reviews = filtered[filtered['rating'] == selected_review_rating]
   st.dataframe(
       filtered_reviews[['store_address', 'rating', 'review_time', 'review']]
       .drop_duplicates()
       .sort_values(by='review_time', ascending=True)
       .sample(min(10, len(filtered_reviews)))
   )

# ---- Section 4: Average Rating by State ----
elif section == "Average by State":
   st.subheader("Average Rating by State")
   data['state'] = data['store_address'].str.extract(r', (\w{2}) ')
   avg_rating_state = data.groupby('state')['rating'].mean().sort_values(ascending=False)
   st.bar_chart(avg_rating_state)


# ---- Section 5: Pie Chart of Ratings by State ----
elif section == "Rating Pie by State":
   st.subheader(f"Pie Chart of Ratings for {selected_state}")
   if selected_state == "All":
       st.warning("Please select a specific state from the dropdown to view the pie chart.")
   else:
       state_data = data[data['store_address'].str.contains(f", {selected_state} ", na=False)]
       rating_counts = state_data['rating'].value_counts().sort_index()


       # Static Pie Chart
       fig, ax = plt.subplots()
       ax.pie(rating_counts, labels=rating_counts.index.astype(str), autopct='%1.1f%%', startangle=140)
       ax.set_title(f"Rating Distribution in {selected_state}")
       st.pyplot(fig)


       # Let user filter by one rating (simulate interaction)
       selected_rating = st.selectbox("View stores with rating:", rating_counts.index.astype(int))


       filtered_stores = state_data[state_data['rating'] == selected_rating]
       st.markdown(f"**Showing {len(filtered_stores)} stores with {selected_rating} stars**")
       st.dataframe(filtered_stores[['store_address', 'rating']])

       st.markdown("### Full Breakdown")
       df_sorted = state_data[['store_address', 'rating']].drop_duplicates().sort_values(by='rating', ascending=True)
       st.dataframe(df_sorted)

# [DA6] Pivot Table Section
elif section == "Pivot Table":
    st.subheader("Pivot Table: Review Counts by State and Rating")

    # Ensure state column exists
    data['state'] = data['store_address'].str.extract(r', (\w{2}) ')

    # Drop rows with missing state or rating
    pivot_data = data.dropna(subset=['state', 'rating'])

    # Create pivot table: count number of reviews per state per rating
    pivotab = pd.pivot_table(
        pivot_data,
        index='state',
        columns='rating',
        aggfunc='size',  # counts rows in each group
        fill_value=0
    )

    # Display pivot table
    st.dataframe(pivotab)

    # Show summary of states with high 5-star review counts
    st.markdown("### States with Most 5-Star Reviews")
    if 5 in pivotab.columns:
        top_states = pivotab[pivotab[5] > 50][5].sort_values(ascending=False)
        for state, count in top_states.items():
            st.write(f"⭐ {state}: {count} five-star reviews")
    else:
        st.info("No 5-star ratings found in the current dataset.")
