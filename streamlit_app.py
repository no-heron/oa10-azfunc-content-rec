import os
import random

import streamlit as st
import requests

API_URL = "https://oc10-bookrecs-hacad4gmd8bjdfdd.francecentral-01.azurewebsites.net/api/"
# API_URL = "http://localhost:7071/api"
# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Recommendation System", page_icon="✨", layout="centered")

st.markdown("<h2 style='text-align: center;'>Recommendation System</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter an optional User ID or Article ID to get recommendations.</p>", unsafe_allow_html=True)

# ---------------------------
# Input Section (Centered)
# ---------------------------
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(f"Querying from: {API_URL}")
        # Fetch random users
        try:
            with st.spinner("Fetching users..."):
                response = requests.get(f"{API_URL}/random_users")
                response.raise_for_status()
                user_options = response.json()
            # Create dropdown for article selection
            user_id = st.selectbox("Select a user (optional)", user_options)
        except Exception as e:
            st.error(f"Failed to fetch users: {e}")
            user_id = st.text_input("User ID (optional, integers from 0 to 706)", placeholder="e.g. 123")
# ---------------------------
# Helper for badge colors
# ---------------------------
BADGE_COLORS = {
    "Collaborative Filtering": "#007bff",
    "Content Similarity": "#28a745",
    "Popularity": "#ffc107",
    "Freshness": "#17a2b8"
}

def make_badge(label: str) -> str:
    """Return HTML for a colored badge."""
    color = BADGE_COLORS.get(label, "#6c757d")
    return f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-right: 6px;
        ">
            {label}
        </span>
    """

# ---------------------------
# API Call and Display
# ---------------------------
if st.button("Get Recommendations", use_container_width=True):
    params = {}
    query_url = f"{API_URL}/recommendations"
    if user_id:
        query_url += f"?user_id={user_id}"

    st.write(query_url)
    with st.spinner("Fetching recommendations..."):
        try:
            response = requests.get(query_url)
            response.raise_for_status()
            results = response.json()

            if not results:
                st.info("No recommendations found.")
            else:
                # ---------------------------
                # Display Each Recommendation as a Card
                # ---------------------------
                for r in sorted(results, key=lambda x: x["overall_score"], reverse=True):
                    scores = {
                        "Collaborative Filtering": r.get("cf_score"),
                        "Content Similarity": r.get("cb_score"),
                        "Popularity": r.get("popularity_score"),
                        "Freshness": r.get("freshness_score"),
                    }
                    # Keep only scores that are not None
                    scores = {k: v for k, v in scores.items() if v is not None}
                    top_two = sorted(scores, key=scores.get, reverse=True)[:2] # type:ignore
                    badges_html = "".join(make_badge(tag) for tag in top_two)

                    card_html = f"""
                        <div style="
                            background-color: #1e1e1e;
                            color: #f5f5f5;
                            border-radius: 12px;
                            padding: 18px 22px;
                            margin-top: 18px;
                            box-shadow: 0 0 10px rgba(0,0,0,0.4);
                            border: 1px solid #333;
                        ">
                            <h4 style="margin: 0;">📰 Article ID:
                                <span style="color:#4da6ff;">{r['article_id']}</span>
                            </h4>
                            <p style="margin: 6px 0 10px 0; color: #ccc;">
                                Overall Score: <b style="color:#ffd700;">{r['overall_score']:.3f}</b>
                            </p>
                            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                                {badges_html}
                        """

                    st.markdown(card_html, unsafe_allow_html=True)
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")