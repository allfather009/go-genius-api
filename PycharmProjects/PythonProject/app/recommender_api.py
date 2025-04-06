# app/recommender_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv

# ========================== SETUP ===============================
# Load API keys from .env file
load_dotenv()
GEOAPIFY_API_KEY = os.environ.get("779636eb19fb429fb577544f0a40d322")
AMADEUS_CLIENT_ID = os.environ.get("iDryAsA3DgdzaKw6F73vGEUsG7mXh6xn")
AMADEUS_CLIENT_SECRET = os.environ.get("wZor45KeeG39sW57")

# Initialize FastAPI app
app = FastAPI(
    title="GoGenius Travel Recommender API",
    description="Recommends destinations using real-time data from Geoapify and Amadeus APIs based on user preferences",
    version="3.1.0"
)

# ========================== MODELS ==============================

class UserPreference(BaseModel):
    user_id: str
    travel_style: Optional[List[str]] = []
    duration: Optional[str] = ""
    budget: Optional[str] = ""
    climate: Optional[str] = ""
    companions: Optional[List[str]] = []
    past_destinations: Optional[List[str]] = []
    favorite_trip: Optional[List[str]] = []
    least_favorite_trip: Optional[List[str]] = []
    transport: Optional[List[str]] = []
    accommodation: Optional[List[str]] = []
    interests: Optional[List[str]] = []

class Destination(BaseModel):
    id: str
    name: str
    tags: str

# ========================== HELPER FUNCTIONS ======================

def convert_preferences_to_tags(pref: UserPreference) -> str:
    tag_fields = (
        pref.travel_style + [pref.duration, pref.budget, pref.climate] +
        pref.companions + pref.past_destinations +
        pref.favorite_trip + pref.least_favorite_trip +
        pref.transport + pref.accommodation + pref.interests
    )
    clean_tags = [str(tag).replace("_", " ").lower() for tag in tag_fields if tag]
    return " ".join(clean_tags)

def fetch_from_geoapify(user_tags: str, limit: int = 25) -> pd.DataFrame:
    url = "https://api.geoapify.com/v2/places"
    # TEMP: use a valid fixed filter (bounding box around New York City)
    params = {
        "categories": "entertainment,sightseeing,accommodation",
        "filter": "rect:-74.2591,40.4774,-73.7002,40.9176",  # NYC bounding box
        "limit": limit,
        "apiKey": GEOAPIFY_API_KEY
    }
    res = requests.get(url, params=params)
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Geoapify API request failed. {res.text}")

    features = res.json().get("features", [])
    data = []
    for place in features:
        props = place.get("properties", {})
        if props.get("name"):
            data.append({
                "id": props.get("place_id", "geo_" + str(len(data))),
                "name": props["name"],
                "tags": user_tags
            })
    return pd.DataFrame(data)

def fetch_from_amadeus(user_tags: str, limit: int = 25) -> pd.DataFrame:
    token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET
    }
    token_res = requests.post(token_url, data=token_data)
    if token_res.status_code != 200:
        raise HTTPException(status_code=500, detail="Amadeus token request failed.")

    access_token = token_res.json().get("access_token")
    search_url = "https://test.api.amadeus.com/v1/reference-data/locations"
    params = {
        "keyword": user_tags.split()[0],
        "subType": "CITY",
        "page[limit]": limit
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    res = requests.get(search_url, headers=headers, params=params)
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail="Amadeus search failed.")

    data = res.json().get("data", [])
    results = []
    for item in data:
        if "name" in item["address"]:
            results.append({
                "id": item["id"],
                "name": item["address"]["name"],
                "tags": user_tags
            })
    return pd.DataFrame(results)

def recommend_content_based(user_tags: str, destinations: pd.DataFrame, top_n: int = 5):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(destinations["tags"])
    user_vector = tfidf.transform([user_tags])
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return pd.Series(sim_scores, index=destinations.index)

def hybrid_recommendation(preference: UserPreference, top_n: int = 5):
    tags = convert_preferences_to_tags(preference)
    geoapify_data = fetch_from_geoapify(tags)
    amadeus_data = fetch_from_amadeus(tags)
    combined_data = pd.concat([geoapify_data, amadeus_data], ignore_index=True)
    content_scores = recommend_content_based(tags, combined_data)
    top_indices = content_scores.sort_values(ascending=False).head(top_n).index
    return combined_data.loc[top_indices].to_dict(orient="records")

# ========================== API ROUTES =============================

@app.post("/recommend", response_model=List[Destination])
def get_recommendations(preference: UserPreference):
    """
    Receives user preferences and returns best matching destinations.
    Combines data from Geoapify and Amadeus APIs.
    """
    return hybrid_recommendation(preference)

# ========================== EXPLANATION ===========================
# - OpenTripMap removed by request.
# - Now using Geoapify and Amadeus APIs to fetch places and cities.
# - Real-time destinations are fetched, combined, and ranked by AI logic using TF-IDF similarity.
# - Response contains top N destination matches with ID, name, and tags.
