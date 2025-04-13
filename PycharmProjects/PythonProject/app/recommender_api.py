# app/recommender_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # ✅ This loads from .env file

GEOAPIFY_API_KEY = os.environ.get("GEOAPIFY_API_KEY")

app = FastAPI(
    title="GoGenius Travel Recommender API",
    description="Recommends destinations using real-time data from Geoapify API based on user preferences",
    version="3.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    params = {
        "categories": "entertainment.culture,tourism.attraction,accommodation.hotel",
        "filter": "rect:-74.2591,40.4774,-73.7002,40.9176",
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

def recommend_content_based(user_tags: str, destinations: pd.DataFrame, top_n: int = 5):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(destinations["tags"])
    user_vector = tfidf.transform([user_tags])
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    print("\n[🔎 Content-Based Scores]")
    for i, score in enumerate(sim_scores):
        print(f"{destinations.iloc[i]['name']} → Score: {score:.4f}")

    return pd.Series(sim_scores, index=destinations.index)

def hybrid_recommendation(preference: UserPreference, top_n: int = 5):
    tags = convert_preferences_to_tags(preference)
    geoapify_data = fetch_from_geoapify(tags)
    combined_data = geoapify_data  # Only use Geoapify

    print(f"\n📦 [Hybrid] Total destinations: {len(combined_data)}")

    content_scores = recommend_content_based(tags, combined_data)
    top_indices = content_scores.sort_values(ascending=False).head(top_n).index
    top_results = combined_data.loc[top_indices]

    print("\n🏆 [Top Recommendations]")
    for _, row in top_results.iterrows():
        print(f"{row['name']} (ID: {row['id']})")

    return top_results.to_dict(orient="records")

@app.post("/recommend", response_model=List[Destination])
def get_recommendations(preference: UserPreference):
    """
    Receives user preferences and returns best matching destinations.
    Uses data from Geoapify API only.
    """
    return hybrid_recommendation(preference)
