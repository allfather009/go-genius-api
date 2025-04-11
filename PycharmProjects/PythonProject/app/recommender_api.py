# app/recommender_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "airbnb19.p.rapidapi.com"

app = FastAPI(
    title="GoGenius Travel Recommender API",
    description="Recommends destinations using RapidAPI Airbnb data and personalized preferences",
    version="6.1.0"
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
    booking_url: str
    image_url: Optional[str] = None

def convert_preferences_to_tags(pref: UserPreference) -> str:
    tag_fields = (
        pref.travel_style + [pref.duration, pref.budget, pref.climate] +
        pref.companions + pref.past_destinations +
        pref.favorite_trip + pref.least_favorite_trip +
        pref.transport + pref.accommodation + pref.interests
    )
    clean_tags = [str(tag).replace("_", " ").lower() for tag in tag_fields if tag]
    return " ".join(clean_tags)

def extract_locations_from_preferences(preference: UserPreference) -> List[str]:
    locations = preference.favorite_trip + preference.past_destinations + preference.least_favorite_trip + preference.interests + preference.travel_style + preference.accommodation
    keywords = set([x.lower() for x in locations if isinstance(x, str) and len(x) > 2])
    return list(keywords) or ["paris"]

def fetch_airbnb_properties(city: str, limit: int = 10) -> pd.DataFrame:
    url = f"https://{RAPIDAPI_HOST}/api/v1/searchPropertyByLocationV2"
    querystring = {
        "location": city,
        "totalRecords": str(limit),
        "currency": "USD",
        "adults": "1"
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Airbnb API request failed for {city}")

    results = []
    data = response.json().get("data", [])
    for i, item in enumerate(data):
        results.append({
            "id": f"airbnb_{city}_{i+1}",
            "name": item.get("listingName", f"Airbnb stay in {city} #{i+1}"),
            "tags": f"{city} travel explore stay",
            "booking_url": item.get("detailPageUrl", ""),
            "image_url": item.get("optimizedThumbUrls", {}).get("srpDesktop")
        })
    return pd.DataFrame(results)

def recommend_content_based(user_tags: str, destinations: pd.DataFrame, top_n: int = 5):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(destinations["tags"])
    user_vector = tfidf.transform([user_tags])
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return pd.Series(sim_scores, index=destinations.index)

def hybrid_recommendation(preference: UserPreference, top_n: int = 5):
    print("✅ Converting preferences to tags...")
    tags = convert_preferences_to_tags(preference)
    print("🎯 Tags:", tags)

    user_locations = extract_locations_from_preferences(preference)
    print("🌐 Locations to fetch:", user_locations)

    all_data = pd.DataFrame()
    for city in user_locations:
        try:
            city_data = fetch_airbnb_properties(city)
            all_data = pd.concat([all_data, city_data], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Failed to fetch for {city}:", e)

    if all_data.empty:
        raise HTTPException(status_code=404, detail="No destinations found based on preferences.")

    print("🔀 Scoring Airbnb listings...")
    content_scores = recommend_content_based(tags, all_data)
    top_indices = content_scores.sort_values(ascending=False).head(top_n).index
    print("🏆 Final recommendations ready!")

    return all_data.loc[top_indices].to_dict(orient="records")

@app.post("/recommend", response_model=List[Destination])
def get_recommendations(preference: UserPreference):
    try:
        return hybrid_recommendation(preference)
    except Exception as e:
        print("💥 Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
