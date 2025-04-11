# app/recommender_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

STAY22_LETMEALLEZ_ID = os.getenv("STAY22_LETMEALLEZ_ID")

app = FastAPI(
    title="GoGenius Travel Recommender API",
    description="Recommends destinations using Stay22 embedded content and personalized preferences",
    version="5.1.0"
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

def generate_dynamic_destinations(locations: List[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": f"stay22_{i+1}",
            "name": f"Stay22 - {location.title()}",
            "tags": f"{location} travel explore stay",
            "booking_url": f"https://stay22.com/embed/{STAY22_LETMEALLEZ_ID}?location={location}",
            "image_url": f"https://source.unsplash.com/featured/?{location},travel"
        }
        for i, location in enumerate(locations)
    ])

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

    # Use common city keywords for simplicity, can be replaced with dynamic input in production
    fallback_locations = ["paris", "berlin", "rome", "vienna", "barcelona", "london", "amsterdam", "prague"]
    destinations = generate_dynamic_destinations(fallback_locations)

    print("🔀 Scoring dynamic Stay22 destinations...")
    content_scores = recommend_content_based(tags, destinations)
    top_indices = content_scores.sort_values(ascending=False).head(top_n).index
    print("🏆 Final recommendations ready!")

    return destinations.loc[top_indices].to_dict(orient="records")

@app.post("/recommend", response_model=List[Destination])
def get_recommendations(preference: UserPreference):
    try:
        return hybrid_recommendation(preference)
    except Exception as e:
        print("💥 Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
