# app/recommender_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
import traceback
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

load_dotenv()

SUPABASE_URL = "https://byunlkvjaiskurdmwese.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="GoGenius Travel Recommender API",
    description="Recommends destinations using Supabase data based on user preferences",
    version="4.1.0"
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
    images: str
    description: Optional[str] = None
    rating: Optional[str] = None
    budget_range: Optional[str] = None

def convert_preferences_to_tags(pref: UserPreference) -> str:
    tag_fields = (
            pref.travel_style + [pref.duration, pref.budget, pref.climate] +
            pref.companions + pref.past_destinations +
            pref.favorite_trip + pref.least_favorite_trip +
            pref.transport + pref.accommodation + pref.interests
    )
    clean_tags = [str(tag).replace("_", " ").lower() for tag in tag_fields if tag]
    return " ".join(clean_tags)

def fetch_destinations_from_supabase() -> pd.DataFrame:
    response = supabase.table("destinations").select("*").execute()
    data = response.data
    print("🗕️ Raw data from Supabase:", data)
    if not data:
        raise HTTPException(status_code=404, detail="No destinations found in Supabase.")
    df = pd.DataFrame(data)
    df["images"] = df["images"].fillna("https://byunlkvjaiskurdmwese.supabase.co/storage/v1/object/sign/images/ChatGPT%20Image%20Apr%2018,%202025,%2010_59_37%20PM.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJpbWFnZXMvQ2hhdEdQVCBJbWFnZSBBcHIgMTgsIDIwMjUsIDEwXzU5XzM3IFBNLnBuZyIsImlhdCI6MTc0NTEwMjAzNCwiZXhwIjoxNzc2NjM4MDM0fQ.DrjYobr3x3mj03aoPps4yPL1yYUmgNiK-YPkRaccVDM")
    df["tags"] = (
        df["style"].fillna("") + " " +
        df["climate"].fillna("") + " " +
        df["transport_options"].fillna("") + " " +
        df["accommodation_type"].fillna("") + " " +
        df["interests_activities"].fillna("") + " " +
        df["budget_range"].fillna("")
    ).str.lower()
    return df[["id", "name", "tags", "images", "description", "rating", "budget_range"]]

def recommend_content_based(user_tags: str, destinations: pd.DataFrame, top_n: int = 10):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(destinations["tags"])
    user_vector = tfidf.transform([user_tags])

    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return pd.Series(sim_scores, index=destinations.index)

def recommend_collaborative(user_id: str, destinations: pd.DataFrame, top_n: int = 10):
    try:
        rating_response = supabase.table("ratings").select("*").execute()
        ratings_data = pd.DataFrame(rating_response.data)
        if ratings_data.empty:
            return pd.Series([0.0] * len(destinations), index=destinations.index)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_data[["user_id", "destination_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        model = SVD()
        model.fit(trainset)

        predictions = []
        for idx, row in destinations.iterrows():
            pred = model.predict(user_id, str(row["id"])).est
            predictions.append(pred)

        return pd.Series(predictions, index=destinations.index)
    except Exception as e:
        print("⚠️ Collaborative filtering failed:", e)
        return pd.Series([0.0] * len(destinations), index=destinations.index)

def hybrid_recommendation(preference: UserPreference, top_n: int = 10):
    tags = convert_preferences_to_tags(preference)
    destinations_data = fetch_destinations_from_supabase()

    print(f"\n📦 [Hybrid] Total destinations: {len(destinations_data)}")
    print(f"📌 [User Tags]: {tags}\n")

    content_scores = recommend_content_based(tags, destinations_data)
    collab_scores = recommend_collaborative(preference.user_id, destinations_data)

    final_scores = (0.6 * content_scores) + (0.4 * collab_scores)
    top_indices = final_scores.sort_values(ascending=False).head(top_n).index
    top_results = destinations_data.loc[top_indices]
    top_results["id"] = top_results["id"].astype(str)

    return top_results.to_dict(orient="records")

@app.post("/recommend", response_model=List[Destination])
def get_recommendations(preference: UserPreference):
    try:
        return hybrid_recommendation(preference)
    except Exception as e:
        print("❌ INTERNAL ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Something went wrong. Check logs.")
