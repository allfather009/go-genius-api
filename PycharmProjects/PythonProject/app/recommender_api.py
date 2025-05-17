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
import joblib

# Load environment variables
load_dotenv()

SUPABASE_URL = "https://byunlkvjaiskurdmwese.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Path to save/load trained collaborative filtering model (SVD)
SVD_MODEL_PATH = "svd_model.pkl"  # Use 'models/svd_model.pkl' in production

app = FastAPI(
    title="GoGenius Travel Recommender API",
    description="Recommends destinations using Supabase data based on user preferences",
    version="4.2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== DATA MODELS ==========

class UserPreference(BaseModel):
    """
    Input model for user preferences.
    """
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

class RecommendationResult(BaseModel):
    """
        Output model for a recommended destination.
        """
    id: str
    name: str
    images: Optional[str] = None
    description: Optional[str] = None
    rating: Optional[float] = None
    budget_range: Optional[str] = None
    cbf_score: float
    cf_score: float
    hybrid_score: float


# ========== HELPER FUNCTIONS ==========

def convert_preferences_to_tags(pref: UserPreference) -> str:
    """
    Combine user preferences into a single string of tags for CBF.
    """
    tag_fields = (
        pref.travel_style + [pref.duration, pref.budget, pref.climate] +
        pref.companions + pref.past_destinations +
        pref.favorite_trip + pref.least_favorite_trip +
        pref.transport + pref.accommodation + pref.interests
    )
    clean_tags = [str(tag).replace("_", " ").lower() for tag in tag_fields if tag]
    return " ".join(clean_tags)

def fetch_destinations_from_supabase() -> pd.DataFrame:
    """
    Fetch all destination data from Supabase and preprocess for recommendations.
    """
    response = supabase.table("destinations").select("*").execute()
    data = response.data
    if not data:
        raise HTTPException(status_code=404, detail="No destinations found in Supabase.")
    df = pd.DataFrame(data)
    df["images"] = df["images"].fillna("https://your-default-image-link.png")
    # Compose CBF tags field from key columns
    df["tags"] = (
        df["style"].fillna("") + " " +
        df["climate"].fillna("") + " " +
        df["transport_options"].fillna("") + " " +
        df["accommodation_type"].fillna("") + " " +
        df["interests_activities"].fillna("") + " " +
        df["budget_range"].fillna("")
    ).str.lower()
    return df[["id", "name", "tags", "images", "description", "rating", "budget_range"]]

def recommend_content_based(user_tags: str, destinations: pd.DataFrame):
    """
    CBF: Recommend based on user preferences and destination metadata.
    """
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(destinations["tags"])
    user_vector = tfidf.transform([user_tags])
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return pd.Series(sim_scores, index=destinations.index)

def recommend_collaborative(user_id: str, destinations: pd.DataFrame):
    """
    CF: Predict user rating for all destinations using pre-trained SVD model.
    If no model found, trains one using all available ratings.
    """
    try:
        rating_response = supabase.table("ratings").select("*").execute()
        ratings_data = pd.DataFrame(rating_response.data)
        if ratings_data.empty:
            print("No ratings data found.")
            return pd.Series([0.0] * len(destinations), index=destinations.index)
        ratings_data["destination_id"] = ratings_data["destination_id"].astype(str)
        destinations["id"] = destinations["id"].astype(str)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_data[["user_id", "destination_id", "rating"]], reader)

        # Load model if available; else train and save
        if os.path.exists(SVD_MODEL_PATH):
            model = joblib.load(SVD_MODEL_PATH)
        else:
            model = SVD()
            trainset = data.build_full_trainset()
            model.fit(trainset)
            joblib.dump(model, SVD_MODEL_PATH)
        # Predict scores for user and all destinations
        predictions = [model.predict(user_id, str(row["id"])).est for _, row in destinations.iterrows()]
        return pd.Series(predictions, index=destinations.index)
    except Exception as e:
        print("⚠️ Collaborative filtering failed:", e)
        return pd.Series([0.0] * len(destinations), index=destinations.index)

def hybrid_recommendation(preference: UserPreference, top_n: int = 5):
    """
    Hybrid recommendation (weighted average of CBF and CF).
    Returns top_n recommendations with all destination data needed for frontend.
    """
    tags = convert_preferences_to_tags(preference)
    destinations_data = fetch_destinations_from_supabase()
    print(f"\n📦 [Hybrid] Total destinations: {len(destinations_data)}")
    print(f"📌 [User Tags]: {tags}\n")
    content_scores = recommend_content_based(tags, destinations_data)
    collab_scores = recommend_collaborative(preference.user_id, destinations_data)
    # Weighted sum: 60% CBF, 40% CF
    final_scores = (0.6 * content_scores) + (0.4 * collab_scores)
    destinations_data = destinations_data.copy()
    destinations_data['cbf_score'] = content_scores
    destinations_data['cf_score'] = collab_scores
    destinations_data['hybrid_score'] = final_scores
    top_df = destinations_data.sort_values("hybrid_score", ascending=False).head(top_n)
    # Now return all fields needed by frontend:
    return top_df[[
        'id',
        'name',
        'images',
        'description',
        'rating',
        'budget_range',
        'cbf_score',
        'cf_score',
        'hybrid_score'
    ]].to_dict(orient="records")



# ========== ENDPOINTS ==========

@app.post("/recommend", response_model=List[RecommendationResult])
def get_recommendations(preference: UserPreference):
    """
    Main hybrid endpoint: Returns top 10 recommendations for the user.
    """
    try:
        return hybrid_recommendation(preference)
    except Exception as e:
        print("❌ INTERNAL ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Something went wrong. Check logs.")

@app.get("/cbf-scores/{user_id}")
def cbf_scores(user_id: str):
    """
    CBF-only scores for all destinations for a given user (for debugging/analysis).
    """
    pref_resp = supabase.table("travel_preferences").select("*").eq("user_id", user_id).single().execute()
    if not pref_resp.data:
        raise HTTPException(status_code=404, detail="User preferences not found.")
    user_pref = UserPreference(
        user_id=user_id,
        travel_style=[pref_resp.data.get("favorite_travel_style", "")],
        duration=pref_resp.data.get("trip_duration", ""),
        budget=pref_resp.data.get("budget_range", ""),
        climate=pref_resp.data.get("climate", ""),
        companions=[pref_resp.data.get("travel_companions", "")],
        past_destinations=pref_resp.data.get("past_destinations_visited", "").split(","),
        favorite_trip=[pref_resp.data.get("favorite_past_trip", "")],
        least_favorite_trip=[pref_resp.data.get("least_favorite_trip", "")],
        transport=[pref_resp.data.get("preferred_transport", "")],
        accommodation=[pref_resp.data.get("preferred_accommodation_type", "")],
        interests=[pref_resp.data.get("top_interests_activities", "")]
    )
    tags = convert_preferences_to_tags(user_pref)
    destinations = fetch_destinations_from_supabase()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(destinations["tags"])
    user_vector = tfidf.transform([tags])
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return [
        {"destination": destinations.iloc[idx]['name'], "score": float(score)}
        for idx, score in enumerate(sim_scores)
    ]

@app.get("/cf-scores/{user_id}")
def cf_scores(user_id: str):
    """
    CF-only scores for all destinations for a given user (for debugging/analysis).
    """
    destinations = fetch_destinations_from_supabase()
    cf_scores = recommend_collaborative(user_id, destinations)
    return [
        {"destination": destinations.iloc[idx]['name'], "score": float(score)}
        for idx, score in enumerate(cf_scores)
    ]

@app.get("/hybrid-scores/{user_id}")
def hybrid_scores(user_id: str, top_n: int = 10):
    """
    Hybrid-only: Get top-N hybrid recommendations for a given user ID (uses user preferences from Supabase).
    """
    pref_resp = supabase.table("travel_preferences").select("*").eq("user_id", user_id).single().execute()
    if not pref_resp.data:
        raise HTTPException(status_code=404, detail="User preferences not found.")
    user_pref = UserPreference(
        user_id=user_id,
        travel_style=[pref_resp.data.get("favorite_travel_style", "")],
        duration=pref_resp.data.get("trip_duration", ""),
        budget=pref_resp.data.get("budget_range", ""),
        climate=pref_resp.data.get("climate", ""),
        companions=[pref_resp.data.get("travel_companions", "")],
        past_destinations=pref_resp.data.get("past_destinations_visited", "").split(","),
        favorite_trip=[pref_resp.data.get("favorite_past_trip", "")],
        least_favorite_trip=[pref_resp.data.get("least_favorite_trip", "")],
        transport=[pref_resp.data.get("preferred_transport", "")],
        accommodation=[pref_resp.data.get("preferred_accommodation_type", "")],
        interests=[pref_resp.data.get("top_interests_activities", "")]
    )
    return hybrid_recommendation(user_pref, top_n=top_n)

@app.post("/train-cf")
def train_cf_model():
    """
    Manually (re-)trains the SVD collaborative filtering model on all ratings in Supabase.
    Call this after adding many new ratings, or via a scheduled cron job.
    """
    try:
        rating_response = supabase.table("ratings").select("*").execute()
        ratings_data = pd.DataFrame(rating_response.data)
        if ratings_data.empty:
            raise HTTPException(status_code=404, detail="No ratings data found.")
        ratings_data["destination_id"] = ratings_data["destination_id"].astype(str)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_data[["user_id", "destination_id", "rating"]], reader)
        model = SVD()
        model.fit(data.build_full_trainset())
        joblib.dump(model, SVD_MODEL_PATH)
        return {"message": "SVD model trained and saved successfully."}
    except Exception as e:
        print("❌ TRAIN CF ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal training error: {str(e)}")

# END OF FILE
