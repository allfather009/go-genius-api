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

load_dotenv()

SUPABASE_URL = "https://byunlkvjaiskurdmwese.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SVD_MODEL_PATH = "svd_model.pkl"  # For production, use a 'models/' subfolder

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

class RecommendationResult(BaseModel):
    id: str
    name: str
    cbf_score: float
    cf_score: float
    hybrid_score: float

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
    Fetch destination data from Supabase and preprocess for recommendation.
    """
    response = supabase.table("destinations").select("*").execute()
    data = response.data
    if not data:
        raise HTTPException(status_code=404, detail="No destinations found in Supabase.")
    df = pd.DataFrame(data)
    df["images"] = df["images"].fillna("https://your-default-image-link.png")
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
    Content-Based Filtering using TF-IDF and cosine similarity.
    """
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(destinations["tags"])
    user_vector = tfidf.transform([user_tags])
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return pd.Series(sim_scores, index=destinations.index)

def recommend_collaborative(user_id: str, destinations: pd.DataFrame):
    """
    Collaborative Filtering (CF) using SVD. Loads model if available, otherwise trains and saves new model.
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

        if os.path.exists(SVD_MODEL_PATH):
            model = joblib.load(SVD_MODEL_PATH)
        else:
            model = SVD()
            trainset = data.build_full_trainset()
            model.fit(trainset)
            joblib.dump(model, SVD_MODEL_PATH)

        full_predictions = []
        for idx, row in destinations.iterrows():
            pred = model.predict(user_id, str(row["id"]))
            full_predictions.append(pred.est)
        return pd.Series(full_predictions, index=destinations.index)
    except Exception as e:
        print("⚠️ Collaborative filtering failed:", e)
        return pd.Series([0.0] * len(destinations), index=destinations.index)

def hybrid_recommendation(preference: UserPreference, top_n: int = 10):
    """
    Hybrid recommendation (weighted average of CBF and CF).
    Returns top_n recommendations.
    """
    tags = convert_preferences_to_tags(preference)
    destinations_data = fetch_destinations_from_supabase()
    content_scores = recommend_content_based(tags, destinations_data)
    collab_scores = recommend_collaborative(preference.user_id, destinations_data)
    final_scores = (0.6 * content_scores) + (0.4 * collab_scores)
    destinations_data = destinations_data.copy()
    destinations_data['cbf_score'] = content_scores
    destinations_data['cf_score'] = collab_scores
    destinations_data['hybrid_score'] = final_scores
    top_df = destinations_data.sort_values("hybrid_score", ascending=False).head(top_n)
    return top_df[['id', 'name', 'cbf_score', 'cf_score', 'hybrid_score']].to_dict(orient="records")

@app.post("/recommend", response_model=List[RecommendationResult])
def get_recommendations(preference: UserPreference):
    """
    Main endpoint: Returns top recommendations using hybrid recommender.
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
    Endpoint to show CBF scores for all destinations for a given user (for debugging/analysis).
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
        past_destinations=[v for v in (pref_resp.data.get("past_destinations_visited", "") or "").split(",") if v],
        favorite_trip=[pref_resp.data.get("favorite_past_trip", "")],
        least_favorite_trip=[pref_resp.data.get("least_favorite_trip", "")],
        transport=[pref_resp.data.get("preferred_transport") or ""],
        accommodation=[pref_resp.data.get("preferred_accommodation_type") or ""],
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

@app.get("/evaluate-hybrid")
def evaluate_hybrid(k: int = 5):
    """
    Evaluates the hybrid recommender algorithm with offline test split.
    Returns RMSE, MAE for CF, and Precision@k, Recall@k, and AUC for Hybrid.
    """
    try:
        # 1. Get all ratings
        rating_response = supabase.table("ratings").select("*").execute()
        ratings_data = pd.DataFrame(rating_response.data)
        if ratings_data.empty:
            raise HTTPException(status_code=404, detail="No ratings found for evaluation.")
        ratings_data["destination_id"] = ratings_data["destination_id"].astype(str)
        ratings_data["user_id"] = ratings_data["user_id"].astype(str)

        # 2. Train-test split (80/20)
        train, test = train_test_split(ratings_data, test_size=0.2, random_state=42)

        # 3. Retrain SVD on train set only
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train[["user_id", "destination_id", "rating"]], reader)
        model = SVD()
        trainset = data.build_full_trainset()
        model.fit(trainset)
        joblib.dump(model, SVD_MODEL_PATH)

        # 4. RMSE/MAE for Collaborative Filtering
        cf_preds, cf_trues = [], []
        for _, row in test.iterrows():
            pred = model.predict(row["user_id"], row["destination_id"]).est
            cf_preds.append(pred)
            cf_trues.append(row["rating"])
        rmse = np.sqrt(mean_squared_error(cf_trues, cf_preds))
        mae = mean_absolute_error(cf_trues, cf_preds)

        # 5. Hybrid metrics
        users = test["user_id"].unique()
        precisions, recalls, aucs = [], [], []
        destinations_df = fetch_destinations_from_supabase()
        destinations_df["id"] = destinations_df["id"].astype(str)

        for user in users:
            pref_resp = supabase.table("travel_preferences").select("*").eq("user_id", user).single().execute()
            if not pref_resp.data:
                continue
            # Defensive get for list fields to avoid None errors
            def get_safe_list(val):
                return [val] if val is not None and val != "" else []
            user_pref = UserPreference(
                user_id=user,
                travel_style=get_safe_list(pref_resp.data.get("favorite_travel_style", "")),
                duration=pref_resp.data.get("trip_duration", "") or "",
                budget=pref_resp.data.get("budget_range", "") or "",
                climate=pref_resp.data.get("climate", "") or "",
                companions=get_safe_list(pref_resp.data.get("travel_companions", "")),
                past_destinations=[v for v in (pref_resp.data.get("past_destinations_visited", "") or "").split(",") if v],
                favorite_trip=get_safe_list(pref_resp.data.get("favorite_past_trip", "")),
                least_favorite_trip=get_safe_list(pref_resp.data.get("least_favorite_trip", "")),
                transport=get_safe_list(pref_resp.data.get("preferred_transport", "")),
                accommodation=get_safe_list(pref_resp.data.get("preferred_accommodation_type", "")),
                interests=get_safe_list(pref_resp.data.get("top_interests_activities", ""))
            )

            user_test = test[test["user_id"] == user]
            relevant = set(user_test[user_test["rating"] >= 4]["destination_id"])

            # Get top-K hybrid recommendations for this user
            recs = hybrid_recommendation(user_pref, top_n=k)
            recommended = set([rec["id"] for rec in recs])
            hits = len(recommended & relevant)
            precisions.append(hits / k if k else 0)
            recalls.append(hits / len(relevant) if relevant else 0)

            # AUC for this user
            all_recs = hybrid_recommendation(user_pref, top_n=len(destinations_df))
            y_true = [1 if rec["id"] in relevant else 0 for rec in all_recs]
            y_scores = [rec["hybrid_score"] for rec in all_recs]
            try:
                # Only calculate if both 1s and 0s exist
                if sum(y_true) > 0 and sum(y_true) < len(y_true):
                    auc = roc_auc_score(y_true, y_scores)
                    aucs.append(auc)
            except Exception:
                continue

        return {
            "RMSE (Collaborative)": round(float(rmse), 3),
            "MAE (Collaborative)": round(float(mae), 3),
            f"Precision@{k} (Hybrid Model)": round(np.mean(precisions), 3) if precisions else None,
            f"Recall@{k} (Hybrid Model)": round(np.mean(recalls), 3) if recalls else None,
            "AUC (Hybrid Model)": round(np.mean(aucs), 3) if aucs else None,
            "Users Evaluated": len(precisions)
        }
    except Exception as e:
        print("❌ EVALUATE HYBRID ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal evaluation error: {str(e)}")
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

@app.get("/evaluate-hybrid")
def evaluate_hybrid(k: int = 5):
    """
    Evaluates the hybrid recommender algorithm with offline test split.
    Returns RMSE, MAE for CF, and Precision@k, Recall@k, and AUC for Hybrid.
    """
    try:
        # 1. Fetch all ratings from Supabase
        rating_response = supabase.table("ratings").select("*").execute()
        ratings_data = pd.DataFrame(rating_response.data)
        if ratings_data.empty:
            raise HTTPException(status_code=404, detail="No ratings found for evaluation.")
        ratings_data["destination_id"] = ratings_data["destination_id"].astype(str)
        ratings_data["user_id"] = ratings_data["user_id"].astype(str)

        # 2. Split ratings into train/test (80/20)
        train, test = train_test_split(ratings_data, test_size=0.2, random_state=42)

        # 3. Retrain SVD on train set only
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train[["user_id", "destination_id", "rating"]], reader)
        model = SVD()
        trainset = data.build_full_trainset()
        model.fit(trainset)
        joblib.dump(model, SVD_MODEL_PATH)

        # 4. Evaluate RMSE/MAE for Collaborative Filtering (CF)
        cf_preds, cf_trues = [], []
        for _, row in test.iterrows():
            pred = model.predict(row["user_id"], row["destination_id"]).est
            cf_preds.append(pred)
            cf_trues.append(row["rating"])
        rmse = np.sqrt(mean_squared_error(cf_trues, cf_preds))
        mae = mean_absolute_error(cf_trues, cf_preds)

        # 5. Evaluate Precision@k, Recall@k, AUC for Hybrid
        users = test["user_id"].unique()
        precisions, recalls, aucs = [], [], []
        destinations_df = fetch_destinations_from_supabase()
        destinations_df["id"] = destinations_df["id"].astype(str)

        def get_safe_list(val):
            # Helper to handle None or empty values for list fields
            return [val] if val is not None and val != "" else []

        for user in users:
            pref_resp = supabase.table("travel_preferences").select("*").eq("user_id", user).single().execute()
            if not pref_resp.data:
                continue
            user_pref = UserPreference(
                user_id=user,
                travel_style=get_safe_list(pref_resp.data.get("favorite_travel_style", "")),
                duration=pref_resp.data.get("trip_duration", "") or "",
                budget=pref_resp.data.get("budget_range", "") or "",
                climate=pref_resp.data.get("climate", "") or "",
                companions=get_safe_list(pref_resp.data.get("travel_companions", "")),
                past_destinations=[v for v in (pref_resp.data.get("past_destinations_visited", "") or "").split(",") if v],
                favorite_trip=get_safe_list(pref_resp.data.get("favorite_past_trip", "")),
                least_favorite_trip=get_safe_list(pref_resp.data.get("least_favorite_trip", "")),
                transport=get_safe_list(pref_resp.data.get("preferred_transport", "")),
                accommodation=get_safe_list(pref_resp.data.get("preferred_accommodation_type", "")),
                interests=get_safe_list(pref_resp.data.get("top_interests_activities", ""))
            )

            user_test = test[test["user_id"] == user]
            relevant = set(user_test[user_test["rating"] >= 4]["destination_id"])

            # Get top-K hybrid recommendations for this user
            recs = hybrid_recommendation(user_pref, top_n=k)
            recommended = set([rec["id"] for rec in recs])
            hits = len(recommended & relevant)
            precisions.append(hits / k if k else 0)
            recalls.append(hits / len(relevant) if relevant else 0)

            # AUC for this user
            all_recs = hybrid_recommendation(user_pref, top_n=len(destinations_df))
            y_true = [1 if rec["id"] in relevant else 0 for rec in all_recs]
            y_scores = [rec["hybrid_score"] for rec in all_recs]
            try:
                # Only calculate if both 1s and 0s exist
                if sum(y_true) > 0 and sum(y_true) < len(y_true):
                    auc = roc_auc_score(y_true, y_scores)
                    aucs.append(auc)
            except Exception:
                continue

        return {
            "RMSE (Collaborative)": round(float(rmse), 3),
            "MAE (Collaborative)": round(float(mae), 3),
            f"Precision@{k} (Hybrid Model)": round(np.mean(precisions), 3) if precisions else None,
            f"Recall@{k} (Hybrid Model)": round(np.mean(recalls), 3) if recalls else None,
            "AUC (Hybrid Model)": round(np.mean(aucs), 3) if aucs else None,
            "Users Evaluated": len(precisions)
        }
    except Exception as e:
        print("❌ EVALUATE HYBRID ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal evaluation error: {str(e)}")
