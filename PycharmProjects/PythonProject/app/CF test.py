# scripts/train_svd_offline.py

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import joblib
import os

SVD_MODEL_PATH = "svd_model.pkl"

def fetch_ratings():
    # Replace with your actual ratings loader, here is a sample CSV fallback
    # ratings = pd.read_csv("ratings.csv")
    from dotenv import load_dotenv
    from supabase import create_client
    import os
    load_dotenv()
    SUPABASE_URL = "https://byunlkvjaiskurdmwese.supabase.co"
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    rating_response = supabase.table("ratings").select("*").execute()
    ratings = pd.DataFrame(rating_response.data)
    ratings["destination_id"] = ratings["destination_id"].astype(str)
    return ratings

def train_and_evaluate():
    ratings = fetch_ratings()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "destination_id", "rating"]], reader)

    # Split into train (80%) and test (20%) for evaluation
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train SVD
    model = SVD()
    model.fit(trainset)

    # Evaluate on test set
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Save trained model for inference (API use)
    joblib.dump(model, SVD_MODEL_PATH)
    print(f"✅ Model saved to: {SVD_MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
