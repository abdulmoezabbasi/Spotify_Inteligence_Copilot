from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import (
    get_top_genres, get_top_artists, find_similar_tracks,
    get_genre_mood_profile, _load_data
)
from src.monte_carlo import simulate_playlist as _mc_simulate
from src.neural_net import PopularityNet, load_model as load_nn_model

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")

model     = None
FEATURES  = []
nn_scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, FEATURES, nn_scaler
    try:
        model, FEATURES = load_nn_model(MODELS)
        with open(os.path.join(MODELS, "scaler.pkl"), "rb") as f:
            nn_scaler = pickle.load(f)
        print("✅ Neural network loaded")
    except Exception as e:
        print(f"❌ FATAL: {e}")
        raise
    yield


app = FastAPI(
    title="Spotify Intelligence API",
    description="ML-powered music intelligence",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    danceability:          float
    energy:                float
    loudness:              float
    speechiness:           float
    acousticness:          float
    instrumentalness:      float
    liveness:              float
    valence:               float
    tempo:                 float
    duration_min:          float = 3.5
    explicit:              int   = 0
    mood_score:            Optional[float] = None
    dance_energy:          Optional[float] = None
    acoustic_score:        Optional[float] = None
    genre_avg_popularity:  float = 40.0
    artist_avg_popularity: float = 40.0

class RecommendRequest(BaseModel):
    track_name: str
    n: int = 5

class PlaylistRequest(BaseModel):
    track_names: List[str]
    n_simulations: int = 10000


@app.get("/")
def root():
    return {"name": "Spotify Intelligence API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "PopularityNet", "version": "1.0.0"}

@app.post("/predict-popularity")
def predict_popularity(req: PredictRequest):
    if model is None or nn_scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        mood_score     = req.mood_score     or round(req.valence * 0.6 + req.energy * 0.4, 3)
        dance_energy   = req.dance_energy   or round(req.danceability * req.energy, 3)
        acoustic_score = req.acoustic_score or round(req.acousticness * 0.7 + (1 - req.energy) * 0.3, 3)

        features = np.array([[
            req.danceability, req.energy, req.loudness, req.speechiness,
            req.acousticness, req.instrumentalness, req.liveness, req.valence,
            req.tempo, req.duration_min, req.explicit,
            mood_score, dance_energy, acoustic_score,
            req.genre_avg_popularity, req.artist_avg_popularity,
        ]])
        features_scaled = nn_scaler.transform(features)
        tensor = torch.FloatTensor(features_scaled)
        with torch.no_grad():
            prediction = float(model(tensor).item())
        prediction = max(0.0, min(100.0, prediction))

        return {
            "predicted_popularity": round(prediction, 1),
            "confidence_range": {
                "low":  round(max(0, prediction - 6.1), 1),
                "high": round(min(100, prediction + 6.1), 1),
            },
            "tier": (
                "viral"  if prediction >= 80 else
                "high"   if prediction >= 60 else
                "medium" if prediction >= 40 else
                "low"    if prediction >= 20 else "unknown"
            ),
            "interpretation": f"This track is predicted to score {prediction:.0f}/100 on Spotify popularity",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        results = find_similar_tracks(req.track_name, req.n)
        if results is None:
            raise HTTPException(status_code=404, detail=f"Track '{req.track_name}' not found")
        return {
            "query": req.track_name,
            "recommendations": results[
                ["track_name", "artists", "track_genre", "popularity", "similarity"]
            ].to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/genre-profile/{genre}")
def genre_profile(genre: str):
    profile = get_genre_mood_profile(genre)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Genre '{genre}' not found")
    return profile

@app.get("/top-genres")
def top_genres(n: int = 10):
    return get_top_genres(n).to_dict(orient="records")

@app.get("/top-artists")
def top_artists(n: int = 10):
    return get_top_artists(n).to_dict(orient="records")

@app.post("/simulate-playlist")
def simulate_playlist_endpoint(req: PlaylistRequest):
    df = _load_data()
    result = _mc_simulate(df, req.track_names, req.n_simulations)
    if result is None:
        raise HTTPException(status_code=400, detail="Need at least 2 valid tracks in the dataset")
    return result
