import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from functools import lru_cache

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

@lru_cache(maxsize=1)
def _load_data():
    return pd.read_csv(os.path.join(DATA, "spotify_clean.csv"))

@lru_cache(maxsize=1)
def _load_similarity():
    df = _load_data()
    scaler = pickle.load(open(os.path.join(MODELS, "similarity_scaler.pkl"), "rb"))
    matrix = np.load(os.path.join(MODELS, "feature_matrix.npy"))
    return df, scaler, matrix

SIMILARITY_FEATURES = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness","valence",
    "tempo","mood_score","dance_energy"
]

def get_top_genres(n=10):
    df = _load_data()
    return df.groupby("track_genre").agg(
        avg_popularity=("popularity","mean"),
        track_count=("track_name","count"),
        avg_danceability=("danceability","mean"),
        avg_energy=("energy","mean"),
        avg_valence=("valence","mean"),
        mood_score=("mood_score","mean")
    ).reset_index().sort_values("avg_popularity", ascending=False).head(n).round(3)

def get_top_artists(n=10):
    df = _load_data()
    stats = df.groupby("artists").agg(
        avg_popularity=("popularity","mean"),
        track_count=("track_name","count"),
        avg_energy=("energy","mean"),
        avg_danceability=("danceability","mean")
    ).reset_index()
    return stats[stats["track_count"] >= 2].sort_values("avg_popularity", ascending=False).head(n).round(3)

def _search_mask(df, query):
    """Smart search: tries 'track + artist' split first, falls back to either field."""
    q = query.lower().strip()
    words = q.split()

    if len(words) >= 2:
        for split in range(1, len(words)):
            track_part  = " ".join(words[:split])
            artist_part = " ".join(words[split:])
            mask = (
                df["track_name"].str.lower().str.contains(track_part, na=False) &
                df["artists"].str.lower().str.contains(artist_part, na=False)
            )
            if mask.any():
                return mask

    return (
        df["track_name"].str.lower().str.contains(q, na=False) |
        df["artists"].str.lower().str.contains(q, na=False)
    )

def find_similar_tracks(track_name, n=5):
    df, scaler, matrix = _load_similarity()
    mask = _search_mask(df, track_name)
    if not mask.any():
        return None
    idx = df[mask]["popularity"].idxmax()
    sims = cosine_similarity(matrix[idx].reshape(1,-1), matrix)[0]
    top_idx = sims.argsort()[::-1][1:n+1]
    results = df.iloc[top_idx][["track_name","artists","track_genre","popularity"]].copy()
    results["similarity"] = sims[top_idx].round(3)
    results = results[results["popularity"] > 10]
    return results.head(n)

def get_genre_mood_profile(genre):
    df = _load_data()
    g = df[df["track_genre"].str.lower().str.contains(genre.lower())]
    if len(g) == 0:
        return None
    v = g["valence"].mean()
    e = g["energy"].mean()
    if v > 0.6 and e > 0.6: mood = "Energetic & Happy"
    elif v > 0.6: mood = "Calm & Positive"
    elif e > 0.6: mood = "Intense & Dark"
    elif v <= 0.4 and e <= 0.4: mood = "Melancholic & Quiet"
    else: mood = "Balanced"
    return {
        "genre": genre, "mood": mood,
        "avg_popularity": round(g["popularity"].mean(),1),
        "avg_energy": round(e,3),
        "avg_valence": round(v,3),
        "avg_danceability": round(g["danceability"].mean(),3),
        "track_count": len(g),
        "top_artists": g.nlargest(3,"popularity")["artists"].tolist()
    }