"""Music Intelligence Agent — LangChain + Groq."""

import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


@tool
def tool_get_top_genres() -> str:
    """Returns top music genres ranked by average popularity."""
    from src.analytics import get_top_genres
    df = get_top_genres(10)
    lines = ["Top 10 genres by popularity:"]
    for _, row in df.iterrows():
        lines.append(
            f"  {row['track_genre'].title():25s} popularity: {row['avg_popularity']:.1f} "
            f"| energy: {row['avg_energy']:.2f} | danceability: {row['avg_danceability']:.2f}"
        )
    return "\n".join(lines)


@tool
def tool_get_top_artists() -> str:
    """Returns top artists ranked by average popularity."""
    from src.analytics import get_top_artists
    df = get_top_artists(10)
    lines = ["Top 10 artists by popularity:"]
    for _, row in df.iterrows():
        lines.append(
            f"  {str(row['artists'])[:35]:35s} popularity: {row['avg_popularity']:.1f} "
            f"| tracks: {int(row['track_count'])}"
        )
    return "\n".join(lines)


@tool
def tool_find_similar_tracks(track_name: str) -> str:
    """Finds songs similar to a given track using ML. Use for recommendations."""
    try:
        r = requests.post(f"{API_BASE}/recommend", json={"track_name": track_name, "n": 5}, timeout=10)
        data = r.json()
        if "recommendations" not in data:
            return f"Track '{track_name}' not found in the dataset."
        lines = [f"Tracks similar to '{track_name}':"]
        for rec in data["recommendations"]:
            lines.append(
                f"  {rec['track_name'][:35]:35s} by {str(rec['artists'])[:25]:25s} "
                f"| genre: {rec['track_genre']} | popularity: {rec['popularity']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error finding similar tracks: {e}"


@tool
def tool_get_genre_mood(genre: str) -> str:
    """Returns mood analysis and audio profile of a music genre."""
    try:
        r = requests.get(f"{API_BASE}/genre-profile/{genre}", timeout=10)
        if r.status_code == 404:
            return f"Genre '{genre}' not found."
        data = r.json()
        return (
            f"Genre: {data['genre'].title()}\n"
            f"Mood: {data['mood']}\n"
            f"Avg Popularity: {data['avg_popularity']}\n"
            f"Energy: {data['avg_energy']} | Valence: {data['avg_valence']} | Danceability: {data['avg_danceability']}\n"
            f"Track count: {data['track_count']}\n"
            f"Top artists: {', '.join(data['top_artists'][:3])}"
        )
    except Exception as e:
        return f"Error fetching genre profile: {e}"


@tool
def tool_predict_popularity(
    danceability: float = 0.5,
    energy: float = 0.7,
    loudness: float = -6.0,
    valence: float = 0.5,
    tempo: float = 120.0,
    acousticness: float = 0.1,
    genre_avg_popularity: float = 47.0,
    artist_avg_popularity: float = 50.0,
) -> str:
    """Predicts how popular a track will be using a neural network."""
    try:
        payload = {
            "danceability": float(danceability),
            "energy": float(energy),
            "loudness": float(loudness),
            "speechiness": 0.05,
            "acousticness": float(acousticness),
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": float(valence),
            "tempo": float(tempo),
            "genre_avg_popularity": float(genre_avg_popularity),
            "artist_avg_popularity": float(artist_avg_popularity),
        }
        r = requests.post(f"{API_BASE}/predict-popularity", json=payload, timeout=10)
        data = r.json()
        return (
            f"Predicted popularity: {data['predicted_popularity']}/100\n"
            f"Confidence range: {data['confidence_range']['low']} – {data['confidence_range']['high']}\n"
            f"Tier: {data['tier']}\n"
            f"{data['interpretation']}"
        )
    except Exception as e:
        return f"Error predicting popularity: {e}"


@tool
def tool_simulate_playlist(track_names: str) -> str:
    """Runs Monte Carlo simulation to predict playlist engagement. Pass track names as comma-separated string."""
    try:
        tracks = [t.strip() for t in track_names.split(",")]
        r = requests.post(
            f"{API_BASE}/simulate-playlist",
            json={"track_names": tracks, "n_simulations": 10000},
            timeout=30,
        )
        data = r.json()
        lines = [
            f"Playlist simulation ({len(tracks)} tracks, 10,000 scenarios):",
            f"Full completion rate: {data['full_completion_rate']}%",
            f"Avg tracks heard: {data['avg_tracks_heard']}/{data['playlist_length']}",
            f"Verdict: {data['verdict']}",
            "Track retention:",
        ]
        for track, retention in zip(data["track_names"], data["track_retention"]):
            bar = "█" * int(retention / 10)
            lines.append(f"  {track[:30]:30s} {retention}% {bar}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error running simulation: {e}"


TOOLS = [
    tool_get_top_genres,
    tool_get_top_artists,
    tool_find_similar_tracks,
    tool_get_genre_mood,
    tool_predict_popularity,
    tool_simulate_playlist,
]

SYSTEM_PROMPT = """You are a music intelligence AI with access to real data on 114,000 Spotify tracks across 125 genres.

When answering:
- Use the appropriate tools to get real data before answering
- Give specific recommendations with actual track names, artist names, and numbers
- Sound like a knowledgeable music analyst, not a generic chatbot
- Write in plain English paragraphs only — no bullet lists, no tool names, no code
- If a question involves recommendations, always call the similarity tool
- If a question involves a playlist, always run the simulation
- Use multiple tools when the question calls for it

Never mention tool names, function calls, or technical internals in your response."""


def build_agent():
    """Build and return the music intelligence agent."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(agent, question: str) -> str:
    """Run the agent on a question and return the response string."""
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content