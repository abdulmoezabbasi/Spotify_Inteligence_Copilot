"""Spotify Intelligence Copilot — Streamlit Dashboard"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from typing import Optional, Dict

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Spotify Intelligence Copilot",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #f0f0f0; }
    section[data-testid="stSidebar"] { background-color: #111 !important; border-right: 1px solid #222; }
    .stButton>button {
        background: #1DB954 !important; color: #000 !important;
        border-radius: 30px !important; border: none !important;
        font-weight: 700 !important; letter-spacing: 0.5px !important;
    }
    .stTextInput>div>div>input, .stTextArea textarea {
        background-color: #1a1a1a !important; color: #f0f0f0 !important;
        border: 1px solid #333 !important; border-radius: 8px !important;
    }
    .chat-bubble-user {
        background: #1DB954; color: #000; padding: 10px 16px;
        border-radius: 18px 18px 4px 18px; margin: 6px 0;
        max-width: 80%; margin-left: auto; font-weight: 500;
    }
    .chat-bubble-ai {
        background: #1a1a1a; color: #f0f0f0; padding: 12px 16px;
        border-radius: 18px 18px 18px 4px; margin: 6px 0;
        max-width: 85%; border: 1px solid #2a2a2a; line-height: 1.6;
    }
    .stat-card {
        background: #161616; border: 1px solid #222; border-radius: 10px;
        padding: 16px 20px; margin-bottom: 10px;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────

def api_get(path: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE_URL}{path}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def api_post(path: str, data: Dict) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=data, timeout=30)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def api_online() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=3).status_code == 200
    except:
        return False


# ── Session state ─────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None


def get_agent():
    if st.session_state.agent is not None:
        return st.session_state.agent
    if st.session_state.agent_error is not None:
        return None
    try:
        from src.agent import build_agent
        st.session_state.agent = build_agent()
    except Exception as e:
        st.session_state.agent_error = str(e)
    return st.session_state.agent


# ── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='color:#1DB954; margin-bottom:4px;'>🎵 Spotify Intel</h2>", unsafe_allow_html=True)
    st.caption("Music intelligence powered by real data on 114k tracks")
    st.divider()

    page = st.radio("", ["Chat", "Recommendations", "Genre Explorer", "Playlist Checker", "Rankings"], label_visibility="collapsed")

    st.divider()
    online = api_online()
    dot = "🟢" if online else "🔴"
    st.markdown(f"{dot} API {'online' if online else 'offline'}")
    if not online:
        st.caption("Run: `uvicorn api.main:app --reload`")


# ── Page: Chat ────────────────────────────────────────────

if page == "Chat":
    st.markdown("<h1 style='color:#1DB954; font-size:2rem;'>Ask anything about music</h1>", unsafe_allow_html=True)
    st.caption("Powered by a real AI agent with access to 114,000 Spotify tracks. Ask about artists, genres, playlists, recommendations — anything.")

    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    examples = [
        "Find songs similar to Blinding Lights",
        "What's the vibe of k-pop music?",
        "Will a high-energy dance track perform well?",
    ]
    for col, ex in zip([col1, col2, col3], examples):
        if col.button(ex, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": ex})
            st.rerun()

    st.divider()

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>{msg['content']}</div>", unsafe_allow_html=True)

    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        question = st.session_state.chat_history[-1]["content"]
        agent = get_agent()
        if agent is None:
            err = st.session_state.agent_error or "Unknown error"
            if "GROQ_API_KEY" in err:
                reply = "⚠️ GROQ_API_KEY is not set in your .env file. Add it to enable the AI chat. You can get a free key at console.groq.com."
            else:
                reply = f"⚠️ Could not start the AI agent: {err}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
        else:
            with st.spinner("Thinking..."):
                try:
                    from src.agent import run_agent
                    reply = run_agent(agent, question)
                except Exception as e:
                    reply = f"Something went wrong: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        user_input = col1.text_input("", placeholder="Ask about music...", label_visibility="collapsed")
        submitted = col2.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ── Page: Recommendations ─────────────────────────────────

elif page == "Recommendations":
    st.markdown("<h1 style='color:#1DB954; font-size:2rem;'>Find similar songs</h1>", unsafe_allow_html=True)
    st.caption("Enter any song and get 5 similar tracks based on sound, energy, and style.")

    query = st.text_input("Song name", placeholder="e.g. Blinding Lights")

    if st.button("Find similar songs") and query:
        with st.spinner("Searching..."):
            result = api_post("/recommend", {"track_name": query, "n": 6})

        if result and "recommendations" in result:
            st.success(f"Found {len(result['recommendations'])} similar tracks")
            for i, rec in enumerate(result["recommendations"], 1):
                with st.container():
                    c1, c2, c3 = st.columns([3, 2, 1])
                    c1.markdown(f"**{i}. {rec['track_name']}**  \n{rec['artists']}")
                    c2.caption(f"Genre: {rec['track_genre']}  \nPopularity: {rec['popularity']}/100")
                    sim = rec.get("similarity", 0)
                    c3.metric("Match", f"{sim:.0%}")
                    st.divider()
        elif result:
            st.warning(f"No results found for '{query}'. Try a different spelling.")
        else:
            st.error("Could not reach the API.")


# ── Page: Genre Explorer ──────────────────────────────────

elif page == "Genre Explorer":
    st.markdown("<h1 style='color:#1DB954; font-size:2rem;'>Explore a genre</h1>", unsafe_allow_html=True)
    st.caption("Get the sound profile and mood of any of 125 music genres.")

    genre = st.text_input("Genre name", placeholder="e.g. pop, metal, jazz, k-pop, synthwave")

    if st.button("Explore") and genre:
        with st.spinner("Analysing..."):
            result = api_get(f"/genre-profile/{genre}")

        if result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Genre", result.get("genre", genre).title())
            c2.metric("Mood", result.get("mood", "—"))
            c3.metric("Popularity", f"{result.get('avg_popularity', 0):.0f}/100")
            c4.metric("Tracks in dataset", f"{result.get('track_count', 0):,}")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sound profile")
                features = {
                    "Energy":       result.get("avg_energy", 0),
                    "Danceability": result.get("avg_danceability", 0),
                    "Positivity":   result.get("avg_valence", 0),
                }
                df_f = pd.DataFrame(features.items(), columns=["Feature", "Score"])
                fig = px.bar(df_f, x="Feature", y="Score", color="Score",
                             color_continuous_scale="Greens", template="plotly_dark",
                             range_y=[0, 1])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Top artists in this genre")
                for i, artist in enumerate(result.get("top_artists", [])[:5], 1):
                    st.markdown(f"**{i}.** {artist}")
        else:
            st.warning(f"Genre '{genre}' not found. Try a common genre name.")


# ── Page: Playlist Checker ────────────────────────────────

elif page == "Playlist Checker":
    st.markdown("<h1 style='color:#1DB954; font-size:2rem;'>Check your playlist</h1>", unsafe_allow_html=True)
    st.caption("Paste your playlist and we'll predict how many listeners will make it to the end — using 10,000 simulated plays.")

    tracks_input = st.text_area(
        "One track per line",
        placeholder="Blinding Lights\nShape of You\nBad Guy\nDespacito",
        height=160,
    )

    if st.button("Analyse playlist") and tracks_input:
        tracks = [t.strip() for t in tracks_input.strip().split("\n") if t.strip()]
        if len(tracks) < 2:
            st.warning("Add at least 2 tracks.")
        else:
            with st.spinner(f"Running 10,000 simulations on {len(tracks)} tracks..."):
                result = api_post("/simulate-playlist", {"track_names": tracks, "n_simulations": 10000})

            if result:
                rate = result["full_completion_rate"]
                verdict = result["verdict"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Finish rate", f"{rate}%")
                c2.metric("Avg tracks heard", f"{result['avg_tracks_heard']:.1f} / {result['playlist_length']}")
                c3.metric("Verdict", verdict)

                st.divider()
                st.subheader("Track-by-track drop-off")

                df_ret = pd.DataFrame({
                    "Track": result["track_names"],
                    "% of listeners still listening": result["track_retention"],
                })
                fig = px.area(df_ret, x="Track", y="% of listeners still listening",
                              color_discrete_sequence=["#1DB954"], template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Couldn't run the simulation. Make sure the API is running and track names match songs in the dataset.")


# ── Page: Rankings ────────────────────────────────────────

elif page == "Rankings":
    st.markdown("<h1 style='color:#1DB954; font-size:2rem;'>Rankings</h1>", unsafe_allow_html=True)
    st.caption("Top genres and artists by average popularity across 114,000 tracks.")

    tab1, tab2 = st.tabs(["Genres", "Artists"])

    with tab1:
        n = st.slider("How many to show", 5, 50, 20, key="genres_n")
        data = api_get(f"/top-genres?n={n}")
        if data:
            df = pd.DataFrame(data)
            fig = px.bar(df, x="avg_popularity", y="track_genre", orientation="h",
                         color="avg_popularity", color_continuous_scale="Greens",
                         template="plotly_dark", labels={"track_genre": "Genre", "avg_popularity": "Popularity"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        n = st.slider("How many to show", 5, 50, 20, key="artists_n")
        data = api_get(f"/top-artists?n={n}")
        if data:
            df = pd.DataFrame(data)
            fig = px.bar(df, x="avg_popularity", y="artists", orientation="h",
                         color="avg_popularity", color_continuous_scale="Teal",
                         template="plotly_dark", labels={"artists": "Artist", "avg_popularity": "Popularity"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)