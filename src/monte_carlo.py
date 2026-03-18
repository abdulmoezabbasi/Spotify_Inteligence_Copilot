"""Monte Carlo simulation for playlist engagement analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def simulate_playlist(
    df: pd.DataFrame,
    track_names: List[str],
    n_simulations: int = 10000
) -> Optional[Dict]:

    playlist_tracks = []
    for name in track_names:
        q = name.lower().strip()
        words = q.split()
        mask = None

        if len(words) >= 2:
            for split in range(1, len(words)):
                track_part  = " ".join(words[:split])
                artist_part = " ".join(words[split:])
                m = (
                    df['track_name'].str.lower().str.contains(track_part, na=False) &
                    df['artists'].str.lower().str.contains(artist_part, na=False)
                )
                if m.any():
                    mask = m
                    break

        if mask is None or not mask.any():
            mask = (
                df['track_name'].str.lower().str.contains(q, na=False) |
                df['artists'].str.lower().str.contains(q, na=False)
            )

        if mask.any():
            track = df[mask].sort_values('popularity', ascending=False).iloc[0]
            playlist_tracks.append(track)

    if len(playlist_tracks) < 2:
        return None

    n_tracks = len(playlist_tracks)
    energies = np.array([t['energy'] for t in playlist_tracks])
    popularities = np.array([t['popularity'] / 100.0 for t in playlist_tracks])

    completion_counts = np.zeros(n_tracks)
    full_completions = 0

    for _ in range(n_simulations):
        listening = True

        for i in range(n_tracks):
            if not listening:
                break

            base_retention = 0.6 + popularities[i] * 0.3

            if i > 0:
                energy_jump = abs(energies[i] - energies[i - 1])
                energy_penalty = energy_jump * 0.2
            else:
                energy_penalty = 0.0

            random_factor = np.random.normal(0, 0.05)
            retention_prob = np.clip(base_retention - energy_penalty + random_factor, 0, 1)

            if np.random.random() < retention_prob:
                completion_counts[i] += 1
            else:
                listening = False

        if listening:
            full_completions += 1

    full_completion_rate = full_completions / n_simulations * 100
    track_retention = (completion_counts / n_simulations * 100).tolist()
    avg_tracks_heard = sum(completion_counts) / n_simulations

    if full_completion_rate > 60:
        verdict = "Strong playlist 💪"
    elif full_completion_rate > 40:
        verdict = "Average playlist 👍"
    else:
        verdict = "High drop-off risk ⚠️"

    return {
        'playlist_length': n_tracks,
        'simulations': n_simulations,
        'full_completion_rate': round(full_completion_rate, 1),
        'track_retention': [round(r, 1) for r in track_retention],
        'avg_tracks_heard': round(avg_tracks_heard, 2),
        'track_names': [t['track_name'] for t in playlist_tracks],
        'verdict': verdict
    }