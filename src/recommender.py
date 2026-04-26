import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score_song_obj(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []

        if song.genre == user.favorite_genre:
            score += 2.0
            reasons.append("genre match (+2.0)")

        if song.mood == user.favorite_mood:
            score += 1.0
            reasons.append("mood match (+1.0)")

        energy_proximity = 1.0 - abs(user.target_energy - song.energy)
        score += 1.0 * energy_proximity
        reasons.append(f"energy proximity (+{energy_proximity:.2f})")

        acousticness_target = 0.8 if user.likes_acoustic else 0.2
        acousticness_proximity = 1.0 - abs(acousticness_target - song.acousticness)
        score += 0.5 * acousticness_proximity
        reasons.append(f"acousticness proximity (+{acousticness_proximity:.2f})")

        return score, reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        logger.debug("Recommender.recommend called: genre=%s mood=%s energy=%.2f k=%d",
                     user.favorite_genre, user.favorite_mood, user.target_energy, k)
        scored = []
        for song in self.songs:
            score, _ = self._score_song_obj(user, song)
            scored.append((song, score))

        scored.sort(key=lambda x: (x[1], -abs(user.target_energy - x[0].energy)), reverse=True)
        results = [s for s, _ in scored[:k]]
        logger.debug("Top recommendation: %s (score=%.2f)", results[0].title if results else "none",
                     scored[0][1] if scored else 0)
        return results

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = self._score_song_obj(user, song)
        explanation = ", ".join(reasons)
        logger.debug("Explanation for '%s': %s", song.title, explanation)
        return explanation

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with numeric fields cast to int/float."""
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    logger.info("Loaded %d songs from %s", len(songs), csv_path)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song 0–4.5 pts using genre (+2.0), mood (+1.0), energy (+1.0), and acousticness (+0.5)."""
    score = 0.0
    reasons = []

    if song["genre"] == user_prefs["genre"]:
        score += 2.0
        reasons.append("genre match (+2.0)")

    if song["mood"] == user_prefs["mood"]:
        score += 1.0
        reasons.append("mood match (+1.0)")

    energy_proximity = 1.0 - abs(user_prefs["target_energy"] - song["energy"])
    score += 1.0 * energy_proximity
    reasons.append(f"energy proximity (+{1.0 * energy_proximity:.2f})")

    target_acousticness = 0.8 if user_prefs["likes_acoustic"] else 0.2
    acousticness_proximity = 1.0 - abs(target_acousticness - song["acousticness"])
    score += 0.5 * acousticness_proximity
    reasons.append(f"acousticness proximity (+{0.5 * acousticness_proximity:.2f})")

    logger.debug("Scored '%s': %.2f  [%s]", song["title"], score, ", ".join(reasons))
    return score, reasons


_MAX_SCORE = 4.5


def confidence_score(score: float) -> float:
    """Normalize a raw 0–4.5 score to a 0–1 confidence value."""
    return round(min(max(score / _MAX_SCORE, 0.0), 1.0), 3)


def confidence_band(score: float) -> str:
    """Return 'high' (≥0.75), 'medium' (≥0.45), or 'low' for a raw score."""
    c = confidence_score(score)
    if c >= 0.75:
        return "high"
    if c >= 0.45:
        return "medium"
    return "low"


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort descending, return top-k as (song, score, explanation) tuples."""
    if not songs:
        logger.warning("recommend_songs called with empty catalog")
        return []

    logger.info(
        "recommend_songs: genre=%s mood=%s energy=%.2f acoustic=%s k=%d",
        user_prefs.get("genre"), user_prefs.get("mood"),
        user_prefs.get("target_energy", 0), user_prefs.get("likes_acoustic"), k,
    )

    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = ", ".join(reasons)
        scored.append((song, score, explanation))

    scored.sort(key=lambda x: (x[1], -abs(user_prefs["target_energy"] - x[0]["energy"])), reverse=True)
    results = scored[:k]

    if results:
        top_song, top_score, _ = results[0]
        band = confidence_band(top_score)
        logger.info("Top result: '%s' score=%.2f confidence=%.2f (%s)",
                    top_song["title"], top_score, confidence_score(top_score), band)
        if band == "low":
            logger.warning(
                "LOW CONFIDENCE (%.2f): best match for genre=%r is '%s' — "
                "genre may be absent from catalog or preferences conflict.",
                confidence_score(top_score), user_prefs.get("genre"), top_song["title"],
            )
        elif band == "medium":
            catalog_genres = {s["genre"] for s in songs}
            if user_prefs.get("genre") not in catalog_genres:
                logger.warning(
                    "DEGRADED RESULTS: genre=%r not in catalog — "
                    "recommendations are mood/energy fallbacks only.",
                    user_prefs.get("genre"),
                )

    return results
