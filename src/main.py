"""
Music Recommender — CLI runner.

Modes:
  python -m src.main               → run all built-in test profiles (no API key needed)
  python -m src.main --ai          → interactive natural-language mode (requires ANTHROPIC_API_KEY)
  python -m src.main --ai "query"  → single query in natural-language mode
"""

import argparse
import logging
import os
import sys

# Ensure src/ is on the path regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(__file__))

from recommender import load_songs, recommend_songs, score_song, confidence_score, confidence_band
from ai_layer import parse_user_query, generate_ai_explanation

# ---------------------------------------------------------------------------
# Logging setup — INFO to stdout, DEBUG available via LOG_LEVEL=DEBUG
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared display helpers
# ---------------------------------------------------------------------------

def _print_recommendations(label: str, user_prefs: dict, songs: list, k: int = 5,
                            use_ai_explanations: bool = False) -> None:
    recommendations = recommend_songs(user_prefs, songs, k=k)
    print(f"\nProfile: {label}")
    print(f"  genre={user_prefs['genre']!r}, mood={user_prefs['mood']!r}, "
          f"energy={user_prefs['target_energy']}, acoustic={user_prefs['likes_acoustic']}")
    print("=" * 52)
    print(f"  Top {len(recommendations)} Recommendations")
    print("=" * 52)

    for i, (song, score, rule_explanation) in enumerate(recommendations, 1):
        if use_ai_explanations:
            _, reasons = score_song(user_prefs, song)
            explanation = generate_ai_explanation(user_prefs, song, score, reasons)
        else:
            explanation = rule_explanation

        conf = confidence_score(score)
        band = confidence_band(score)
        print(f"\n  #{i}  {song['title']} — {song['artist']}")
        print(f"       Score      : {score:.2f} / 4.50  (confidence {conf:.2f} — {band})")
        print(f"       Why        : {explanation}")

    print()


# ---------------------------------------------------------------------------
# Standard profile battery (no API key required)
# ---------------------------------------------------------------------------

def run_standard_profiles(songs: list) -> None:
    logger.info("Running standard profile battery (%d songs in catalog)", len(songs))

    _print_recommendations("High-Energy Pop", {
        "genre": "pop", "mood": "intense",
        "target_energy": 0.92, "likes_acoustic": False,
    }, songs)

    _print_recommendations("Chill Lofi", {
        "genre": "lofi", "mood": "chill",
        "target_energy": 0.38, "likes_acoustic": True,
    }, songs)

    _print_recommendations("Deep Intense Rock", {
        "genre": "rock", "mood": "intense",
        "target_energy": 0.90, "likes_acoustic": False,
    }, songs)

    _print_recommendations("EDGE: High Energy + Sad Mood", {
        "genre": "soul", "mood": "sad",
        "target_energy": 0.90, "likes_acoustic": False,
    }, songs)

    _print_recommendations("EDGE: Unknown Genre", {
        "genre": "bossa nova", "mood": "relaxed",
        "target_energy": 0.40, "likes_acoustic": True,
    }, songs)

    _print_recommendations("EDGE: Max Acoustic Preference", {
        "genre": "classical", "mood": "melancholic",
        "target_energy": 0.21, "likes_acoustic": True,
    }, songs)


# ---------------------------------------------------------------------------
# Natural-language / AI mode
# ---------------------------------------------------------------------------

def run_ai_mode(songs: list, query: str | None = None) -> None:
    """Parse a free-text music request with Claude, then show AI-explained recommendations."""
    if query is None:
        print("\nDescribe the kind of music you want (e.g. 'something chill for studying'):")
        query = input("> ").strip()
        if not query:
            logger.warning("Empty query entered; exiting.")
            sys.exit(0)

    logger.info("AI mode query: %r", query)

    try:
        prefs = parse_user_query(query)
    except EnvironmentError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\nCould not parse your request: {exc}")
        sys.exit(1)

    _print_recommendations(
        label=f"AI Query: {query!r}",
        user_prefs=prefs,
        songs=songs,
        k=5,
        use_ai_explanations=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Music Recommender — rule-based scoring with optional Claude AI layer"
    )
    parser.add_argument(
        "--ai",
        nargs="?",
        const=True,
        metavar="QUERY",
        help=(
            "Enable natural-language mode. "
            "Pass a quoted query directly, or omit to enter one interactively. "
            "Requires ANTHROPIC_API_KEY."
        ),
    )
    args = parser.parse_args()

    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from catalog.")

    if args.ai is not None:
        # --ai with an inline query string, or --ai alone (interactive)
        inline_query = args.ai if isinstance(args.ai, str) else None
        run_ai_mode(songs, query=inline_query)
    else:
        run_standard_profiles(songs)


if __name__ == "__main__":
    main()
