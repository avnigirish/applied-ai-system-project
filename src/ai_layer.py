"""
AI layer: Claude API integration for natural-language preference parsing
and AI-generated song explanations.

Agentic workflow:
  1. parse_user_query   — Claude converts free-text input to structured preferences
  2. generate_ai_explanation — Claude writes a conversational recommendation reason
"""

import logging
import os
from typing import Dict, List

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema used for structured extraction
# ---------------------------------------------------------------------------
_PREFERENCE_TOOL = {
    "name": "set_music_preferences",
    "description": (
        "Extract the listener's music preferences from their description. "
        "Choose the closest matching genre and mood from the catalog."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "genre": {
                "type": "string",
                "description": (
                    "Music genre. Choose from: pop, lofi, rock, ambient, jazz, "
                    "synthwave, indie pop, r&b, metal, classical, hip-hop, folk, "
                    "electronic, soul, country. Use the closest match."
                ),
            },
            "mood": {
                "type": "string",
                "description": (
                    "Desired mood. Choose from: happy, chill, intense, relaxed, "
                    "focused, moody, romantic, nostalgic, angry, melancholic, sad. "
                    "Use the closest match."
                ),
            },
            "target_energy": {
                "type": "number",
                "description": (
                    "Energy level from 0.0 (very calm/quiet) to 1.0 (very "
                    "energetic/loud). Use 0.3 for study/sleep, 0.6 for background, "
                    "0.9 for workout/dance."
                ),
            },
            "likes_acoustic": {
                "type": "boolean",
                "description": (
                    "True if the user prefers acoustic/organic sound, "
                    "False if they prefer electronic/produced sound."
                ),
            },
        },
        "required": ["genre", "mood", "target_energy", "likes_acoustic"],
    },
}

_VALID_GENRES = {
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop",
    "r&b", "metal", "classical", "hip-hop", "folk", "electronic", "soul", "country",
}
_VALID_MOODS = {
    "happy", "chill", "intense", "relaxed", "focused", "moody",
    "romantic", "nostalgic", "angry", "melancholic", "sad",
}


def _get_client() -> anthropic.Anthropic | None:
    """Return an Anthropic client, or None if no API key is configured."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def parse_user_query(natural_language: str) -> Dict:
    """
    Convert a free-text music request into structured preferences.

    Uses Claude when ANTHROPIC_API_KEY is set; falls back to keyword-based
    parsing otherwise. Always returns a dict with keys:
    genre, mood, target_energy, likes_acoustic.
    """
    logger.info("Parsing natural-language query: %r", natural_language)

    if not natural_language or not natural_language.strip():
        raise ValueError("Query cannot be empty.")

    client = _get_client()

    if client is None:
        logger.info("No API key found — using keyword-based parser.")
        prefs = _keyword_parse(natural_language)
        _validate_preferences(prefs)
        return prefs

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        tools=[_PREFERENCE_TOOL],
        tool_choice={"type": "tool", "name": "set_music_preferences"},
        messages=[
            {
                "role": "user",
                "content": (
                    "I need music recommendations. Here is what I want:\n\n"
                    f"{natural_language.strip()}\n\n"
                    "Extract my music preferences using the tool."
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "set_music_preferences":
            prefs = block.input
            logger.info(
                "Parsed preferences via Claude: genre=%s mood=%s energy=%.2f acoustic=%s",
                prefs.get("genre"), prefs.get("mood"),
                prefs.get("target_energy", 0), prefs.get("likes_acoustic"),
            )
            _validate_preferences(prefs)
            return prefs

    logger.error("Claude did not return a tool_use block; falling back to keyword parser.")
    prefs = _keyword_parse(natural_language)
    _validate_preferences(prefs)
    return prefs


def generate_ai_explanation(
    user_prefs: Dict,
    song: Dict,
    score: float,
    rule_reasons: List[str],
) -> str:
    """
    Generate a conversational, one-sentence explanation for why this song
    was recommended.

    Uses Claude when ANTHROPIC_API_KEY is set; falls back to rule-based
    explanation string otherwise.
    """
    fallback = ", ".join(rule_reasons)

    try:
        client = _get_client()
        if client is None:
            return fallback

        prompt = (
            "A music recommender selected this song for a listener. "
            "Write exactly one natural sentence (max 30 words) explaining why "
            "it was a good match. Sound warm and helpful, not technical.\n\n"
            f"Listener wants: genre={user_prefs['genre']}, "
            f"mood={user_prefs['mood']}, "
            f"energy={user_prefs['target_energy']:.2f}, "
            f"acoustic={'yes' if user_prefs['likes_acoustic'] else 'no'}\n"
            f"Song: \"{song['title']}\" by {song['artist']} "
            f"(genre={song['genre']}, mood={song['mood']}, "
            f"energy={song['energy']:.2f})\n"
            f"Match score: {score:.2f}/4.50\n"
            f"Scoring signals: {', '.join(rule_reasons)}\n\n"
            "Reply with only the one-sentence explanation."
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )

        explanation = response.content[0].text.strip()
        logger.debug("AI explanation for '%s': %s", song["title"], explanation)
        return explanation

    except Exception as exc:
        logger.warning(
            "AI explanation failed for '%s' (%s); using rule-based fallback.",
            song.get("title", "unknown"), exc,
        )
        return fallback


def _keyword_parse(text: str) -> Dict:
    """
    Simple keyword-based fallback parser for natural-language queries.
    Used when no Anthropic API key is configured.
    """
    t = text.lower()

    genre = "pop"
    for g, keywords in [
        ("lofi",       ["lofi", "lo-fi", "study", "studying"]),
        ("rock",       ["rock", "guitar", "band"]),
        ("classical",  ["classical", "piano", "orchestra"]),
        ("jazz",       ["jazz", "cafe", "coffee shop"]),
        ("hip-hop",    ["hip hop", "hip-hop", "rap"]),
        ("electronic", ["electronic", "edm", "synth"]),
        ("folk",       ["folk", "campfire", "acoustic guitar"]),
        ("ambient",    ["ambient", "background", "sleep"]),
        ("metal",      ["metal", "heavy"]),
        ("soul",       ["soul", "r&b", "rnb"]),
        ("country",    ["country", "cowboy"]),
        ("synthwave",  ["synthwave", "retrowave", "80s"]),
    ]:
        if any(kw in t for kw in keywords):
            genre = g
            break

    mood = "chill"
    for m, keywords in [
        ("happy",      ["happy", "upbeat", "fun", "cheerful", "party"]),
        ("intense",    ["intense", "pump", "workout", "hype", "aggressive"]),
        ("sad",        ["sad", "cry", "heartbreak", "down"]),
        ("focused",    ["focus", "concentrate", "productive", "work"]),
        ("relaxed",    ["relaxed", "peaceful", "mellow"]),
        ("romantic",   ["romantic", "love", "date"]),
        ("nostalgic",  ["nostalgic", "memories", "throwback"]),
        ("angry",      ["angry", "rage", "mad"]),
        ("moody",      ["moody", "dark", "atmospheric"]),
        ("melancholic",["melancholic", "melancholy", "bittersweet"]),
        ("chill",      ["chill", "calm", "lazy", "easy", "relax", "study"]),
    ]:
        if any(kw in t for kw in keywords):
            mood = m
            break

    high_energy_kws = ["workout", "gym", "pump", "hype", "dance", "intense", "run", "energy"]
    low_energy_kws  = ["sleep", "calm", "quiet", "relax", "chill", "study", "focus", "background"]
    if any(kw in t for kw in high_energy_kws):
        target_energy = 0.85
    elif any(kw in t for kw in low_energy_kws):
        target_energy = 0.35
    else:
        target_energy = 0.60

    acoustic_kws    = ["acoustic", "unplugged", "organic", "natural", "folk", "piano"]
    electronic_kws  = ["electronic", "edm", "produced", "synth", "beats", "digital"]
    if any(kw in t for kw in acoustic_kws):
        likes_acoustic = True
    elif any(kw in t for kw in electronic_kws):
        likes_acoustic = False
    else:
        likes_acoustic = False

    logger.info(
        "Keyword parse result: genre=%s mood=%s energy=%.2f acoustic=%s",
        genre, mood, target_energy, likes_acoustic,
    )
    return {"genre": genre, "mood": mood, "target_energy": target_energy, "likes_acoustic": likes_acoustic}


def _validate_preferences(prefs: Dict) -> None:
    """Warn (but don't crash) if Claude chose values outside the known catalog."""
    genre = prefs.get("genre", "")
    mood = prefs.get("mood", "")
    energy = prefs.get("target_energy", 0.5)

    if genre not in _VALID_GENRES:
        logger.warning(
            "Parsed genre %r is not in catalog; recommendations will lack genre points.", genre
        )
    if mood not in _VALID_MOODS:
        logger.warning("Parsed mood %r is not in catalog; mood match will not fire.", mood)
    if not (0.0 <= energy <= 1.0):
        logger.warning("Parsed energy %.2f is out of [0, 1]; clamping.", energy)
        prefs["target_energy"] = max(0.0, min(1.0, energy))
