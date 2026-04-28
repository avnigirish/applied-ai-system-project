"""
AI layer: Gemini API integration (google-genai SDK).

Four-step agentic workflow (observable intermediate steps):
  Step 1  parse_user_query          — NL text → structured preferences (function calling)
  Step 2  catalog_reasoning_step    — inspect catalog coverage, decide how to proceed
  Step 3  [scoring engine]          — rule-based scoring in recommender.py
  Step 4  generate_ai_explanation   — conversational explanation (zero-shot or few-shot)

Fine-tuning / specialization:
  generate_ai_explanation_fewshot   — same task as Step 4 but constrained by
                                       3 few-shot examples in a "warm vinyl DJ" voice;
                                       output measurably differs from the zero-shot baseline
"""

import logging
import os
from typing import Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
_PREFERENCE_FN = types.FunctionDeclaration(
    name="set_music_preferences",
    description=(
        "Extract the listener's music preferences from their description. "
        "Choose the closest matching genre and mood from the catalog."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "genre": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Music genre. Choose from: pop, lofi, rock, ambient, jazz, "
                    "synthwave, indie pop, r&b, metal, classical, hip-hop, folk, "
                    "electronic, soul, country. Use the closest match."
                ),
            ),
            "mood": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Desired mood. Choose from: happy, chill, intense, relaxed, "
                    "focused, moody, romantic, nostalgic, angry, melancholic, sad. "
                    "Use the closest match."
                ),
            ),
            "target_energy": types.Schema(
                type=types.Type.NUMBER,
                description=(
                    "Energy level from 0.0 (very calm/quiet) to 1.0 (very "
                    "energetic/loud). Use 0.3 for study/sleep, 0.6 for background, "
                    "0.9 for workout/dance."
                ),
            ),
            "likes_acoustic": types.Schema(
                type=types.Type.BOOLEAN,
                description=(
                    "True if the user prefers acoustic/organic sound, "
                    "False if they prefer electronic/produced sound."
                ),
            ),
        },
        required=["genre", "mood", "target_energy", "likes_acoustic"],
    ),
)

_CATALOG_FN = types.FunctionDeclaration(
    name="catalog_decision",
    description="After inspecting catalog coverage, decide how to proceed with these preferences.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "genre_in_catalog": types.Schema(
                type=types.Type.BOOLEAN,
                description="Is the requested genre present in the catalog?",
            ),
            "mood_in_catalog": types.Schema(
                type=types.Type.BOOLEAN,
                description="Is the requested mood present in the catalog?",
            ),
            "decision": types.Schema(
                type=types.Type.STRING,
                description=(
                    "proceed: both genre and mood are available — run as-is. "
                    "suggest_alternative: genre is missing but a close match exists. "
                    "warn_degraded: genre is missing and no close alternative exists."
                ),
            ),
            "suggested_genre": types.Schema(
                type=types.Type.STRING,
                description=(
                    "If decision is suggest_alternative, the closest available genre. "
                    "Empty string otherwise."
                ),
            ),
            "reasoning": types.Schema(
                type=types.Type.STRING,
                description="One sentence explaining the decision.",
            ),
        },
        required=["genre_in_catalog", "mood_in_catalog", "decision", "suggested_genre", "reasoning"],
    ),
)

_VALID_GENRES = {
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop",
    "r&b", "metal", "classical", "hip-hop", "folk", "electronic", "soul", "country",
}
_VALID_MOODS = {
    "happy", "chill", "intense", "relaxed", "focused", "moody",
    "romantic", "nostalgic", "angry", "melancholic", "sad",
}

# Known closest-genre mappings used by the keyword fallback for Step 2
_GENRE_FALLBACKS = {
    "bossa nova": "jazz",
    "reggae":     "folk",
    "blues":      "soul",
    "punk":       "rock",
    "k-pop":      "pop",
    "latin":      "pop",
    "trap":       "hip-hop",
    "gospel":     "soul",
    "edm":        "electronic",
    "indie":      "indie pop",
}

# ---------------------------------------------------------------------------
# Step 4 — Few-shot examples (Fine-Tuning / Specialization stretch)
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES = [
    {
        "song":    "Library Rain by Paper Lanterns",
        "context": "listener wants lofi, chill, low energy, acoustic",
        "explanation": (
            "This one sounds like rain on a window — soft and unhurried, "
            "the kind of track that turns studying into something almost peaceful."
        ),
    },
    {
        "song":    "Gym Hero by Max Pulse",
        "context": "listener wants pop, intense, high energy, electronic",
        "explanation": (
            "Exactly what the name promises — it hits hard from the first beat "
            "and doesn't let you slow down."
        ),
    },
    {
        "song":    "Rainy Sunday by Clara Voss",
        "context": "listener wants classical, melancholic, very low energy, acoustic",
        "explanation": (
            "A solo piano piece that feels like a Sunday that decided to stay grey — "
            "not sad exactly, just beautifully still."
        ),
    },
]


def _get_client() -> Optional[genai.Client]:
    """Return a configured Gemini client, or None if GEMINI_API_KEY is not set."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _extract_fn_args(response, fn_name: str) -> Optional[Dict]:
    """Pull function-call arguments from a Gemini response, or return None."""
    try:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == fn_name:
                    return dict(fc.args)
    except Exception:
        pass
    return None


def parse_user_query(natural_language: str) -> Dict:
    """
    Convert a free-text music request into structured preferences.

    Uses Gemini when GEMINI_API_KEY is set; falls back to keyword-based
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

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=(
                "I need music recommendations. Here is what I want:\n\n"
                f"{natural_language.strip()}\n\n"
                "Extract my music preferences using the set_music_preferences function."
            ),
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[_PREFERENCE_FN])],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["set_music_preferences"],
                    )
                ),
            ),
        )
        prefs = _extract_fn_args(response, "set_music_preferences")
        if prefs is not None:
            logger.info(
                "Parsed preferences via Gemini: genre=%s mood=%s energy=%.2f acoustic=%s",
                prefs.get("genre"), prefs.get("mood"),
                prefs.get("target_energy", 0), prefs.get("likes_acoustic"),
            )
            _validate_preferences(prefs)
            return prefs
    except Exception as exc:
        logger.warning("Gemini parse_user_query failed (%s); falling back to keyword parser.", exc)

    logger.warning("Gemini did not return a function call; falling back to keyword parser.")
    prefs = _keyword_parse(natural_language)
    _validate_preferences(prefs)
    return prefs


def catalog_reasoning_step(
    user_prefs: Dict,
    catalog_genres: List[str],
    catalog_moods: List[str],
) -> Dict:
    """
    Step 2 of the agentic chain: inspect catalog coverage and decide how to proceed.

    Returns a dict with keys:
      genre_in_catalog  bool
      mood_in_catalog   bool
      decision          "proceed" | "suggest_alternative" | "warn_degraded"
      suggested_genre   str  (non-empty only when decision == "suggest_alternative")
      reasoning         str  (one-sentence explanation)
    """
    requested_genre = user_prefs.get("genre", "")
    requested_mood  = user_prefs.get("mood", "")

    genre_available = requested_genre in catalog_genres
    mood_available  = requested_mood  in catalog_moods

    logger.info(
        "Catalog reasoning: genre=%r %s, mood=%r %s",
        requested_genre, "✓" if genre_available else "✗",
        requested_mood,  "✓" if mood_available  else "✗",
    )

    client = _get_client()

    # --- Fallback path (no API key) ---
    if client is None:
        if genre_available:
            return {
                "genre_in_catalog": True,
                "mood_in_catalog": mood_available,
                "decision": "proceed",
                "suggested_genre": "",
                "reasoning": "Requested genre is available in catalog.",
            }
        alt = _GENRE_FALLBACKS.get(requested_genre, "")
        if alt and alt in catalog_genres:
            return {
                "genre_in_catalog": False,
                "mood_in_catalog": mood_available,
                "decision": "suggest_alternative",
                "suggested_genre": alt,
                "reasoning": (
                    f"'{requested_genre}' is not in catalog; "
                    f"'{alt}' is the closest available match."
                ),
            }
        return {
            "genre_in_catalog": False,
            "mood_in_catalog": mood_available,
            "decision": "warn_degraded",
            "suggested_genre": "",
            "reasoning": (
                f"'{requested_genre}' is not in catalog and no close alternative is known; "
                "proceeding with mood/energy signals only."
            ),
        }

    # --- Gemini path ---
    genres_str = ", ".join(sorted(catalog_genres))
    moods_str  = ", ".join(sorted(catalog_moods))
    prompt = (
        f"A listener asked for: genre={requested_genre!r}, mood={requested_mood!r}.\n\n"
        f"Available catalog genres: {genres_str}\n"
        f"Available catalog moods:  {moods_str}\n\n"
        "Inspect coverage and decide how to proceed using the catalog_decision function."
    )

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[_CATALOG_FN])],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["catalog_decision"],
                    )
                ),
            ),
        )
        result = _extract_fn_args(response, "catalog_decision")
        if result is not None:
            logger.info(
                "Catalog decision: %s | suggested=%r | %s",
                result.get("decision"), result.get("suggested_genre"), result.get("reasoning"),
            )
            return result
    except Exception as exc:
        logger.warning("Catalog reasoning step failed (%s); using keyword fallback.", exc)

    # Gemini unavailable — apply the same keyword logic as the no-key path
    if genre_available:
        return {
            "genre_in_catalog": True,
            "mood_in_catalog": mood_available,
            "decision": "proceed",
            "suggested_genre": "",
            "reasoning": "Requested genre is available in catalog.",
        }
    alt = _GENRE_FALLBACKS.get(requested_genre, "")
    if alt and alt in catalog_genres:
        return {
            "genre_in_catalog": False,
            "mood_in_catalog": mood_available,
            "decision": "suggest_alternative",
            "suggested_genre": alt,
            "reasoning": (
                f"'{requested_genre}' is not in catalog; "
                f"'{alt}' is the closest available match."
            ),
        }
    return {
        "genre_in_catalog": False,
        "mood_in_catalog": mood_available,
        "decision": "warn_degraded",
        "suggested_genre": "",
        "reasoning": (
            f"'{requested_genre}' is not in catalog and no close alternative is known; "
            "proceeding with mood/energy signals only."
        ),
    }


def generate_ai_explanation(
    user_prefs: Dict,
    song: Dict,
    score: float,
    rule_reasons: List[str],
) -> str:
    """
    Generate a conversational, one-sentence explanation for why this song
    was recommended. Falls back to rule-based string if API is unavailable.
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

        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
        )
        explanation = response.text.strip()
        logger.debug("AI explanation for '%s': %s", song["title"], explanation)
        return explanation

    except Exception as exc:
        logger.warning(
            "AI explanation failed for '%s' (%s); using rule-based fallback.",
            song.get("title", "unknown"), exc,
        )
        return fallback


def generate_ai_explanation_fewshot(
    user_prefs: Dict,
    song: Dict,
    score: float,
    rule_reasons: List[str],
) -> str:
    """
    Step 4 variant — few-shot specialization.

    Uses 3 hand-crafted examples to constrain Gemini to a specific
    'warm vinyl DJ' voice: sensory language, no hedging, one punchy sentence.
    Falls back to the zero-shot version if the API is unavailable.
    """
    client = _get_client()
    if client is None:
        return ", ".join(rule_reasons)

    examples_block = "\n\n".join(
        f"Song: {ex['song']}\nContext: {ex['context']}\nExplanation: {ex['explanation']}"
        for ex in _FEW_SHOT_EXAMPLES
    )

    prompt = (
        "You write music recommendations in a warm, sensory, vinyl-DJ voice. "
        "One sentence. No hedging. Use a concrete image or feeling, not feature names.\n\n"
        "Examples:\n\n"
        f"{examples_block}\n\n"
        "---\n"
        f"Song: \"{song['title']}\" by {song['artist']}\n"
        f"Context: listener wants {user_prefs['genre']}, {user_prefs['mood']}, "
        f"energy={user_prefs['target_energy']:.2f}, "
        f"acoustic={'yes' if user_prefs['likes_acoustic'] else 'no'}\n"
        "Explanation:"
    )

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
        )
        explanation = response.text.strip().lstrip("Explanation:").strip()
        logger.debug("Few-shot explanation for '%s': %s", song["title"], explanation)
        return explanation
    except Exception as exc:
        logger.warning(
            "Few-shot explanation failed for '%s' (%s); falling back to zero-shot.",
            song.get("title", "unknown"), exc,
        )
        return generate_ai_explanation(user_prefs, song, score, rule_reasons)


def _keyword_parse(text: str) -> Dict:
    """
    Simple keyword-based fallback parser for natural-language queries.
    Used when no Gemini API key is configured.
    """
    t = text.lower()

    genre = "pop"
    for g, keywords in [
        ("lofi",       ["lofi", "lo-fi", "study", "studying"]),
        ("rock",       ["rock", "guitar", "band"]),
        ("classical",  ["classical", "piano", "orchestra"]),
        ("bossa nova", ["bossa nova", "bossa"]),
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
        ("happy",       ["happy", "upbeat", "fun", "cheerful", "party"]),
        ("intense",     ["intense", "pump", "workout", "hype", "aggressive"]),
        ("sad",         ["sad", "cry", "heartbreak", "down"]),
        ("focused",     ["focus", "concentrate", "productive", "work"]),
        ("relaxed",     ["relaxed", "peaceful", "mellow"]),
        ("romantic",    ["romantic", "love", "date"]),
        ("nostalgic",   ["nostalgic", "memories", "throwback"]),
        ("angry",       ["angry", "rage", "mad"]),
        ("moody",       ["moody", "dark", "atmospheric"]),
        ("melancholic", ["melancholic", "melancholy", "bittersweet"]),
        ("chill",       ["chill", "calm", "lazy", "easy", "relax", "study"]),
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

    acoustic_kws   = ["acoustic", "unplugged", "organic", "natural", "folk", "piano"]
    electronic_kws = ["electronic", "edm", "produced", "synth", "beats", "digital"]
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
    """Warn (but don't crash) if the parsed values fall outside the known catalog."""
    genre  = prefs.get("genre", "")
    mood   = prefs.get("mood", "")
    energy = prefs.get("target_energy", 0.5)

    if genre not in _VALID_GENRES:
        logger.warning(
            "Parsed genre %r is not in catalog; catalog_reasoning_step will handle the fallback.", genre
        )
    if mood not in _VALID_MOODS:
        logger.warning("Parsed mood %r is not in catalog; mood match will not fire.", mood)
    if not (0.0 <= energy <= 1.0):
        logger.warning("Parsed energy %.2f is out of [0, 1]; clamping.", energy)
        prefs["target_energy"] = max(0.0, min(1.0, energy))
