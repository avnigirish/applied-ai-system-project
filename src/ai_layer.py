"""
AI layer: Claude API integration.

Four-step agentic workflow (observable intermediate steps):
  Step 1  parse_user_query          — NL text → structured preferences (tool-use)
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

# ---------------------------------------------------------------------------
# Step 2 — Catalog reasoning tool (Agentic Enhancement)
# ---------------------------------------------------------------------------
_CATALOG_REASONING_TOOL = {
    "name": "catalog_decision",
    "description": (
        "After inspecting catalog coverage, decide how to proceed with these preferences."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "genre_in_catalog": {
                "type": "boolean",
                "description": "Is the requested genre present in the catalog?",
            },
            "mood_in_catalog": {
                "type": "boolean",
                "description": "Is the requested mood present in the catalog?",
            },
            "decision": {
                "type": "string",
                "enum": ["proceed", "suggest_alternative", "warn_degraded"],
                "description": (
                    "proceed: both genre and mood are available — run as-is. "
                    "suggest_alternative: genre is missing but a close match exists. "
                    "warn_degraded: genre is missing and no close alternative exists."
                ),
            },
            "suggested_genre": {
                "type": "string",
                "description": (
                    "If decision is suggest_alternative, the closest available genre. "
                    "Empty string otherwise."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the decision.",
            },
        },
        "required": [
            "genre_in_catalog", "mood_in_catalog",
            "decision", "suggested_genre", "reasoning",
        ],
    },
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
# Three hand-crafted examples that demonstrate a specific "warm vinyl DJ" voice.
# Including these in the prompt measurably shifts Claude's output style vs.
# the zero-shot baseline: shorter, more sensory, no hedging language.
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

    Uses Claude when ANTHROPIC_API_KEY is set; falls back to a rule-based check
    with hardcoded genre-similarity mappings otherwise.
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

    # --- Claude path ---
    genres_str = ", ".join(sorted(catalog_genres))
    moods_str  = ", ".join(sorted(catalog_moods))
    prompt = (
        f"A listener asked for: genre={requested_genre!r}, mood={requested_mood!r}.\n\n"
        f"Available catalog genres: {genres_str}\n"
        f"Available catalog moods:  {moods_str}\n\n"
        "Inspect coverage and decide how to proceed using the catalog_decision tool."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            tools=[_CATALOG_REASONING_TOOL],
            tool_choice={"type": "tool", "name": "catalog_decision"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "catalog_decision":
                result = dict(block.input)
                logger.info(
                    "Catalog decision: %s | suggested=%r | %s",
                    result.get("decision"), result.get("suggested_genre"), result.get("reasoning"),
                )
                return result
    except Exception as exc:
        logger.warning("Catalog reasoning step failed (%s); defaulting to proceed.", exc)

    # Final fallback: just proceed as-is
    return {
        "genre_in_catalog": genre_available,
        "mood_in_catalog":  mood_available,
        "decision": "proceed",
        "suggested_genre": "",
        "reasoning": "Could not complete reasoning step; proceeding with original preferences.",
    }


def generate_ai_explanation_fewshot(
    user_prefs: Dict,
    song: Dict,
    score: float,
    rule_reasons: List[str],
) -> str:
    """
    Step 4 variant — few-shot specialization.

    Uses 3 hand-crafted examples to constrain Claude to a specific
    'warm vinyl DJ' voice: sensory language, no hedging, one punchy sentence.
    Output measurably differs from the zero-shot generate_ai_explanation():
      - Shorter sentences
      - Metaphor/sensory imagery over feature description
      - No phrases like 'perfectly matches' or 'ideal for'

    Falls back to the zero-shot version if the API is unavailable.
    """
    client = _get_client()
    if client is None:
        return ", ".join(rule_reasons)

    # Build the few-shot block
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
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        explanation = response.content[0].text.strip().lstrip("Explanation:").strip()
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
