"""
Reliability evaluation harness for the music recommender.

Measures two things separately:
  1. Consistency  — does the system always return the deterministic expected result?
  2. Quality      — is that result actually a good musical match?

Some cases are deliberately adversarial (marked musically_correct=False) to expose
known limitations rather than hide them. This distinction matters: a consistent but
wrong output is a design problem, not a code bug.

Run with:
    python tests/eval_reliability.py

Exit code 0 if all consistency checks pass; 1 otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import load_songs, recommend_songs, confidence_score, confidence_band

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
# Each case defines:
#   prefs            — user preference dict
#   expected_top     — title of the expected #1 result (deterministic)
#   musically_correct — whether that result is actually a good recommendation
#   note             — human-readable explanation of what this case tests
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "name": "High-Energy Pop",
        "prefs": {
            "genre": "pop", "mood": "intense",
            "target_energy": 0.92, "likes_acoustic": False,
        },
        "expected_top": "Gym Hero",
        "musically_correct": True,
        "note": "All four signals align — expected near-perfect confidence.",
    },
    {
        "name": "Chill Lofi",
        "prefs": {
            "genre": "lofi", "mood": "chill",
            "target_energy": 0.38, "likes_acoustic": True,
        },
        "expected_top": "Library Rain",
        "musically_correct": True,
        "note": "Genre has 3 catalog entries; top result should feel right.",
    },
    {
        "name": "Deep Intense Rock",
        "prefs": {
            "genre": "rock", "mood": "intense",
            "target_energy": 0.90, "likes_acoustic": False,
        },
        "expected_top": "Storm Runner",
        "musically_correct": True,
        "note": "Only one rock song — genre lock-in is expected here.",
    },
    {
        "name": "EDGE: High Energy + Sad Mood",
        "prefs": {
            "genre": "soul", "mood": "sad",
            "target_energy": 0.90, "likes_acoustic": False,
        },
        "expected_top": "Blue Porch",
        "musically_correct": False,
        "note": (
            "Known flaw: genre+mood (+3.0) overwhelms energy mismatch. "
            "Blue Porch (energy=0.29) wins despite being a quiet ballad "
            "when the user wants high-energy sad music."
        ),
    },
    {
        "name": "EDGE: Unknown Genre (bossa nova)",
        "prefs": {
            "genre": "bossa nova", "mood": "relaxed",
            "target_energy": 0.40, "likes_acoustic": True,
        },
        "expected_top": "Dust Road Home",
        "musically_correct": False,
        "note": (
            "Genre not in catalog — genre signal never fires. "
            "Score ceiling drops to ~2.5/4.5; DEGRADED WARNING should appear in logs."
        ),
    },
    {
        "name": "EDGE: Max Acoustic / Perfect Energy",
        "prefs": {
            "genre": "classical", "mood": "melancholic",
            "target_energy": 0.21, "likes_acoustic": True,
        },
        "expected_top": "Rainy Sunday",
        "musically_correct": True,
        "note": "Perfect energy proximity (0.21 == 0.21); all signals align.",
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> bool:
    catalog_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(catalog_path)

    consistency_pass = 0
    quality_pass = 0
    quality_applicable = 0
    confidences = []

    W = 62
    print("\n" + "=" * W)
    print("  Music Recommender — Reliability Evaluation")
    print("=" * W)

    for tc in TEST_CASES:
        results = recommend_songs(tc["prefs"], songs, k=1)
        top_song, top_score, _ = results[0]
        conf = confidence_score(top_score)
        band = confidence_band(top_score)
        confidences.append(conf)

        title_match = top_song["title"] == tc["expected_top"]
        if title_match:
            consistency_pass += 1

        quality_applicable += 1
        if tc["musically_correct"] and title_match:
            quality_pass += 1

        # Build status label
        if not title_match:
            status = "FAIL — unexpected result"
        elif not tc["musically_correct"]:
            status = "PASS (known flaw)"
        else:
            status = "PASS"

        print(f"\n  {status}  ·  {tc['name']}")
        print(f"    Expected : {tc['expected_top']!r}")
        got_label = f"{top_song['title']!r}"
        if not title_match:
            got_label += f"  ← expected {tc['expected_top']!r}"
        print(f"    Got      : {got_label}")
        print(f"    Score    : {top_score:.2f} / 4.50  |  confidence {conf:.2f} ({band})")
        print(f"    Note     : {tc['note']}")

    # Summary stats
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)

    print("\n" + "=" * W)
    print("  SUMMARY")
    print("=" * W)
    print(f"  Consistency  : {consistency_pass}/{len(TEST_CASES)} expected outputs matched")
    print(f"  Quality      : {quality_pass}/{quality_applicable} results musically correct")
    print(f"               ({len(TEST_CASES) - quality_pass} known adversarial cases expose design limits)")
    print(f"  Confidence   : avg={avg_conf:.2f}  min={min_conf:.2f}  max={max_conf:.2f}")
    print()

    # One-line summary (copy this into README)
    print("  One-line summary:")
    print(
        f"  {consistency_pass}/{len(TEST_CASES)} consistency checks passed; "
        f"{quality_pass}/{quality_applicable} results musically correct; "
        f"average confidence {avg_conf:.2f} "
        f"(drops to {min_conf:.2f} when genre is missing from catalog)."
    )
    print("=" * W + "\n")

    return consistency_pass == len(TEST_CASES)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
