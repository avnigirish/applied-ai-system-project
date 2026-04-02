"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Taste profile: a focused late-night indie listener
    # - Prefers indie pop with a moody emotional tone
    # - Moderate-to-high energy (alert but not intense)
    # - Not acoustic-leaning — comfortable with produced, electronic textures
    # - valence kept low-mid to allow moody/bittersweet songs through
    user_prefs = {
        "genre":         "indie pop",
        "mood":          "moody",
        "target_energy": 0.72,
        "likes_acoustic": False,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
