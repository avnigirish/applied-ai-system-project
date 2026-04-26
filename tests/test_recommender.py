"""
Tests for recommender.py (OOP interface) and ai_layer.py.

Run with:  pytest
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from src.recommender import Song, UserProfile, Recommender, score_song, recommend_songs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_songs() -> list[Song]:
    return [
        Song(id=1, title="Test Pop Track", artist="A",
             genre="pop", mood="happy", energy=0.8,
             tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=2, title="Chill Lofi Loop", artist="B",
             genre="lofi", mood="chill", energy=0.4,
             tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9),
    ]


def _make_recommender() -> Recommender:
    return Recommender(_make_songs())


def _pop_user() -> UserProfile:
    return UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )


# ---------------------------------------------------------------------------
# Recommender OOP tests
# ---------------------------------------------------------------------------

def test_recommend_returns_correct_count():
    rec = _make_recommender()
    results = rec.recommend(_pop_user(), k=2)
    assert len(results) == 2


def test_recommend_returns_song_objects():
    rec = _make_recommender()
    results = rec.recommend(_pop_user(), k=1)
    assert isinstance(results[0], Song)


def test_recommend_top_result_matches_genre_and_mood():
    rec = _make_recommender()
    results = rec.recommend(_pop_user(), k=2)
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_k_larger_than_catalog():
    rec = _make_recommender()
    results = rec.recommend(_pop_user(), k=100)
    assert len(results) == 2  # only 2 songs exist


def test_explain_recommendation_is_non_empty_string():
    rec = _make_recommender()
    explanation = rec.explain_recommendation(_pop_user(), rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_explain_recommendation_mentions_genre():
    rec = _make_recommender()
    explanation = rec.explain_recommendation(_pop_user(), rec.songs[0])
    assert "genre" in explanation.lower()


# ---------------------------------------------------------------------------
# Functional API tests (score_song / recommend_songs)
# ---------------------------------------------------------------------------

def _song_dict(genre="pop", mood="happy", energy=0.8, acousticness=0.2) -> dict:
    return {
        "id": 1, "title": "T", "artist": "A",
        "genre": genre, "mood": mood,
        "energy": energy, "tempo_bpm": 120,
        "valence": 0.9, "danceability": 0.8,
        "acousticness": acousticness,
    }


def _prefs(genre="pop", mood="happy", energy=0.8, acoustic=False) -> dict:
    return {"genre": genre, "mood": mood,
            "target_energy": energy, "likes_acoustic": acoustic}


def test_score_song_perfect_match():
    score, _ = score_song(_prefs(), _song_dict())
    assert score == pytest.approx(4.5, abs=0.01)


def test_score_song_no_match():
    score, _ = score_song(_prefs(genre="rock", mood="sad"),
                          _song_dict(genre="lofi", mood="chill"))
    assert score < 2.0  # no genre or mood points


def test_score_song_genre_contributes_two_points():
    score_match, _ = score_song(_prefs(), _song_dict())
    score_no_match, _ = score_song(_prefs(genre="rock"), _song_dict())
    assert score_match - score_no_match == pytest.approx(2.0, abs=0.01)


def test_score_song_mood_contributes_one_point():
    score_match, _ = score_song(_prefs(), _song_dict())
    score_no_match, _ = score_song(_prefs(mood="sad"), _song_dict())
    assert score_match - score_no_match == pytest.approx(1.0, abs=0.01)


def test_recommend_songs_sorted_descending():
    songs = [_song_dict("lofi", "chill", 0.4), _song_dict("pop", "happy", 0.8)]
    results = recommend_songs(_prefs(), songs, k=2)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_songs_respects_k():
    songs = [_song_dict() for _ in range(10)]
    results = recommend_songs(_prefs(), songs, k=3)
    assert len(results) == 3


def test_recommend_songs_empty_catalog():
    results = recommend_songs(_prefs(), [], k=5)
    assert results == []


# ---------------------------------------------------------------------------
# AI layer tests (keyword parser — no API key needed)
# ---------------------------------------------------------------------------

from src.ai_layer import parse_user_query, generate_ai_explanation, _keyword_parse


def test_keyword_parse_gym_query():
    prefs = _keyword_parse("I want something to pump me up at the gym")
    assert prefs["target_energy"] >= 0.7
    assert prefs["mood"] == "intense"


def test_keyword_parse_study_query():
    prefs = _keyword_parse("calm lofi music for studying")
    assert prefs["genre"] == "lofi"
    assert prefs["target_energy"] <= 0.5


def test_keyword_parse_acoustic_query():
    prefs = _keyword_parse("acoustic folk songs around a campfire")
    assert prefs["likes_acoustic"] is True


def test_keyword_parse_returns_all_required_keys():
    prefs = _keyword_parse("happy dance music")
    for key in ("genre", "mood", "target_energy", "likes_acoustic"):
        assert key in prefs


def test_keyword_parse_energy_in_valid_range():
    prefs = _keyword_parse("something nice to listen to")
    assert 0.0 <= prefs["target_energy"] <= 1.0


@patch("src.ai_layer._get_client", return_value=None)
def test_parse_user_query_fallback_without_key(_mock):
    prefs = parse_user_query("chill lofi music for studying")
    assert prefs["genre"] == "lofi"
    assert "target_energy" in prefs


@patch("src.ai_layer._get_client", return_value=None)
def test_generate_ai_explanation_fallback_without_key(_mock):
    song = _song_dict()
    explanation = generate_ai_explanation(
        _prefs(), song, 4.5, ["genre match (+2.0)", "mood match (+1.0)"]
    )
    assert "genre" in explanation


def test_parse_user_query_raises_on_empty_input():
    with pytest.raises(ValueError):
        parse_user_query("   ")


# ---------------------------------------------------------------------------
# Claude API tests (mocked — no real network call)
# ---------------------------------------------------------------------------

def _make_tool_use_block(prefs: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "set_music_preferences"
    block.input = prefs
    return block


@patch("src.ai_layer._get_client")
def test_parse_user_query_calls_claude(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    expected_prefs = {
        "genre": "pop", "mood": "happy",
        "target_energy": 0.8, "likes_acoustic": False,
    }
    mock_response = MagicMock()
    mock_response.content = [_make_tool_use_block(expected_prefs)]
    mock_client.messages.create.return_value = mock_response

    result = parse_user_query("upbeat pop for a party")
    assert result["genre"] == "pop"
    assert result["mood"] == "happy"
    mock_client.messages.create.assert_called_once()


@patch("src.ai_layer._get_client")
def test_generate_ai_explanation_calls_claude(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Great energy match for your workout.")]
    mock_client.messages.create.return_value = mock_response

    song = _song_dict()
    result = generate_ai_explanation(_prefs(), song, 4.2, ["genre match (+2.0)"])
    assert "workout" in result
    mock_client.messages.create.assert_called_once()


@patch("src.ai_layer._get_client")
def test_generate_ai_explanation_falls_back_on_exception(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.messages.create.side_effect = RuntimeError("network error")

    song = _song_dict()
    reasons = ["genre match (+2.0)", "mood match (+1.0)"]
    result = generate_ai_explanation(_prefs(), song, 3.0, reasons)
    assert result == ", ".join(reasons)
