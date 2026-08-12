"""
Tests for load_and_sample(): correct count, no duplicates (no-replacement
requirement), and correct handling of missing tweet text.
"""
import sys
import os
import pandas as pd
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_tweets import load_and_sample


def _make_test_csv(rows):
    """Writes a small temporary CSV matching the real dataset's schema."""
    df = pd.DataFrame(rows, columns=["Tweet", "Suicide"])
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    return tmp.name


def test_sample_returns_exact_requested_count():
    rows = [(f"tweet {i}", "Not Suicide post") for i in range(200)]
    path = _make_test_csv(rows)

    sample = load_and_sample(csv_path=path, sample_size=50)

    assert len(sample) == 50


def test_sample_has_no_duplicate_rows():
    """Confirms sampling without replacement - no tweet appears twice."""
    rows = [(f"unique tweet {i}", "Not Suicide post") for i in range(200)]
    path = _make_test_csv(rows)

    sample = load_and_sample(csv_path=path, sample_size=100)

    assert sample["Tweet"].nunique() == len(sample)


def test_missing_tweet_text_is_dropped():
    rows = [
        ("a real tweet", "Not Suicide post"),
        (None, "Not Suicide post"),  # missing text - should be dropped
        ("another real tweet", "Not Suicide post"),
    ]
    path = _make_test_csv(rows)

    sample = load_and_sample(csv_path=path, sample_size=2)

    assert sample["Tweet"].isnull().sum() == 0
    assert len(sample) == 2


def test_label_whitespace_is_stripped():
    """
    The real dataset has a known quirk: 'Potential Suicide post ' has a
    trailing space. Confirms this gets cleaned during sampling.
    """
    rows = [("a tweet", "Potential Suicide post ")]  # trailing space, deliberate
    path = _make_test_csv(rows)

    sample = load_and_sample(csv_path=path, sample_size=1)

    assert sample.iloc[0]["Suicide"] == "Potential Suicide post"