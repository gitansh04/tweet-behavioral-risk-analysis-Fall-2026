"""
Tests for write_result(): confirms a failed DB write returns False and
does not raise, which is what allows main.py's loop to skip a bad write
and continue processing the rest of the batch instead of crashing.
"""
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db import write_result


def test_successful_write_returns_true_and_commits():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    result = write_result(mock_conn, "a tweet", "neutral", False)

    assert result is True
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()


def test_failed_write_returns_false_not_raises_and_rolls_back():
    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = Exception("connection lost")

    result = write_result(mock_conn, "a tweet", "neutral", False)

    assert result is False
    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()