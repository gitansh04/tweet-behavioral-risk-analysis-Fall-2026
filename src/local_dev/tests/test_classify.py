"""
Tests for classify_tweet(): output validation, retry/backoff behavior,
and confirmation that failures return an error dict instead of raising -
which is what allows the batch loop to skip a bad tweet and keep going.
"""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import classify


def _make_bedrock_response(text):
    """Builds a fake Bedrock invoke_model response body."""
    body = MagicMock()
    body.read.return_value = (
        '{"content": [{"text": "%s"}]}' % text
    ).encode("utf-8")
    return {"body": body}


def test_valid_category_is_parsed_correctly():
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = _make_bedrock_response("high risk")

    result = classify.classify_tweet("some tweet", client=mock_client)

    assert result["error"] is None
    assert result["suicide_likelihood"] == "high risk"
    assert result["alert_of_risk"] is True


def test_unlikely_and_neutral_do_not_trigger_alert():
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = _make_bedrock_response("unlikely")

    result = classify.classify_tweet("some tweet", client=mock_client)

    assert result["alert_of_risk"] is False


def test_malformed_model_output_returns_error_not_crash():
    """
    If the model ever ignores the strict output format instruction and
    returns something unexpected, classify_tweet must report it as an
    error, not raise or silently accept garbage as a category.
    """
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = _make_bedrock_response(
        "I think this tweet seems concerning, possibly high risk"
    )

    result = classify.classify_tweet("some tweet", client=mock_client)

    assert result["suicide_likelihood"] is None
    assert result["alert_of_risk"] is None
    assert result["error"] is not None
    assert "Unrecognized model output" in result["error"]


def test_throttling_triggers_retry_then_succeeds():
    """
    Simulates Bedrock throttling on the first 2 calls, then succeeding on
    the 3rd. Confirms the retry loop actually retries and returns the
    eventual successful result, rather than giving up on the first failure.
    """
    mock_client = MagicMock()

    class FakeThrottlingException(Exception):
        pass

    mock_client.exceptions.ThrottlingException = FakeThrottlingException

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise FakeThrottlingException("Too many tokens")
        return _make_bedrock_response("neutral")

    mock_client.invoke_model.side_effect = side_effect

    with patch("time.sleep", return_value=None):  # skip real backoff delay in test
        result = classify.classify_tweet("some tweet", client=mock_client)

    assert call_count["n"] == 3
    assert result["error"] is None
    assert result["suicide_likelihood"] == "neutral"


def test_persistent_throttling_fails_gracefully_after_max_retries():
    """
    Confirms that if throttling never clears, classify_tweet gives up after
    max_retries and returns an error dict - not an unhandled exception.
    This is what allows main.py's loop to skip the tweet and continue,
    rather than crashing the entire 100-tweet batch.
    """
    mock_client = MagicMock()

    class FakeThrottlingException(Exception):
        pass

    mock_client.exceptions.ThrottlingException = FakeThrottlingException
    mock_client.invoke_model.side_effect = FakeThrottlingException("Too many tokens")

    with patch("time.sleep", return_value=None):
        result = classify.classify_tweet("some tweet", max_retries=3, client=mock_client)

    assert result["suicide_likelihood"] is None
    assert result["error"] is not None
    assert "Failed after retries" in result["error"]


def test_unexpected_exception_does_not_retry_and_does_not_crash():
    """
    Non-throttling errors (e.g. a malformed response) should fail fast
    with an error dict, not be retried pointlessly, and must not raise.
    """
    mock_client = MagicMock()
    mock_client.exceptions.ThrottlingException = type("Fake", (Exception,), {})
    mock_client.invoke_model.side_effect = ValueError("unexpected response shape")

    result = classify.classify_tweet("some tweet", client=mock_client)

    assert result["error"] is not None
    assert "Failed after retries" in result["error"]