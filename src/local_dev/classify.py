import boto3
import json
import time
import random as pyrandom
import os

USE_MOCK = False

# Fill in the exact model ID you noted from the Bedrock model catalog page earlier
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # <-- confirm this matches what you saw on the model detail page

REGION = "us-east-1"

VALID_CATEGORIES = {"high risk", "potentially likely", "neutral", "unlikely"}

SYSTEM_PROMPT = """You are a content classification system that assesses tweets for suicide risk indicators. Your task is to classify a single tweet into exactly one of four categories based on the level of suicide risk expressed.

Categories (choose exactly one):
- high risk: The tweet expresses explicit suicidal ideation, a stated plan or intent to end one's life, or direct expressions of wanting to die that are not clearly hyperbolic or metaphorical.
- potentially likely: The tweet expresses significant distress, hopelessness, or indirect suicidal ideation (e.g., feeling like a burden, wanting the pain to stop, references to not being around much longer) without an explicit statement of intent or plan.
- neutral: The tweet expresses general negative emotion, sadness, frustration, or stress that does not indicate suicidal ideation or risk.
- unlikely: The tweet shows no indication of distress or risk, including neutral, positive, humorous, or unrelated content.

Important guidance:
- Common hyperbolic expressions (e.g., "I could just die," "this is killing me," "kill me now" used about mundane frustrations like homework or a bad day) should generally be classified as neutral or unlikely based on context, not as risk indicators, unless combined with genuine distress signals.
- Song lyrics, quotes, or clearly fictional/joking content should be classified based on whether the tweet's own framing suggests genuine personal distress, not the literal words alone.
- If a tweet is ambiguous, choose the more conservative (higher-risk) category only when there is a reasonable indication of genuine distress — do not default to high risk for uncertain cases.
- Base your classification only on the content of the tweet provided. Do not ask for clarification or additional context.

Output format:
Respond with only the category label, exactly as written above (one of: high risk, potentially likely, neutral, unlikely), in lowercase, with no additional text, punctuation, explanation, or formatting."""

_client = boto3.client("bedrock-runtime", region_name=REGION)

MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds

def _mock_classify(tweet_text):
    """
    Stand-in for the real Bedrock call while quota is throttled.
    Returns a randomly chosen valid category so the rest of the pipeline
    (rate limiter, RDS writes, full 100-tweet loop) can be tested end-to-end.
    Weighted toward 'unlikely'/'neutral' since that reflects the real dataset's
    actual class balance (most tweets aren't risk-indicating).
    """
    category = pyrandom.choices(
        population=["unlikely", "neutral", "potentially likely", "high risk"],
        weights=[50, 30, 15, 5],
        k=1,
    )[0]
    alert = category in ("high risk", "potentially likely")
    # Simulate realistic latency so your rate limiter's timing logic gets tested properly
    time.sleep(0.05)
    return {
        "suicide_likelihood": category,
        "alert_of_risk": alert,
        "error": None,
    }


def classify_tweet(tweet_text, max_retries=MAX_RETRIES):
    if USE_MOCK:
        return _mock_classify(tweet_text)
    """
    Calls Bedrock to classify a single tweet's suicide risk level.
    Returns a dict: {"suicide_likelihood": <category>, "alert_of_risk": <bool>, "error": <str or None>}
    Retries on throttling with exponential backoff. Never raises - always returns a result dict,
    so a single failed tweet doesn't crash the whole batch.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 20,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"Classify this tweet: {tweet_text}"}
        ],
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = _client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(body),
            )
            response_body = json.loads(response["body"].read())
            raw_output = response_body["content"][0]["text"].strip().lower()

            if raw_output in VALID_CATEGORIES:
                category = raw_output
            else:
                # Model didn't follow the format - log it but don't crash
                return {
                    "suicide_likelihood": None,
                    "alert_of_risk": None,
                    "error": f"Unrecognized model output: '{raw_output}'",
                }

            alert = category in ("high risk", "potentially likely")
            return {
                "suicide_likelihood": category,
                "alert_of_risk": alert,
                "error": None,
            }

        except _client.exceptions.ThrottlingException as e:
            last_error = str(e)
            # Exponential backoff with jitter: wait longer each retry, plus randomness
            # so multiple retries don't all collide again at the exact same moment
            delay = BASE_DELAY * (2 ** attempt) + pyrandom.uniform(0, 0.5)
            print(f"Throttled. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)

        except Exception as e:
            # Any other error (malformed response, network issue, etc.) - don't retry, just report it
            last_error = str(e)
            break

    return {
        "suicide_likelihood": None,
        "alert_of_risk": None,
        "error": f"Failed after retries: {last_error}",
    }


if __name__ == "__main__":
    # Quick manual test with 2 sample tweets - minimal token usage
    test_tweets = [
        "making some lunch",
        "I hate my life lmao I hope I die soon or summ I'm too tired of everything",
    ]

    for t in test_tweets:
        result = classify_tweet(t)
        print(f"Tweet: {t}")
        print(f"Result: {result}\n")