import os
import csv
import json
import random
import time
import threading
import boto3

# ---------- Config ----------
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-east-1"
VALID_CATEGORIES = {"high risk", "potentially likely", "neutral", "unlikely"}
SAMPLE_SIZE = 100
RATE_PER_SEC = 10
USE_MOCK = True  # flip to False once Bedrock quota clears

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

_bedrock = boto3.client("bedrock-runtime", region_name=REGION)


# ---------- Sampling (no pandas) ----------
def load_and_sample(csv_path, sample_size=SAMPLE_SIZE):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Tweet") and row["Tweet"].strip():
                rows.append(row["Tweet"])
    return random.sample(rows, min(sample_size, len(rows)))


# ---------- Rate limiter ----------
class TokenBucket:
    def __init__(self, rate=10, capacity=None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        added = elapsed * self.rate
        if added > 0:
            self.tokens = min(self.capacity, self.tokens + added)
            self.last_refill = now

    def acquire(self):
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                deficit = 1 - self.tokens
                wait_time = deficit / self.rate
            time.sleep(wait_time)


# ---------- Classification ----------
def _mock_classify(tweet_text):
    category = random.choices(
        population=["unlikely", "neutral", "potentially likely", "high risk"],
        weights=[50, 30, 15, 5],
        k=1,
    )[0]
    alert = category in ("high risk", "potentially likely")
    time.sleep(0.05)
    return {"suicide_likelihood": category, "alert_of_risk": alert, "error": None}


def classify_tweet(tweet_text, max_retries=5):
    if USE_MOCK:
        return _mock_classify(tweet_text)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 20,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Classify this tweet: {tweet_text}"}],
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = _bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
            response_body = json.loads(response["body"].read())
            raw_output = response_body["content"][0]["text"].strip().lower()

            if raw_output in VALID_CATEGORIES:
                alert = raw_output in ("high risk", "potentially likely")
                return {"suicide_likelihood": raw_output, "alert_of_risk": alert, "error": None}
            else:
                return {"suicide_likelihood": None, "alert_of_risk": None,
                         "error": f"Unrecognized model output: '{raw_output}'"}

        except _bedrock.exceptions.ThrottlingException as e:
            last_error = str(e)
            delay = 1.0 * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
        except Exception as e:
            last_error = str(e)
            break

    return {"suicide_likelihood": None, "alert_of_risk": None,
             "error": f"Failed after retries: {last_error}"}


# ---------- DB ----------
def get_connection():
    import psycopg2
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        dbname=os.environ.get("DB_NAME", "tweetriskdb"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
        port=os.environ.get("DB_PORT", "5432"),
    )


def write_result(conn, tweet_text, suicide_likelihood, alert_of_risk):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO tweet_risk_results (tweet_text, suicide_likelihood, alert_of_risk)
                   VALUES (%s, %s, %s)""",
                (tweet_text, suicide_likelihood, alert_of_risk),
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"DB write failed: {e}")
        conn.rollback()
        return False


# ---------- Lambda entry point ----------
def lambda_handler(event, context):
    csv_path = "tweets_dataset.csv"  # see packaging note below
    sample = load_and_sample(csv_path)

    conn = get_connection()
    bucket = TokenBucket(rate=RATE_PER_SEC)

    success_count = 0
    error_count = 0

    for tweet_text in sample:
        bucket.acquire()
        result = classify_tweet(tweet_text)

        if result["error"]:
            error_count += 1
            continue

        written = write_result(conn, tweet_text, result["suicide_likelihood"], result["alert_of_risk"])
        if written:
            success_count += 1
        else:
            error_count += 1

    conn.close()

    return {
        "statusCode": 200,
        "body": json.dumps({"succeeded": success_count, "failed": error_count})
    }
