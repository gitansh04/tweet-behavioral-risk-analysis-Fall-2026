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
USE_MOCK = False

# Default S3 location for the normal (non-interview) dataset run
DEFAULT_BUCKET = "tweet-risk-interview-files-gsharma"
DEFAULT_KEY = "tweets_dataset.csv"

SYSTEM_PROMPT = """You are a content classification system that assesses tweets for suicide risk indicators. Your task is to classify a single tweet into exactly one of four categories based on the level of suicide risk expressed.

Categories (choose exactly one):
- high risk: The tweet expresses explicit suicidal ideation, a stated plan or intent to end one's life, or direct expressions of wanting to die that are not clearly hyperbolic or metaphorical.
- potentially likely: The tweet expresses significant distress, hopelessness, or indirect suicidal ideation (e.g., feeling like a burden, wanting the pain to stop, references to not being around much longer) without an explicit statement of intent or plan.
- neutral: The tweet expresses general negative emotion, sadness, frustration, or stress that does not indicate suicidal ideation or risk.
- unlikely: The tweet shows no indication of distress or risk, including neutral, positive, humorous, or unrelated content.

Important guidance:
- Common hyperbolic expressions (e.g., "I could just die," "this is killing me," "kill me now" used about mundane frustrations like homework or a bad day) should generally be classified as neutral or unlikely based on context, not as risk indicators, unless combined with genuine distress signals.
- Song lyrics, quotes, or clearly fictional/joking content should be classified based on whether the tweet's own framing suggests genuine personal distress, not the literal words alone.
- If a tweet is ambiguous, choose the more conservative (higher-risk) category only when there is a reasonable indication of genuine distress - do not default to high risk for uncertain cases.
- Base your classification only on the content of the tweet provided. Do not ask for clarification or additional context.

Output format:
Respond with only the category label, exactly as written above (one of: high risk, potentially likely, neutral, unlikely), in lowercase, with no additional text, punctuation, explanation, or formatting."""

_bedrock = boto3.client("bedrock-runtime", region_name=REGION)
_s3 = boto3.client("s3")


# ---------- S3 fetch ----------
def download_csv_from_s3(bucket, key, local_path="/tmp/input.csv"):
    """
    Downloads the target CSV from S3 into Lambda's writable /tmp storage.
    Using S3 (rather than bundling the CSV into the deployment zip) means a
    new input file - like the interviewer's live assessment file - can be
    swapped in without redeploying the function.
    """
    _s3.download_file(bucket, key, local_path)
    return local_path


# ---------- Sampling (no pandas) ----------
def load_and_sample(csv_path, sample_size=SAMPLE_SIZE, sample_all=False):
    """
    Loads tweets from csv_path and returns a sample.

    sample_all=True processes every row in the file instead of randomly
    sampling - used for the interview assessment file, where the goal is
    to classify every provided tweet rather than a random subset.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet = row.get("Tweet")
            if tweet and tweet.strip():
                rows.append(tweet)

    if sample_all:
        return rows

    actual_sample_size = min(sample_size, len(rows))
    return random.sample(rows, actual_sample_size)


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
    """
    event fields (all optional):
      bucket:     S3 bucket containing the input CSV (defaults to DEFAULT_BUCKET)
      key:        S3 object key for the input CSV (defaults to DEFAULT_KEY)
      sample_all: if true, process every row in the file instead of randomly
                  sampling SAMPLE_SIZE rows. Use this for the interview
                  assessment file.
    """
    bucket = event.get("bucket", DEFAULT_BUCKET)
    key = event.get("key", DEFAULT_KEY)
    sample_all = event.get("sample_all", False)

    print(f"Downloading s3://{bucket}/{key} ...")
    csv_path = download_csv_from_s3(bucket, key)

    sample = load_and_sample(csv_path, sample_all=sample_all)
    print(f"Loaded {len(sample)} tweet(s) from {key} (sample_all={sample_all})")

    conn = get_connection()
    bucket_limiter = TokenBucket(rate=RATE_PER_SEC)

    success_count = 0
    error_count = 0
    total = len(sample)

    for i, tweet_text in enumerate(sample, start=1):
        bucket_limiter.acquire()
        result = classify_tweet(tweet_text)

        if result["error"]:
            error_count += 1
            print(f"[{i}/{total}] ERROR: {result['error']}")
            continue

        written = write_result(conn, tweet_text, result["suicide_likelihood"], result["alert_of_risk"])
        if written:
            success_count += 1
            print(f"[{i}/{total}] {result['suicide_likelihood']} (alert={result['alert_of_risk']})")
        else:
            error_count += 1
            print(f"[{i}/{total}] Classified but DB write failed.")

    conn.close()

    print(f"Done. {success_count} succeeded, {error_count} failed.")

    return {
        "statusCode": 200,
        "body": json.dumps({"succeeded": success_count, "failed": error_count})
    }