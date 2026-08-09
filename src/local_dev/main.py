import os
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables - must happen before importing db.py's config reads

from sample_tweets import load_and_sample
from rate_limiter import TokenBucket
from classify import classify_tweet
from db import get_connection, write_result

RATE_PER_SEC = 10  # configurable, per the extra-credit requirement


def run_pipeline():
    print("Sampling tweets...")
    sample = load_and_sample()
    print(f"Sampled {len(sample)} tweets.\n")

    print("Connecting to database...")
    conn = get_connection()
    print("Connected.\n")

    bucket = TokenBucket(rate=RATE_PER_SEC)

    success_count = 0
    error_count = 0

    for i, row in sample.iterrows():
        tweet_text = row["Tweet"]

        # Wait for rate limiter before making the call - enforces 10/sec ceiling
        bucket.acquire()

        result = classify_tweet(tweet_text)

        if result["error"]:
            print(f"[{i+1}/{len(sample)}] ERROR classifying tweet: {result['error']}")
            error_count += 1
            continue

        written = write_result(
            conn,
            tweet_text=tweet_text,
            suicide_likelihood=result["suicide_likelihood"],
            alert_of_risk=result["alert_of_risk"],
        )

        if written:
            success_count += 1
            print(f"[{i+1}/{len(sample)}] {result['suicide_likelihood']} (alert={result['alert_of_risk']})")
        else:
            error_count += 1
            print(f"[{i+1}/{len(sample)}] Classified but DB write failed.")

    conn.close()

    print(f"\nDone. {success_count} succeeded, {error_count} failed.")


if __name__ == "__main__":
    run_pipeline()