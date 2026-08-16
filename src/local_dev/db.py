import os
import psycopg2

# Pull from environment variables - never hardcode credentials in code
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME", "tweetriskdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT", "5432")


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
    )


def write_result(conn, tweet_text, suicide_likelihood, alert_of_risk):
    """
    Writes a single classification result to the tweet_risk_results table.
    Returns True on success, False on failure (logs the error, doesn't raise -
    so one bad DB write doesn't kill the whole batch).
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tweet_risk_results (tweet_text, suicide_likelihood, alert_of_risk)
                VALUES (%s, %s, %s)
                """,
                (tweet_text, suicide_likelihood, alert_of_risk),
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"DB write failed: {e}")
        conn.rollback()
        return False