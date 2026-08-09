from dotenv import load_dotenv
load_dotenv()

from db import get_connection

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tweet_risk_results (
    tweet_id SERIAL PRIMARY KEY,
    tweet_text TEXT NOT NULL,
    suicide_likelihood VARCHAR(50) NOT NULL,
    alert_of_risk BOOLEAN NOT NULL,
    processed_at TIMESTAMP DEFAULT NOW()
);
"""

if __name__ == "__main__":
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    conn.close()
    print("Table created (or already existed).")