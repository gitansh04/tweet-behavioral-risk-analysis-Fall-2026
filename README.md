# Tweet Behavioral Risk Analysis Pipeline

Real-time content risk classification pipeline built for The Commons XR's AI Engineering take-home assignment. Samples tweets, classifies each for suicide risk indicators using Amazon Bedrock, and writes the results to PostgreSQL — while enforcing a strict real-time throughput ceiling of 10 tweets/second.

Full architecture, design rationale, and challenges encountered are documented in [`docs/architecture-design.md`](docs/architecture-design.md) and [`docs/project_documentation.md`](docs/project_documentation.md).

## How It Works

1. **Sample** — 100 tweets are pulled from the source dataset at random, without replacement (or, for the interview assessment file, every provided tweet is processed).
2. **Rate limit** — a token-bucket limiter caps outbound classification calls at a configurable rate (default 10/sec).
3. **Classify** — each tweet is sent to Amazon Bedrock (Claude Haiku) with a system prompt constrained to return one of four categories: `high risk`, `potentially likely`, `neutral`, `unlikely`. Transient throttling is retried with exponential backoff; other failures are logged and skipped rather than crashing the batch.
4. **Persist** — each result, including a derived `alert_of_risk` flag, is written to a PostgreSQL table in Amazon RDS.

The pipeline exists in two forms:

|                 | Path             | Purpose                                                                                                                                                      |
| --------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Production**  | `src/lambda/`    | Deployed AWS Lambda function. Reads its input CSV from S3 (bucket/key passed via the invocation event), so a new file can be swapped in without redeploying. |
| **Development** | `src/local_dev/` | Local version used for fast iteration and running the automated test suite.                                                                                  |

## Project Structure

```
├── data/                    # Source dataset (gitignored)
├── docs/                    # Architecture design + project documentation
├── src/
│   ├── lambda/               # Deployed Lambda function + requirements.txt
│   └── local_dev/            # Local dev scripts + tests/
```

## Prerequisites

- Python 3.11+
- An AWS account with Bedrock access enabled and a PostgreSQL RDS instance
- AWS credentials configured locally (`aws configure`)

## Setup (Local Development)

```bash
cd src/local_dev
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # macOS/Linux

pip install -r ../lambda/requirements.txt pandas python-dotenv pytest
```

Create a `.env` file in `src/local_dev/` (never committed — see `.gitignore`):

```
DB_HOST=<your RDS endpoint>
DB_NAME=tweetriskdb
DB_USER=postgres
DB_PASSWORD=<your RDS master password>
DB_PORT=5432
```

## Running the Pipeline

```bash
cd src/local_dev
python main.py
```

Processes 100 randomly sampled tweets end-to-end and prints per-tweet results plus a final success/failure summary.

## Running Tests

```bash
cd src/local_dev
python -m pytest tests/ -v
```

16 unit tests covering the rate limiter, classification error handling and retry logic, sampling correctness, and database write resilience. External dependencies (Bedrock, RDS) are mocked, so these run in seconds without needing live AWS access.

This is deliberately distinct from **integration verification**: the full pipeline has also been run successfully against live AWS services — real Bedrock model calls and real RDS writes — both locally and in the deployed Lambda function, with 100/100 tweets processed successfully and the rate limiter correctly recovering from transient throttling. Unit tests prove the internal logic is correct in isolation; the live runs prove the real external services actually work together end-to-end.

## Lambda Invocation

The deployed function reads its input file from S3. Test event shape:

```json
{
  "bucket": "your-bucket-name",
  "key": "tweets_dataset.csv",
  "sample_all": false
}
```

- `bucket` / `key` — defaults to the standard dataset if omitted.
- `sample_all` — when `true`, processes every row in the file instead of sampling 100 (used for the interview assessment file).

## Known Limitations

- RDS is configured for public access rather than fully private VPC-only access — a deliberate scope decision given the assignment's time constraints.
- The Lambda function processes a full batch in a single invocation rather than one invocation per tweet, keeping the architecture simple for this assignment's linear, single-batch use case.
