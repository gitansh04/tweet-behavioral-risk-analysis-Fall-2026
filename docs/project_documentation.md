# Tweet Behavioral Risk Analysis Pipeline — Project Documentation

## Overview

This project is a real-time content risk classification pipeline built for The Commons XR's AI Engineering take-home assignment. It ingests a randomly sampled batch of tweets, classifies each one for suicide risk indicators using a large language model via Amazon Bedrock, and persists the results to a relational database — all while simulating a real-time inference constraint of no more than 10 tweets processed per second.

The system is designed around three core requirements from the assignment: (1) a strict real-time throughput ceiling that must be enforced correctly rather than assumed, (2) a classification approach that produces reliable, parseable output rather than free-form text, and (3) resilience — the pipeline must survive individual failures (a throttled API call, a bad model response, a failed database write) without crashing the entire run.

## Architecture

The pipeline follows a linear flow, matching the approved architecture design:

```
Source Dataset (CSV)
        │
        ▼  (1) Sample 100 tweets, no replacement
AWS Lambda: risk_classifier
        │
        │◄── Token Bucket Rate Limiter (max 10 req/sec)
        │
        ▼  (2) Invoke at ≤10 tweets/sec
Amazon Bedrock (Claude Haiku-class model)
        │
        ▼  (3) Classify tweet
        │
        ▼  (4) Write result
Amazon RDS (PostgreSQL)
```

Each stage is deliberately decoupled so it can be reasoned about, tested, and fail independently of the others:

- **Sampling** (`sample_tweets.py`) is a one-time, unthrottled operation performed before any inference begins. It loads the source CSV, drops rows with missing tweet text, and selects exactly 100 rows at random without replacement.
- **Rate limiting** (`rate_limiter.py`) is implemented as a token bucket, refilling continuously at a configurable rate rather than using a fixed sleep between calls. This matters because real API latency is variable — a fixed delay either under-throttles or wastes time depending on how long each individual call takes. The token bucket enforces the rate ceiling correctly regardless of call latency.
- **Classification** (`classify.py`) invokes Amazon Bedrock with a system prompt that constrains the model to return exactly one of four category labels (`high risk`, `potentially likely`, `neutral`, `unlikely`), with no additional text. This makes the output directly parseable without fragile text extraction. The function includes retry logic with exponential backoff for transient throttling, and returns a structured error result rather than raising an exception for any other failure — so a single bad tweet never halts the batch.
- **Persistence** (`db.py`) writes each classified tweet to a PostgreSQL table in RDS, with the same fail-soft design: a failed write is logged and reported, not allowed to crash the run.
- **Orchestration** (`main.py` locally, `lambda_function.py` in AWS) ties these together: sample once, then loop through the sample, acquiring a rate-limiter token before each classification call, writing each result, and tracking a running success/failure count.

The `alert_of_risk` field is derived directly from the classification category: `True` for `high risk` or `potentially likely`, `False` for `neutral` or `unlikely`.

## Deployment

The pipeline is implemented in two parallel forms:

- **`src/lambda/`** — the deployed, production version, packaged as an AWS Lambda function (`risk-classifier`) with a bundled `psycopg2-binary` dependency (compiled for Lambda's Linux runtime) and connected to the RDS instance via Lambda's VPC networking integration. This is the actual event-driven service satisfying the assignment's architectural requirement.
- **`src/local_dev/`** — the local development and testing version of the same logic, using `pandas` for sampling. This version was used to iterate quickly during development and to run and validate the automated test suite (see below) without needing a full Lambda deployment cycle for every change.

Both versions were verified against live AWS services: 100 tweets sampled, classified via real Bedrock calls, and written to RDS, with 100 successes and 0 failures in the final verified run (see "Testing" below for the distinction between this integration verification and the unit tests).

## Setup and Running Locally

```bash
cd src/local_dev
python -m venv venv          # if not already created
venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r ../lambda/requirements.txt pandas python-dotenv
```

Create a `.env` file in `src/local_dev/` (not committed — see `.gitignore`) with:

```
DB_HOST=<your RDS endpoint>
DB_NAME=tweetriskdb
DB_USER=postgres
DB_PASSWORD=<your RDS master password>
DB_PORT=5432
```

Configure AWS credentials locally via `aws configure` (requires an IAM user with Bedrock invoke permissions).

Run the full pipeline:

```bash
python main.py
```

## Testing

The project includes two distinct layers of verification, and it's worth being explicit about what each one actually proves:

### Unit tests (`src/local_dev/tests/`)

These mock external dependencies (the Bedrock client, the database connection) to verify internal logic in isolation, and run in seconds with no live AWS access required:

- **`test_rate_limiter.py`** — exercises the real `TokenBucket` class with real timing (no mocking). Directly addresses reviewer feedback expressing concern that the rate limiter might crash under load exceeding its configured rate: `test_burst_exceeding_rate_does_not_crash` fires 50 requests back-to-back with zero delay against a 10/sec limiter and confirms no exception is raised. `test_burst_actually_enforces_the_rate_ceiling` goes further and measures elapsed time, confirming the limiter isn't merely surviving the burst but genuinely throttling it (the 50-call burst takes several seconds, not milliseconds). Additional tests confirm the configurable rate parameter actually changes throttling behavior proportionally, and that token accumulation is correctly capped during idle periods.
- **`test_sample_tweets.py`** — verifies exact sample count, no duplicate rows (confirming true no-replacement sampling), correct handling of missing tweet text, and cleanup of a known data quirk (a trailing space in one of the source label values).
- **`test_classify.py`** — uses a mocked Bedrock client to verify: valid categories are parsed correctly and `alert_of_risk` is derived correctly; a malformed/unexpected model response is caught and reported as a structured error rather than crashing or silently accepting bad data; transient throttling triggers the retry loop and eventually succeeds; persistent throttling fails gracefully after the configured max retries; and non-throttling exceptions fail fast without pointless retries.
- **`test_db.py`** — uses a mocked database connection to verify a successful write commits and returns `True`, and a failed write rolls back and returns `False` without raising.

Run with:

```bash
cd src/local_dev
python -m pytest tests/ -v
```

All 16 tests pass.

### Integration verification (live AWS)

Unit tests with mocked dependencies prove the code's internal logic is correct, but they do not prove the real external services actually work together end-to-end — that requires a genuine integration test. This pipeline has two:

1. A full local run (`main.py`) against live Bedrock and live RDS, processing 100 real tweets with 100 successes and 0 failures, including several transient Bedrock throttling events that were automatically retried and recovered without failing the batch.
2. The same verification repeated inside the actual deployed Lambda function, confirming the packaged, production version behaves identically to the local development version when run against real AWS services.

## Challenge Overcome: Bedrock Quota Exhaustion

The most significant challenge during implementation was not a coding problem but an infrastructure constraint that initially looked like one. Early testing of the classification function consistently failed with a `ThrottlingException` reporting "too many tokens per day," both in the Bedrock Playground and in code, across multiple regions (`us-east-1` and `us-west-2`).

The first working hypothesis was that this was a temporary rate-limiting condition — the kind of transient throttling the retry/backoff logic was already designed to absorb. However, after the retry logic exhausted its maximum attempts and the error persisted consistently across every attempt, regardless of region, I checked the Bedrock Quotas console directly rather than continuing to guess. This revealed the actual root cause: the AWS account's on-demand token and request quotas were set to a hard `0` across every listed model, not a small nonzero default that happened to be exhausted. This was an account-level provisioning issue, not a bug in the implementation.

Rather than let this block progress on the rest of the pipeline, I made a deliberate engineering decision: I added a `USE_MOCK` flag to the classification function that, when enabled, returns a randomly weighted valid classification instead of calling Bedrock, while leaving every other part of the system — sampling, the rate limiter, error handling, and the database write — completely real and unaffected. This let me fully build, test, and deploy the entire pipeline (including a successful live Lambda run processing 100 tweets end-to-end) while the actual quota issue was resolved in parallel, rather than being fully blocked by an external dependency outside my control.

In parallel, I submitted a formal quota increase request through the AWS Service Quotas console, and proactively communicated the blocker to the reviewer with a clear explanation of the root cause and what was and wasn't affected by it, rather than silently working around it or discovering it late. Once the quota increase was approved, switching from mock to live classification required changing a single flag (`USE_MOCK = False`) in both the Lambda and local versions, followed by re-verification against real data — confirming the mock/real seam had been designed cleanly enough that swapping it back was low-risk.

This experience reinforced a principle I tried to apply throughout the project: when a blocker's root cause is genuinely outside the code (an infrastructure quota, in this case), the right response is to diagnose it precisely rather than assume it's a bug, isolate its impact so it doesn't stall unrelated work, and communicate it transparently rather than letting a reviewer discover an unexplained gap on their own.

## Known Limitations and Scope Decisions

- **RDS is configured for public access** with a dedicated security group, rather than fully private VPC-only access. This was a deliberate scope decision to keep Lambda-to-RDS connectivity setup manageable within the assignment's time constraints; a production deployment would keep the database entirely private within a VPC.
- **The Lambda function processes the full 100-tweet batch within a single invocation**, rather than one invocation per tweet. Given the linear, single-batch nature of this assignment, this keeps the architecture simpler while still satisfying the requirement for an event-driven service invoking Bedrock and writing to storage; a system built for sustained, high-volume production traffic would likely decouple these into a queue-driven, one-invocation-per-item pattern.
