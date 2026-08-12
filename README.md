# tweet-behavioral-risk-analysis-Fall-2026

# Unit tests (tests/) mock external dependencies (Bedrock, RDS) to verify internal logic in isolation — error handling, retry behavior, rate limiting, sampling correctness — and run in seconds without needing live AWS access.

# Integration verification: the full pipeline was also run successfully against live AWS services (real Bedrock model calls, real RDS writes) both locally and deployed in Lambda, confirming 100/100 tweets processed successfully with the rate limiter correctly recovering from transient throttling.
