"""
Tests for TokenBucket rate limiter.

These specifically address reviewer feedback: "I suspect it might crash if
you try to go over what the rate limiter allows." These tests prove that
firing far more requests than the configured rate does NOT crash the
limiter, and that it correctly enforces the ceiling rather than merely
hoping timing works out.
"""
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rate_limiter import TokenBucket


def test_burst_exceeding_rate_does_not_crash():
    """
    Fires 50 acquire() calls back-to-back with zero delay between them,
    against a bucket configured for only 10/sec. This deliberately exceeds
    what the limiter allows on every single call after the first 10.
    The test passes if this completes without raising any exception.
    """
    bucket = TokenBucket(rate=10)

    for _ in range(50):
        bucket.acquire()  # would raise if the limiter had a bug under load

    assert True  # reaching this line proves no crash occurred


def test_burst_actually_enforces_the_rate_ceiling():
    """
    Proves the limiter doesn't just "not crash" but actually throttles:
    50 calls against a 10/sec bucket should take close to 4+ seconds
    (50 calls, first 10 free from initial capacity, remaining 40 paced
    at 10/sec = ~4 seconds), not near-zero time.
    """
    bucket = TokenBucket(rate=10)
    start = time.monotonic()

    for _ in range(50):
        bucket.acquire()

    elapsed = time.monotonic() - start

    # Loose bounds to avoid flaky failures from system timing jitter,
    # but tight enough to catch a limiter that isn't throttling at all.
    assert elapsed >= 3.5, (
        f"Expected throttling to take at least ~3.5s for 50 calls at 10/sec, "
        f"took {elapsed:.2f}s - rate limiter may not be enforcing the ceiling"
    )
    assert elapsed <= 6.0, (
        f"Took {elapsed:.2f}s, much longer than expected - possible deadlock or bug"
    )


def test_rate_is_configurable():
    """
    Extra-credit requirement: rate must be adjustable. Proves a bucket
    configured for a much slower rate actually behaves proportionally slower.
    """
    fast_bucket = TokenBucket(rate=20)
    slow_bucket = TokenBucket(rate=5)

    start = time.monotonic()
    for _ in range(20):
        fast_bucket.acquire()
    fast_elapsed = time.monotonic() - start

    start = time.monotonic()
    for _ in range(20):
        slow_bucket.acquire()
    slow_elapsed = time.monotonic() - start

    assert slow_elapsed > fast_elapsed, (
        "A bucket configured with a lower rate should take longer to process "
        "the same number of requests than a bucket with a higher rate"
    )


def test_never_exceeds_capacity_even_with_long_idle_period():
    """
    Confirms tokens don't over-accumulate past capacity if acquire() isn't
    called for a while (i.e., the bucket doesn't let you "bank" unlimited
    burst allowance by sitting idle).
    """
    bucket = TokenBucket(rate=10, capacity=10)
    time.sleep(2)  # idle - tokens should refill but cap at capacity=10
    bucket._refill()
    assert bucket.tokens <= bucket.capacity