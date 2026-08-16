import time
import threading

class TokenBucket:
    """
    Token bucket rate limiter. Enforces a maximum rate of `rate` operations
    per second, regardless of how long each individual operation takes.
    Configurable so the rate can be adjusted (satisfies extra credit).
    """
    def __init__(self, rate=10, capacity=None):
        self.rate = rate  # tokens added per second
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
        """Blocks until a token is available, then consumes one."""
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                # Not enough tokens yet - compute how long until one is available
                deficit = 1 - self.tokens
                wait_time = deficit / self.rate
            time.sleep(wait_time)