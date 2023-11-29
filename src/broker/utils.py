import time
import random

def exponential_backoff(retry_count, max_delay=60):
    """
    Calculate the delay for exponential backoff.
    :param retry_count: Number of retries attempted.
    :param max_delay: Maximum delay in seconds.
    :return: Delay time in seconds.
    """
    delay = min(max_delay, (2 ** retry_count) + random.uniform(0, 1))  # Exponential backoff + jitter
    time.sleep(delay)
    return delay
