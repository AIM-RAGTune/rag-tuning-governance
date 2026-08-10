from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimerResult:
    seconds: float = 0.0


@contextmanager
def timer() -> Iterator[TimerResult]:
    result = TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start

