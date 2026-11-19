from collections.abc import Callable
from dataclasses import dataclass
import math
import time


class DeadlineExceeded(Exception):
    """The provider execution budget was exhausted."""

@dataclass(frozen=True)
class ProviderAttempt:
    name: str
    execute: Callable[[float], str]

@dataclass(frozen=True)
class ExecutionResult:
    content: str
    provider: str
    attempts: int

def execute_plan(
    plan: tuple[ProviderAttempt, ...],
    *,
    can_continue: Callable[[Exception], bool],
    max_attempts: int = 2,
    total_timeout_seconds: float = 20.0,
    backoff_seconds: float = 0.05,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ExecutionResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    if not plan or len(plan) > max_attempts:
        raise ValueError("Invalid provider attempt plan")

    if (
            not math.isfinite(total_timeout_seconds)
            or total_timeout_seconds <= 0
    ):
        raise ValueError("Invalid total timeout")

    if not math.isfinite(backoff_seconds) or backoff_seconds < 0:
        raise ValueError("Invalid retry backoff")

    deadline = clock() + total_timeout_seconds


    for attempt_number, attempt in enumerate(plan, start=1):
        remaining = deadline - clock()

        if remaining <= 0:
            raise DeadlineExceeded(
                "Deadline exhausted before provider attempt"
            )

        try:
            content = attempt.execute(remaining)

        except Exception as exc:
            if clock() >= deadline:
                raise DeadlineExceeded(
                    "Deadline exhausted during provider attempt"
                ) from exc

            final_attempt = attempt_number == len(plan)

            if final_attempt or not can_continue(exc):
                raise

            remaining = deadline - clock()

            if remaining <= backoff_seconds:
                raise DeadlineExceeded(
                    "Insufficient budget for another attempt"
                ) from exc

            sleep(backoff_seconds)


        else:
            if clock() >= deadline:
                raise DeadlineExceeded(
                    "Provider completed after deadline"
                )

            return ExecutionResult(
                content=content,
                provider=attempt.name,
                attempts=attempt_number,
            )

    raise RuntimeError("Unexpected execution plan state")