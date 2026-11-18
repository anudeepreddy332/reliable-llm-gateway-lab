from collections.abc import Callable
from dataclasses import dataclass

@dataclass(frozen=True)
class ProviderAttempt:
    name: str
    execute: Callable[[], str]

@dataclass(frozen=True)
class ExecutionResult:
    content: str
    provider: str
    attempts: int

def execute_plan(
    plan: tuple[ProviderAttempt, ...],
    *,
    can_failover: Callable[[Exception], bool],
    max_attempts: int = 2,
) -> ExecutionResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    if not plan or len(plan) > max_attempts:
        raise ValueError("Invalid provider attempt plan")

    for attempt_number, attempt in enumerate(plan, start=1):
        try:
            content = attempt.execute()

        except Exception as exc:
            is_final_attempt = attempt_number == len(plan)

            if is_final_attempt or not can_failover(exc):
                raise

        else:
            return ExecutionResult(
                content=content,
                provider=attempt.name,
                attempts=attempt_number,
            )

    raise RuntimeError("Unexpected execution plan state")