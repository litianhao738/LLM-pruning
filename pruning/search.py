import math
from dataclasses import dataclass
from typing import Any

import torch
from tqdm.auto import tqdm

from pruning.base import BasePruner, PruneResult


@dataclass
class LambdaSearchResult:
    target_sparsity: float
    best_lambda: float
    best_result: PruneResult
    best_actual_sparsity: float
    best_gap: float
    bracket_low: float
    bracket_high: float
    bracket_found: bool
    terminated_reason: str
    trials: list[dict[str, Any]]


def _actual_sparsity(result: PruneResult) -> float:
    return float(result.stats.get("actual_sparsity", 0.0))


def _trial_row(
    *,
    lambda_value: float,
    target_sparsity: float,
    phase: str,
    step: int,
    result: PruneResult,
) -> dict[str, Any]:
    actual = _actual_sparsity(result)
    return {
        "phase": phase,
        "step": step,
        "lambda": float(lambda_value),
        "target_sparsity": float(target_sparsity),
        "actual_sparsity": actual,
        "sparsity_gap": abs(actual - float(target_sparsity)),
        "reconstruction_error": float(result.stats.get("reconstruction_error", 0.0)),
        "objective": float(result.stats.get("objective", 0.0)),
        "num_iters": int(result.stats.get("num_iters", 0)),
    }


def find_lambda_for_target_sparsity(
    *,
    pruner_cls: type[BasePruner],
    W: torch.Tensor,
    X: torch.Tensor,
    target_sparsity: float,
    num_iters: int,
    search_steps: int = 12,
    sparsity_tol: float = 0.02,
    lambda_low: float = 1e-6,
    lambda_high: float = 1.0,
    bracket_scale: float = 10.0,
    max_bracket_steps: int = 12,
    pruner_kwargs: dict[str, Any] | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> LambdaSearchResult:
    if not (0.0 <= target_sparsity <= 1.0):
        raise ValueError(f"target_sparsity must be in [0, 1], got {target_sparsity}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    if search_steps <= 0:
        raise ValueError(f"search_steps must be positive, got {search_steps}")
    if sparsity_tol < 0:
        raise ValueError(f"sparsity_tol must be non-negative, got {sparsity_tol}")
    if lambda_low <= 0 or lambda_high <= 0:
        raise ValueError("lambda_low and lambda_high must be positive")
    if bracket_scale <= 1.0:
        raise ValueError(f"bracket_scale must be > 1.0, got {bracket_scale}")
    if max_bracket_steps <= 0:
        raise ValueError(f"max_bracket_steps must be positive, got {max_bracket_steps}")

    pruner_kwargs = dict(pruner_kwargs or {})
    trial_cache: dict[float, PruneResult] = {}
    trials: list[dict[str, Any]] = []
    best_lambda: float | None = None
    best_result: PruneResult | None = None
    best_actual = 0.0
    best_gap = float("inf")
    max_evaluations = 2 + max_bracket_steps + search_steps
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=max_evaluations,
            desc=progress_desc or f"lambda search @ target={target_sparsity:.3f}",
            leave=False,
        )

    def evaluate(lambda_value: float, phase: str, step: int) -> PruneResult:
        nonlocal best_lambda, best_result, best_actual, best_gap

        lambda_value = float(max(lambda_value, 1e-12))
        if lambda_value not in trial_cache:
            pruner = pruner_cls(lambda_=lambda_value, num_iters=num_iters, **pruner_kwargs)
            result = pruner.prune(W=W, X=X)
            trial_cache[lambda_value] = result
            if progress_bar is not None:
                progress_bar.update(1)

            row = _trial_row(
                lambda_value=lambda_value,
                target_sparsity=target_sparsity,
                phase=phase,
                step=step,
                result=result,
            )
            trials.append(row)

            gap = float(row["sparsity_gap"])
            actual = float(row["actual_sparsity"])
            if best_result is None or gap < best_gap or (
                math.isclose(gap, best_gap, rel_tol=1e-9, abs_tol=1e-12)
                and float(row["reconstruction_error"])
                < float(best_result.stats.get("reconstruction_error", float("inf")))
            ):
                best_lambda = lambda_value
                best_result = result
                best_actual = actual
                best_gap = gap

        return trial_cache[lambda_value]

    low = float(min(lambda_low, lambda_high))
    high = float(max(lambda_low, lambda_high))

    low_result = evaluate(low, phase="init", step=0)
    high_result = evaluate(high, phase="init", step=1)
    bracket_found = False
    terminated_reason = "search_budget_exhausted"

    for step in range(max_bracket_steps):
        low_actual = _actual_sparsity(low_result)
        high_actual = _actual_sparsity(high_result)

        if low_actual <= target_sparsity <= high_actual:
            bracket_found = True
            break

        if low_actual > target_sparsity:
            new_low = max(low / bracket_scale, 1e-12)
            if math.isclose(new_low, low, rel_tol=1e-12, abs_tol=1e-12):
                terminated_reason = "lower_lambda_limit_reached"
                break
            high = low
            high_result = low_result
            low = new_low
            low_result = evaluate(low, phase="expand_low", step=step)
            continue

        if high_actual < target_sparsity:
            low = high
            low_result = high_result
            high = high * bracket_scale
            high_result = evaluate(high, phase="expand_high", step=step)
            continue

    if not bracket_found:
        low_actual = _actual_sparsity(low_result)
        high_actual = _actual_sparsity(high_result)
        bracket_found = low_actual <= target_sparsity <= high_actual

    if bracket_found:
        terminated_reason = "search_budget_exhausted"
        for step in range(search_steps):
            if best_gap <= sparsity_tol:
                terminated_reason = "within_tolerance"
                break
            if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-12):
                terminated_reason = "lambda_interval_collapsed"
                break

            mid = math.sqrt(low * high)
            if math.isclose(mid, low, rel_tol=1e-9, abs_tol=1e-12) or math.isclose(
                mid,
                high,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                terminated_reason = "lambda_interval_collapsed"
                break

            mid_result = evaluate(mid, phase="binary_search", step=step)
            mid_actual = _actual_sparsity(mid_result)

            if mid_actual >= target_sparsity:
                high = mid
                high_result = mid_result
            else:
                low = mid
                low_result = mid_result
    else:
        terminated_reason = "target_not_bracketed"

    if best_result is None or best_lambda is None:
        if progress_bar is not None:
            progress_bar.close()
        raise RuntimeError("Lambda search failed to evaluate any candidate")

    if progress_bar is not None:
        progress_bar.close()

    return LambdaSearchResult(
        target_sparsity=float(target_sparsity),
        best_lambda=float(best_lambda),
        best_result=best_result,
        best_actual_sparsity=float(best_actual),
        best_gap=float(best_gap),
        bracket_low=float(low),
        bracket_high=float(high),
        bracket_found=bool(bracket_found),
        terminated_reason=terminated_reason,
        trials=trials,
    )
