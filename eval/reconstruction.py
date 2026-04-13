from typing import Any

import torch

from utils.math_utils import l1_norm
from utils.sparsity import actual_sparsity, count_nonzero, count_zero


def reconstruction_error(W: torch.Tensor, U: torch.Tensor, X: torch.Tensor) -> float:
    residual = (U - W) @ X
    return float(0.5 * residual.pow(2).sum().item())


def objective_value(W: torch.Tensor, U: torch.Tensor, X: torch.Tensor, lambda_: float) -> float:
    return reconstruction_error(W=W, U=U, X=X) + float(lambda_) * l1_norm(U)


def summarize_pruning_result(
    method: str,
    W: torch.Tensor,
    U: torch.Tensor,
    X: torch.Tensor,
    lambda_: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if extra:
        summary.update(extra)
    summary.update(
        {
        "method": method,
        "actual_sparsity": actual_sparsity(U),
        "num_total": int(U.numel()),
        "num_nonzero": count_nonzero(U),
        "num_zero": count_zero(U),
        "reconstruction_error": reconstruction_error(W=W, U=U, X=X),
        "l1_norm": l1_norm(U),
        }
    )
    if lambda_ is not None:
        summary["lambda"] = float(lambda_)
        summary["objective"] = objective_value(W=W, U=U, X=X, lambda_=lambda_)
    return summary
