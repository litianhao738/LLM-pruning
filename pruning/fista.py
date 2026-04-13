from collections.abc import Callable

import torch

from pruning.base import BasePruner, PruneResult
from utils.math_utils import (
    estimate_lipschitz_from_gram,
    gram_matrix,
    l1_norm,
    nesterov_coefficient,
    soft_threshold,
)
from utils.sparsity import actual_sparsity, count_nonzero, count_zero


ThresholdSchedule = Callable[[float, int, int], float]


def constant_threshold_schedule(lambda0: float, step: int, total_steps: int) -> float:
    del step, total_steps
    return float(lambda0)


def _reconstruction_error(W: torch.Tensor, U: torch.Tensor, X: torch.Tensor) -> float:
    residual = (U - W) @ X
    return float(0.5 * residual.pow(2).sum().item())


def _objective_value(W: torch.Tensor, U: torch.Tensor, X: torch.Tensor, lambda_: float) -> float:
    return _reconstruction_error(W=W, U=U, X=X) + float(lambda_) * l1_norm(U)


class FISTAPruner(BasePruner):
    """
    Baseline optimization pruner for the layer-wise objective:

        min_U 0.5 * ||U X - W X||_F^2 + lambda * ||U||_1

    Notes:
    - The baseline uses a constant threshold schedule, matching standard FISTA.
    - The schedule is configurable so the same class can be reused later for
      the adaptive-threshold innovation in 7503Pre page 8.
    """

    def __init__(
        self,
        lambda_: float,
        num_iters: int = 50,
        lipschitz: float | None = None,
        tolerance: float = 0.0,
        schedule: ThresholdSchedule | None = None,
    ):
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        if num_iters <= 0:
            raise ValueError(f"num_iters must be positive, got {num_iters}")
        if lipschitz is not None and lipschitz <= 0:
            raise ValueError(f"lipschitz must be positive when provided, got {lipschitz}")
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

        self.lambda_ = float(lambda_)
        self.num_iters = int(num_iters)
        self.lipschitz = None if lipschitz is None else float(lipschitz)
        self.tolerance = float(tolerance)
        self.schedule = schedule or constant_threshold_schedule

    def prune(self, W: torch.Tensor, X: torch.Tensor) -> PruneResult:
        if not isinstance(W, torch.Tensor) or not isinstance(X, torch.Tensor):
            raise TypeError("W and X must both be torch.Tensor")
        if W.ndim != 2 or X.ndim != 2:
            raise ValueError("W and X must both be rank-2 tensors")
        if W.shape[1] != X.shape[0]:
            raise ValueError(
                f"Incompatible shapes: W is {tuple(W.shape)} but X is {tuple(X.shape)}"
            )
        if W.numel() == 0 or X.numel() == 0:
            raise ValueError("W and X must be non-empty")

        G = gram_matrix(X)
        WG = W @ G
        L = self.lipschitz or estimate_lipschitz_from_gram(G)
        if L <= 0:
            raise ValueError("Estimated Lipschitz constant must be positive")

        U_k = W.clone()
        Z_k = U_k.clone()
        t_k = 1.0
        history: list[dict[str, float]] = []

        for step in range(self.num_iters):
            lambda_k = self.schedule(self.lambda_, step, self.num_iters)
            grad = Z_k @ G - WG
            V_k = Z_k - grad / L
            U_next = soft_threshold(V_k, threshold=lambda_k / L)
            diff_norm = torch.linalg.norm(U_next - U_k).item()

            current_objective = _objective_value(W=W, U=U_next, X=X, lambda_=lambda_k)
            history.append(
                {
                    "step": float(step),
                    "lambda": float(lambda_k),
                    "objective": float(current_objective),
                    "reconstruction_error": _reconstruction_error(W=W, U=U_next, X=X),
                    "sparsity": actual_sparsity(U_next),
                    "diff_norm": float(diff_norm),
                }
            )

            if self.tolerance > 0 and diff_norm <= self.tolerance:
                U_k = U_next
                break

            t_next, momentum = nesterov_coefficient(t_k)
            Z_k = U_next + momentum * (U_next - U_k)
            U_k = U_next
            t_k = t_next

        stats = {
            "method": "fista",
            "num_iters": len(history),
            "lambda": self.lambda_,
            "lipschitz": float(L),
            "actual_sparsity": actual_sparsity(U_k),
            "num_total": int(U_k.numel()),
            "num_nonzero": count_nonzero(U_k),
            "num_zero": count_zero(U_k),
            "reconstruction_error": _reconstruction_error(W=W, U=U_k, X=X),
            "l1_norm": l1_norm(U_k),
            "objective": _objective_value(W=W, U=U_k, X=X, lambda_=self.lambda_),
        }

        return PruneResult(U=U_k, stats=stats, history=history)
