import torch

from pruning.base import PruneResult
from pruning.fista import FISTAPruner, _objective_value, _reconstruction_error
from utils.math_utils import (
    estimate_lipschitz_from_gram,
    gram_matrix,
    l1_norm,
    nesterov_coefficient,
    soft_threshold,
)
from utils.sparsity import actual_sparsity, count_nonzero, count_zero


def cosine_threshold_schedule(
    lambda0: float,
    step: int,
    total_steps: int,
    r_min: float,
    r_max: float,
) -> float:
    """
    Legacy iteration-driven schedule kept for backward compatibility.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if r_min <= 0 or r_max <= 0:
        raise ValueError("r_min and r_max must be positive")

    if total_steps == 1:
        coeff = r_max
    else:
        import math

        phase = math.pi * step / (total_steps - 1)
        coeff = r_min + 0.5 * (r_max - r_min) * (1.0 - math.cos(phase))

    return float(coeff * lambda0)


def sparsity_gap_threshold(
    lambda0: float,
    *,
    target_sparsity: float,
    estimated_sparsity: float,
    alpha: float,
    r_min: float,
    r_max: float,
) -> float:
    if r_min <= 0 or r_max <= 0:
        raise ValueError("r_min and r_max must be positive")
    if r_min > r_max:
        raise ValueError("r_min must be <= r_max")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    coeff = 1.0 + alpha * (target_sparsity - estimated_sparsity)
    coeff = min(max(coeff, r_min), r_max)
    return float(coeff * lambda0)


class AdaptiveThresholdFISTAPruner(FISTAPruner):
    """
    Adaptive FISTA with sparsity-gap-driven thresholding.

    Instead of changing the threshold only as a function of iteration index,
    this variant adjusts lambda_k based on how far the current sparsity is from
    the target sparsity. The current sparsity estimate is smoothed with EMA to
    avoid oscillation.
    """

    def __init__(
        self,
        lambda_: float,
        num_iters: int = 50,
        r_min: float = 0.9,
        r_max: float = 1.1,
        target_sparsity: float = 0.5,
        alpha: float = 1.0,
        ema_beta: float = 0.9,
        lipschitz: float | None = None,
        tolerance: float = 0.0,
    ):
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.target_sparsity = float(target_sparsity)
        self.alpha = float(alpha)
        self.ema_beta = float(ema_beta)
        super().__init__(
            lambda_=lambda_,
            num_iters=num_iters,
            lipschitz=lipschitz,
            tolerance=tolerance,
        )

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
        if not (0.0 <= self.target_sparsity <= 1.0):
            raise ValueError("target_sparsity must be in [0, 1]")
        if not (0.0 <= self.ema_beta < 1.0):
            raise ValueError("ema_beta must be in [0, 1)")

        G = gram_matrix(X)
        WG = W @ G
        L = self.lipschitz or estimate_lipschitz_from_gram(G)
        if L <= 0:
            raise ValueError("Estimated Lipschitz constant must be positive")

        U_k = W.clone()
        Z_k = U_k.clone()
        t_k = 1.0
        history: list[dict[str, float]] = []
        estimated_sparsity = actual_sparsity(U_k)

        for step in range(self.num_iters):
            lambda_k = sparsity_gap_threshold(
                self.lambda_,
                target_sparsity=self.target_sparsity,
                estimated_sparsity=estimated_sparsity,
                alpha=self.alpha,
                r_min=self.r_min,
                r_max=self.r_max,
            )
            grad = Z_k @ G - WG
            V_k = Z_k - grad / L
            U_next = soft_threshold(V_k, threshold=lambda_k / L)
            diff_norm = torch.linalg.norm(U_next - U_k).item()
            current_sparsity = actual_sparsity(U_next)
            estimated_sparsity = (
                self.ema_beta * estimated_sparsity + (1.0 - self.ema_beta) * current_sparsity
            )

            current_objective = _objective_value(W=W, U=U_next, X=X, lambda_=lambda_k)
            history.append(
                {
                    "step": float(step),
                    "lambda": float(lambda_k),
                    "objective": float(current_objective),
                    "reconstruction_error": _reconstruction_error(W=W, U=U_next, X=X),
                    "sparsity": current_sparsity,
                    "estimated_sparsity": float(estimated_sparsity),
                    "sparsity_gap": float(abs(self.target_sparsity - estimated_sparsity)),
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

        final_lambda = history[-1]["lambda"] if history else self.lambda_
        result = PruneResult(
            U=U_k,
            stats={
                "method": "adaptive_fista",
                "num_iters": len(history),
                "lambda": self.lambda_,
                "lambda_final": float(final_lambda),
                "lipschitz": float(L),
                "actual_sparsity": actual_sparsity(U_k),
                "target_sparsity": self.target_sparsity,
                "alpha": self.alpha,
                "ema_beta": self.ema_beta,
                "r_min": self.r_min,
                "r_max": self.r_max,
                "num_total": int(U_k.numel()),
                "num_nonzero": count_nonzero(U_k),
                "num_zero": count_zero(U_k),
                "reconstruction_error": _reconstruction_error(W=W, U=U_k, X=X),
                "l1_norm": l1_norm(U_k),
                "objective": _objective_value(W=W, U=U_k, X=X, lambda_=self.lambda_),
            },
            history=history,
        )
        result.stats["method"] = "adaptive_fista"
        return result
