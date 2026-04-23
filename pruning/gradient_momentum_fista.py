import torch

from pruning.adaptive_fista import AdaptiveThresholdFISTAPruner, sparsity_gap_threshold
from pruning.base import PruneResult
from pruning.fista import _objective_value, _reconstruction_error
from utils.math_utils import estimate_lipschitz_from_gram, gram_matrix, l1_norm, nesterov_coefficient, soft_threshold
from utils.sparsity import actual_sparsity, count_nonzero, count_zero


class GradientAwareMomentumFISTAPruner(AdaptiveThresholdFISTAPruner):
    """
    Upgraded gradient-aware momentum FISTA:
    - smooth gradient norms with an EMA
    - bound the momentum scale away from zero
    - allow optional restart when the accelerated step becomes unstable
    - only apply momentum modulation in the early portion of the run
    """

    def __init__(
        self,
        lambda_: float,
        num_iters: int = 50,
        r_min: float = 0.9,
        r_max: float = 1.1,
        target_sparsity: float = 0.5,
        momentum_beta: float = 1.0,
        lipschitz: float | None = None,
        tolerance: float = 0.0,
        grad_ema_rho: float = 0.9,
        momentum_floor: float = 0.9,
        modulation_fraction: float = 0.4,
        enable_restart: bool = True,
    ):
        if momentum_beta < 0:
            raise ValueError(f"momentum_beta must be non-negative, got {momentum_beta}")
        if not (0.0 <= grad_ema_rho < 1.0):
            raise ValueError(f"grad_ema_rho must be in [0, 1), got {grad_ema_rho}")
        if not (0.0 < momentum_floor <= 1.0):
            raise ValueError(f"momentum_floor must be in (0, 1], got {momentum_floor}")
        if not (0.0 < modulation_fraction <= 1.0):
            raise ValueError(
                f"modulation_fraction must be in (0, 1], got {modulation_fraction}"
            )
        self.momentum_beta = float(momentum_beta)
        self.grad_ema_rho = float(grad_ema_rho)
        self.momentum_floor = float(momentum_floor)
        self.modulation_fraction = float(modulation_fraction)
        self.enable_restart = bool(enable_restart)
        super().__init__(
            lambda_=lambda_,
            num_iters=num_iters,
            r_min=r_min,
            r_max=r_max,
            target_sparsity=target_sparsity,
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

        G = gram_matrix(X)
        WG = W @ G
        L = self.lipschitz or estimate_lipschitz_from_gram(G)
        if L <= 0:
            raise ValueError("Estimated Lipschitz constant must be positive")

        U_k = W.clone()
        Z_k = U_k.clone()
        t_k = 1.0
        history: list[dict[str, float]] = []
        reference_grad_norm: float | None = None
        smoothed_grad_norm: float | None = None
        prev_delta: torch.Tensor | None = None
        prev_objective: float | None = None
        estimated_sparsity = actual_sparsity(U_k)
        eps = 1e-12
        modulation_steps = max(1, int(round(self.num_iters * self.modulation_fraction)))

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
            grad_norm = float(torch.linalg.norm(grad).item())
            if smoothed_grad_norm is None:
                smoothed_grad_norm = grad_norm
            else:
                smoothed_grad_norm = (
                    self.grad_ema_rho * smoothed_grad_norm
                    + (1.0 - self.grad_ema_rho) * grad_norm
                )

            V_k = Z_k - grad / L
            U_next = soft_threshold(V_k, threshold=lambda_k / L)
            diff_norm = torch.linalg.norm(U_next - U_k).item()
            current_sparsity = actual_sparsity(U_next)
            estimated_sparsity = (
                self.ema_beta * estimated_sparsity + (1.0 - self.ema_beta) * current_sparsity
            )
            base_t_next, base_momentum = nesterov_coefficient(t_k)

            # The first FISTA gradient can be exactly zero because we initialize
            # at U_0 = W. Delay reference initialization until we see a real
            # non-zero gradient, otherwise all later momentum gets squashed.
            if reference_grad_norm is None and grad_norm > eps:
                reference_grad_norm = float(smoothed_grad_norm)

            if step >= modulation_steps:
                momentum_scale = 1.0
            elif reference_grad_norm is None:
                momentum_scale = 1.0
            else:
                raw_scale = 1.0 / (
                    1.0
                    + self.momentum_beta
                    * float(smoothed_grad_norm)
                    / (reference_grad_norm + eps)
                )
                momentum_scale = max(self.momentum_floor, raw_scale)
            adapted_momentum = base_momentum * momentum_scale

            current_objective = _objective_value(W=W, U=U_next, X=X, lambda_=lambda_k)
            delta = U_next - U_k
            restart_due_to_objective = (
                self.enable_restart
                and prev_objective is not None
                and current_objective > prev_objective + eps
            )
            restart_due_to_direction = (
                self.enable_restart
                and prev_delta is not None
                and float((delta * prev_delta).sum().item()) > 0.0
            )
            restart_triggered = restart_due_to_objective or restart_due_to_direction
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
                    "grad_norm": float(grad_norm),
                    "smoothed_grad_norm": float(smoothed_grad_norm),
                    "reference_grad_norm": float(reference_grad_norm or 0.0),
                    "base_momentum": float(base_momentum),
                    "momentum_scale": float(momentum_scale),
                    "adapted_momentum": float(adapted_momentum),
                    "restart_due_to_objective": float(restart_due_to_objective),
                    "restart_due_to_direction": float(restart_due_to_direction),
                    "restart_triggered": float(restart_triggered),
                }
            )

            if self.tolerance > 0 and diff_norm <= self.tolerance:
                U_k = U_next
                break

            if restart_triggered:
                Z_k = U_next.clone()
                t_k = 1.0
            else:
                Z_k = U_next + adapted_momentum * delta
                t_k = base_t_next
            prev_delta = delta.clone()
            prev_objective = float(current_objective)
            U_k = U_next

        stats = {
            "method": "gradient_momentum_fista",
            "num_iters": len(history),
            "lambda": self.lambda_,
            "lambda_final": float(history[-1]["lambda"]) if history else self.lambda_,
            "lipschitz": float(L),
            "actual_sparsity": actual_sparsity(U_k),
            "num_total": int(U_k.numel()),
            "num_nonzero": count_nonzero(U_k),
            "num_zero": count_zero(U_k),
            "reconstruction_error": _reconstruction_error(W=W, U=U_k, X=X),
            "l1_norm": l1_norm(U_k),
            "objective": _objective_value(W=W, U=U_k, X=X, lambda_=self.lambda_),
            "target_sparsity": self.target_sparsity,
            "alpha": self.alpha,
            "ema_beta": self.ema_beta,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "momentum_beta": self.momentum_beta,
            "grad_ema_rho": self.grad_ema_rho,
            "momentum_floor": self.momentum_floor,
            "modulation_fraction": self.modulation_fraction,
            "num_restarts": int(sum(int(item["restart_triggered"]) for item in history)),
        }

        return PruneResult(U=U_k, stats=stats, history=history)


class OriginalGradientAwareMomentumFISTAPruner(AdaptiveThresholdFISTAPruner):
    """
    Original gradient-aware momentum FISTA variant used before the upgrade:
    - no EMA smoothing
    - no restart
    - no lower bound on momentum scaling
    - momentum modulation is applied throughout all iterations
    """

    def __init__(
        self,
        lambda_: float,
        num_iters: int = 50,
        r_min: float = 0.9,
        r_max: float = 1.1,
        target_sparsity: float = 0.5,
        momentum_beta: float = 1.0,
        lipschitz: float | None = None,
        tolerance: float = 0.0,
    ):
        if momentum_beta < 0:
            raise ValueError(f"momentum_beta must be non-negative, got {momentum_beta}")
        self.momentum_beta = float(momentum_beta)
        super().__init__(
            lambda_=lambda_,
            num_iters=num_iters,
            r_min=r_min,
            r_max=r_max,
            target_sparsity=target_sparsity,
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

        G = gram_matrix(X)
        WG = W @ G
        L = self.lipschitz or estimate_lipschitz_from_gram(G)
        if L <= 0:
            raise ValueError("Estimated Lipschitz constant must be positive")

        U_k = W.clone()
        Z_k = U_k.clone()
        t_k = 1.0
        history: list[dict[str, float]] = []
        reference_grad_norm: float | None = None
        estimated_sparsity = actual_sparsity(U_k)
        eps = 1e-12

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
            grad_norm = float(torch.linalg.norm(grad).item())
            V_k = Z_k - grad / L
            U_next = soft_threshold(V_k, threshold=lambda_k / L)
            diff_norm = torch.linalg.norm(U_next - U_k).item()
            current_sparsity = actual_sparsity(U_next)
            estimated_sparsity = (
                self.ema_beta * estimated_sparsity + (1.0 - self.ema_beta) * current_sparsity
            )
            t_next, base_momentum = nesterov_coefficient(t_k)

            if reference_grad_norm is None and grad_norm > eps:
                reference_grad_norm = grad_norm
            if reference_grad_norm is None:
                momentum_scale = 1.0
            else:
                momentum_scale = 1.0 / (
                    1.0 + self.momentum_beta * grad_norm / (reference_grad_norm + eps)
                )
            adapted_momentum = base_momentum * momentum_scale

            history.append(
                {
                    "step": float(step),
                    "lambda": float(lambda_k),
                    "objective": float(_objective_value(W=W, U=U_next, X=X, lambda_=lambda_k)),
                    "reconstruction_error": _reconstruction_error(W=W, U=U_next, X=X),
                    "sparsity": current_sparsity,
                    "estimated_sparsity": float(estimated_sparsity),
                    "sparsity_gap": float(abs(self.target_sparsity - estimated_sparsity)),
                    "diff_norm": float(diff_norm),
                    "grad_norm": float(grad_norm),
                    "reference_grad_norm": float(reference_grad_norm or 0.0),
                    "base_momentum": float(base_momentum),
                    "momentum_scale": float(momentum_scale),
                    "adapted_momentum": float(adapted_momentum),
                }
            )

            if self.tolerance > 0 and diff_norm <= self.tolerance:
                U_k = U_next
                break

            Z_k = U_next + adapted_momentum * (U_next - U_k)
            U_k = U_next
            t_k = t_next

        stats = {
            "method": "gradient_momentum_fista_original",
            "num_iters": len(history),
            "lambda": self.lambda_,
            "lambda_final": float(history[-1]["lambda"]) if history else self.lambda_,
            "lipschitz": float(L),
            "actual_sparsity": actual_sparsity(U_k),
            "num_total": int(U_k.numel()),
            "num_nonzero": count_nonzero(U_k),
            "num_zero": count_zero(U_k),
            "reconstruction_error": _reconstruction_error(W=W, U=U_k, X=X),
            "l1_norm": l1_norm(U_k),
            "objective": _objective_value(W=W, U=U_k, X=X, lambda_=self.lambda_),
            "target_sparsity": self.target_sparsity,
            "alpha": self.alpha,
            "ema_beta": self.ema_beta,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "momentum_beta": self.momentum_beta,
        }

        return PruneResult(U=U_k, stats=stats, history=history)
