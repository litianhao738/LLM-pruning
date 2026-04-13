import torch

from pruning.adaptive_fista import AdaptiveThresholdFISTAPruner
from pruning.base import PruneResult
from pruning.fista import _objective_value, _reconstruction_error
from utils.math_utils import estimate_lipschitz_from_gram, gram_matrix, l1_norm, nesterov_coefficient, soft_threshold
from utils.sparsity import actual_sparsity, count_nonzero, count_zero


class GradientAwareMomentumFISTAPruner(AdaptiveThresholdFISTAPruner):
    """
    Backup innovation from the draft:
    modulate the FISTA momentum using the current gradient norm.

    The momentum scale is:

        phi_k = 1 / (1 + beta * ||grad_k|| / (||grad_0|| + eps))

    So large gradients reduce acceleration and small gradients recover the
    baseline FISTA momentum over time.
    """

    def __init__(
        self,
        lambda_: float,
        num_iters: int = 50,
        r_min: float = 0.5,
        r_max: float = 1.5,
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
        eps = 1e-12

        for step in range(self.num_iters):
            lambda_k = self.schedule(self.lambda_, step, self.num_iters)
            grad = Z_k @ G - WG
            grad_norm = float(torch.linalg.norm(grad).item())

            V_k = Z_k - grad / L
            U_next = soft_threshold(V_k, threshold=lambda_k / L)
            diff_norm = torch.linalg.norm(U_next - U_k).item()
            base_t_next, base_momentum = nesterov_coefficient(t_k)

            # The first FISTA gradient can be exactly zero because we initialize
            # at U_0 = W. Delay reference initialization until we see a real
            # non-zero gradient, otherwise all later momentum gets squashed.
            if reference_grad_norm is None and grad_norm > eps:
                reference_grad_norm = grad_norm

            if reference_grad_norm is None:
                momentum_scale = 1.0
            else:
                momentum_scale = 1.0 / (
                    1.0 + self.momentum_beta * grad_norm / (reference_grad_norm + eps)
                )
            adapted_momentum = base_momentum * momentum_scale

            current_objective = _objective_value(W=W, U=U_next, X=X, lambda_=lambda_k)
            history.append(
                {
                    "step": float(step),
                    "lambda": float(lambda_k),
                    "objective": float(current_objective),
                    "reconstruction_error": _reconstruction_error(W=W, U=U_next, X=X),
                    "sparsity": actual_sparsity(U_next),
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
            t_k = base_t_next

        stats = {
            "method": "gradient_momentum_fista",
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
            "r_min": self.r_min,
            "r_max": self.r_max,
            "momentum_beta": self.momentum_beta,
        }

        return PruneResult(U=U_k, stats=stats, history=history)
