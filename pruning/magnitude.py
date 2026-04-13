import torch

from pruning.base import BasePruner, PruneResult
from utils.sparsity import actual_sparsity, count_nonzero, count_zero


class MagnitudePruner(BasePruner):
    """
    Magnitude pruning baseline.

    Given a target sparsity s in [0, 1), keep the largest (1-s) fraction
    of weights by absolute value and set the rest to zero.

    Notes:
    - This baseline does NOT use activation matrix X.
    - It is a heuristic baseline corresponding to the slides' magnitude pruning idea.
    """

    def __init__(self, sparsity: float):
        if not (0.0 <= sparsity < 1.0):
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
        self.sparsity = float(sparsity)

    def prune(self, W: torch.Tensor, X: torch.Tensor) -> PruneResult:
        """
        Args:
            W: dense weight matrix, shape [m, n]
            X: activation matrix, unused here, included only for unified API

        Returns:
            PruneResult with:
              - U: pruned weight matrix
              - stats: metadata dict
        """
        if not isinstance(W, torch.Tensor):
            raise TypeError("W must be a torch.Tensor")
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor")
        if W.numel() == 0:
            raise ValueError("W must be non-empty")

        num_total = W.numel()
        num_keep = int(round(num_total * (1.0 - self.sparsity)))

        if num_keep <= 0:
            U = torch.zeros_like(W)
        elif num_keep >= num_total:
            U = W.clone()
        else:
            flat_abs = W.abs().flatten()

            # topk returns the largest num_keep entries
            topk_vals = torch.topk(flat_abs, k=num_keep, largest=True, sorted=False).values
            threshold = topk_vals.min()

            # Keep all entries with abs >= threshold
            # Note: ties at threshold may cause slight deviation from exact target sparsity
            mask = (W.abs() >= threshold).to(dtype=W.dtype)
            U = W * mask

        stats = {
            "method": "magnitude",
            "target_sparsity": self.sparsity,
            "actual_sparsity": actual_sparsity(U),
            "num_total": int(num_total),
            "num_nonzero": count_nonzero(U),
            "num_zero": count_zero(U),
        }

        return PruneResult(U=U, stats=stats)
