import math

import torch


def l1_norm(U: torch.Tensor) -> float:
    return float(U.abs().sum().item())


def soft_threshold(Z: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    return Z.sign() * torch.clamp(Z.abs() - threshold, min=0.0)


def gram_matrix(X: torch.Tensor) -> torch.Tensor:
    return X @ X.transpose(0, 1)


def estimate_lipschitz_from_gram(G: torch.Tensor, num_iters: int = 20) -> float:
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be a square matrix")
    if G.numel() == 0:
        raise ValueError("G must be non-empty")

    vec = torch.ones(G.shape[0], 1, dtype=G.dtype, device=G.device)
    vec = vec / torch.linalg.norm(vec)

    for _ in range(num_iters):
        vec = G @ vec
        norm = torch.linalg.norm(vec)
        if float(norm.item()) == 0.0:
            return 0.0
        vec = vec / norm

    rayleigh = (vec.transpose(0, 1) @ G @ vec).item()
    return float(max(rayleigh, 0.0))


def nesterov_coefficient(t_k: float) -> tuple[float, float]:
    t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
    momentum = (t_k - 1.0) / t_next
    return t_next, momentum
