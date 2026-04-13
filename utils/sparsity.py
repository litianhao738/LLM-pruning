import torch


def actual_sparsity(U: torch.Tensor) -> float:
    return float((U == 0).sum().item() / U.numel())


def count_nonzero(U: torch.Tensor) -> int:
    return int((U != 0).sum().item())


def count_zero(U: torch.Tensor) -> int:
    return int((U == 0).sum().item())
