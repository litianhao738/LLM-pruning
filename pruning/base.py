from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PruneResult:
    U: torch.Tensor
    stats: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)


class BasePruner:
    def prune(self, W: torch.Tensor, X: torch.Tensor) -> PruneResult:
        raise NotImplementedError
