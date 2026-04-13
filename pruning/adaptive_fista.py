from pruning.fista import FISTAPruner


def cosine_threshold_schedule(
    lambda0: float,
    step: int,
    total_steps: int,
    r_min: float,
    r_max: float,
) -> float:
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


class AdaptiveThresholdFISTAPruner(FISTAPruner):
    """
    Innovation 2 from 7503Pre:
    use an iteration-dependent threshold lambda_k = c_k * lambda_0.
    """

    def __init__(
        self,
        lambda_: float,
        num_iters: int = 50,
        r_min: float = 0.5,
        r_max: float = 1.5,
        lipschitz: float | None = None,
        tolerance: float = 0.0,
    ):
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        super().__init__(
            lambda_=lambda_,
            num_iters=num_iters,
            lipschitz=lipschitz,
            tolerance=tolerance,
            schedule=self._schedule,
        )

    def _schedule(self, lambda0: float, step: int, total_steps: int) -> float:
        return cosine_threshold_schedule(
            lambda0=lambda0,
            step=step,
            total_steps=total_steps,
            r_min=self.r_min,
            r_max=self.r_max,
        )

    def prune(self, W, X):
        result = super().prune(W=W, X=X)
        result.stats["method"] = "adaptive_fista"
        result.stats["r_min"] = self.r_min
        result.stats["r_max"] = self.r_max
        return result
