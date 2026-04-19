from collections.abc import Iterable

import torch
import torch.nn as nn


ParameterMask = tuple[torch.nn.Parameter, torch.Tensor]


def build_module_weight_mask(module: nn.Module) -> ParameterMask:
    if not hasattr(module, "weight"):
        raise TypeError(f"Module {type(module).__name__} does not expose a weight parameter")
    weight_param = module.weight
    mask = weight_param.detach().ne(0).to(device=weight_param.device, dtype=weight_param.dtype)
    return weight_param, mask


def build_module_weight_masks(modules: Iterable[nn.Module]) -> list[ParameterMask]:
    return [build_module_weight_mask(module) for module in modules]


def mask_parameter_grads(parameter_masks: Iterable[ParameterMask]) -> None:
    for parameter, mask in parameter_masks:
        if parameter.grad is not None:
            parameter.grad.mul_(mask)


def apply_parameter_masks(parameter_masks: Iterable[ParameterMask]) -> None:
    with torch.no_grad():
        for parameter, mask in parameter_masks:
            parameter.mul_(mask)
