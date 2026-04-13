from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ActivationHook:
    module: nn.Module
    move_to_cpu: bool = True
    flatten_batch: bool = True
    _handle: torch.utils.hooks.RemovableHandle | None = field(init=False, default=None)
    _inputs: list[torch.Tensor] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._handle = self.module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, output
        if not inputs:
            return
        tensor = inputs[0].detach()
        if self.flatten_batch and tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if self.move_to_cpu:
            tensor = tensor.cpu()
        self._inputs.append(tensor)

    def stacked_inputs(self) -> torch.Tensor:
        if not self._inputs:
            raise ValueError("No activations have been captured yet")
        stacked = torch.cat(self._inputs, dim=0)
        return stacked.transpose(0, 1).contiguous()

    def clear(self) -> None:
        self._inputs.clear()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def resolve_module(model: nn.Module, module_name: str) -> nn.Module:
    try:
        return model.get_submodule(module_name)
    except AttributeError as exc:
        raise ValueError(f"Model does not have a submodule named '{module_name}'") from exc


def is_supported_prunable_module(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")


def list_supported_prunable_modules(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if name and is_supported_prunable_module(module):
            names.append(name)
    return names


def choose_default_prunable_module(model: nn.Module) -> str:
    candidates = list_supported_prunable_modules(model)
    if not candidates:
        raise ValueError("No supported prunable modules were found in the model")

    for name in candidates:
        lowered = name.lower()
        if "lm_head" not in lowered and "embed" not in lowered and "wte" not in lowered:
            return name
    return candidates[0]


def extract_weight_matrix(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight.detach().cpu().to(dtype=torch.float32)

    if module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
        weight = module.weight.detach().cpu()
        if weight.ndim != 2:
            raise ValueError("Conv1D weight must be rank-2")
        return weight.transpose(0, 1).contiguous().to(dtype=torch.float32)

    raise TypeError(f"Unsupported module type for pruning: {type(module).__name__}")


def apply_weight_matrix(module: nn.Module, weight: torch.Tensor) -> None:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2, got shape {tuple(weight.shape)}")

    target_dtype = module.weight.dtype
    target_device = module.weight.device

    if isinstance(module, nn.Linear):
        expected_shape = tuple(module.weight.shape)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"Linear weight shape mismatch: expected {expected_shape}, got {tuple(weight.shape)}"
            )
        with torch.no_grad():
            module.weight.copy_(weight.to(device=target_device, dtype=target_dtype))
        return

    if module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
        expected_shape = tuple(module.weight.shape)
        source = weight.transpose(0, 1).contiguous()
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"Conv1D weight shape mismatch: expected {expected_shape}, got {tuple(source.shape)}"
            )
        with torch.no_grad():
            module.weight.copy_(source.to(device=target_device, dtype=target_dtype))
        return

    raise TypeError(f"Unsupported module type for pruning: {type(module).__name__}")
