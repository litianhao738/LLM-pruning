from typing import Any

import torch

from data.calibration import load_calibration_text_corpus
from pruning import (
    AdaptiveThresholdFISTAPruner,
    FISTAPruner,
    GradientAwareMomentumFISTAPruner,
    MagnitudePruner,
    OriginalGradientAwareMomentumFISTAPruner,
    find_lambda_for_target_sparsity,
)


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("methods must contain at least one method name")
    return methods


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_texts(texts: list[str], first_count: int, second_count: int) -> tuple[list[str], list[str]]:
    required = first_count + second_count
    if len(texts) < required:
        raise ValueError(f"Not enough texts for split: need {required}, got {len(texts)}")
    first = texts[:first_count]
    second = texts[first_count : first_count + second_count]
    return first, second


def select_eval_texts(
    *,
    source: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_key: str,
    eval_start_index: int,
    eval_texts: int,
    min_chars: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[str], dict[str, Any]]:
    required_texts = eval_start_index + eval_texts
    corpus = load_calibration_text_corpus(
        source,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        text_key=text_key,
        max_texts=required_texts,
        min_chars=min_chars,
        seed=seed,
        shuffle=shuffle,
    )
    if len(corpus.texts) < required_texts:
        raise ValueError(
            f"Not enough texts for evaluation slice: need {required_texts}, got {len(corpus.texts)}"
        )
    eval_slice = corpus.texts[eval_start_index : eval_start_index + eval_texts]
    metadata = dict(corpus.metadata)
    metadata["eval_start_index"] = int(eval_start_index)
    metadata["eval_texts"] = int(eval_texts)
    return eval_slice, metadata


def select_finetune_and_eval_texts(
    *,
    source: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_key: str,
    finetune_texts: int,
    eval_texts: int,
    min_chars: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[str], list[str], dict[str, Any]]:
    corpus = load_calibration_text_corpus(
        source,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        text_key=text_key,
        max_texts=finetune_texts + eval_texts,
        min_chars=min_chars,
        seed=seed,
        shuffle=shuffle,
    )
    finetune_slice, eval_slice = split_texts(corpus.texts, finetune_texts, eval_texts)
    return finetune_slice, eval_slice, dict(corpus.metadata)


def method_settings(
    method: str,
    *,
    default_r_min: float,
    default_r_max: float,
    default_momentum_beta: float,
    adaptive_r_min: float | None = None,
    adaptive_r_max: float | None = None,
    gradient_r_min: float | None = None,
    gradient_r_max: float | None = None,
    gradient_momentum_beta: float | None = None,
) -> dict[str, float | None]:
    settings: dict[str, float | None] = {
        "r_min": default_r_min,
        "r_max": default_r_max,
        "momentum_beta": default_momentum_beta,
    }
    if method == "adaptive_fista":
        settings["r_min"] = adaptive_r_min if adaptive_r_min is not None else default_r_min
        settings["r_max"] = adaptive_r_max if adaptive_r_max is not None else default_r_max
        settings["momentum_beta"] = None
    elif method in {"gradient_momentum_fista", "gradient_momentum_fista_original"}:
        settings["r_min"] = gradient_r_min if gradient_r_min is not None else default_r_min
        settings["r_max"] = gradient_r_max if gradient_r_max is not None else default_r_max
        settings["momentum_beta"] = (
            gradient_momentum_beta
            if gradient_momentum_beta is not None
            else default_momentum_beta
        )
    elif method in {"magnitude", "fista"}:
        settings["r_min"] = None
        settings["r_max"] = None
        settings["momentum_beta"] = None
    else:
        raise ValueError(f"Unsupported method: {method}")
    return settings


def build_prune_result(
    *,
    method: str,
    target_sparsity: float,
    W: torch.Tensor,
    X: torch.Tensor,
    num_iters: int,
    search_steps: int,
    sparsity_tol: float,
    show_progress: bool,
    progress_desc: str,
    settings: dict[str, float | None] | None = None,
    lambda_low: float | None = None,
    lambda_high: float | None = None,
    bracket_scale: float | None = None,
    max_bracket_steps: int | None = None,
) -> dict[str, Any]:
    if method == "magnitude":
        result = MagnitudePruner(sparsity=target_sparsity).prune(W=W, X=X)
        return {
            "prune_result": result,
            "selected_lambda": None,
            "search": None,
        }

    if method == "fista":
        pruner_cls = FISTAPruner
        pruner_kwargs: dict[str, float] = {}
    elif method == "adaptive_fista":
        if settings is None:
            raise ValueError("adaptive_fista requires method settings")
        pruner_cls = AdaptiveThresholdFISTAPruner
        pruner_kwargs = {
            "r_min": float(settings["r_min"]),
            "r_max": float(settings["r_max"]),
            "target_sparsity": float(target_sparsity),
        }
    elif method == "gradient_momentum_fista":
        if settings is None:
            raise ValueError("gradient_momentum_fista requires method settings")
        pruner_cls = GradientAwareMomentumFISTAPruner
        pruner_kwargs = {
            "r_min": float(settings["r_min"]),
            "r_max": float(settings["r_max"]),
            "momentum_beta": float(settings["momentum_beta"]),
        }
    elif method == "gradient_momentum_fista_original":
        if settings is None:
            raise ValueError("gradient_momentum_fista_original requires method settings")
        pruner_cls = OriginalGradientAwareMomentumFISTAPruner
        pruner_kwargs = {
            "r_min": float(settings["r_min"]),
            "r_max": float(settings["r_max"]),
            "momentum_beta": float(settings["momentum_beta"]),
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    search_kwargs: dict[str, Any] = {
        "pruner_cls": pruner_cls,
        "W": W,
        "X": X,
        "target_sparsity": target_sparsity,
        "num_iters": num_iters,
        "search_steps": search_steps,
        "sparsity_tol": sparsity_tol,
        "pruner_kwargs": pruner_kwargs,
        "show_progress": show_progress,
        "progress_desc": progress_desc,
    }
    if lambda_low is not None:
        search_kwargs["lambda_low"] = lambda_low
    if lambda_high is not None:
        search_kwargs["lambda_high"] = lambda_high
    if bracket_scale is not None:
        search_kwargs["bracket_scale"] = bracket_scale
    if max_bracket_steps is not None:
        search_kwargs["max_bracket_steps"] = max_bracket_steps

    search = find_lambda_for_target_sparsity(**search_kwargs)
    return {
        "prune_result": search.best_result,
        "selected_lambda": float(search.best_lambda),
        "search": {
            "target_sparsity": float(search.target_sparsity),
            "best_lambda": float(search.best_lambda),
            "best_actual_sparsity": float(search.best_actual_sparsity),
            "best_gap": float(search.best_gap),
            "bracket_low": float(search.bracket_low),
            "bracket_high": float(search.bracket_high),
            "bracket_found": bool(search.bracket_found),
            "terminated_reason": search.terminated_reason,
            "num_trials": len(search.trials),
            "trials": search.trials,
        },
    }
