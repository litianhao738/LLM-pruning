import math
from typing import Any

import torch
from tqdm.auto import tqdm


def perplexity_from_average_nll(avg_nll: float) -> float:
    if avg_nll < 0:
        raise ValueError(f"avg_nll must be non-negative, got {avg_nll}")
    return float(math.exp(avg_nll))


def average_nll_from_texts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 64,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    show_progress: bool = False,
    progress_desc: str = "Evaluating NLL",
) -> float:
    if not texts:
        raise ValueError("texts must be non-empty")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    device = torch.device(device)
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc, leave=False)

    with torch.no_grad():
        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = encoded["input_ids"].clone()
            if "attention_mask" in encoded:
                labels[encoded["attention_mask"] == 0] = -100

            outputs = model(**encoded, labels=labels)
            valid_tokens = int((labels != -100).sum().item())
            total_nll += float(outputs.loss.item()) * valid_tokens
            total_tokens += valid_tokens

    if was_training:
        model.train()

    if total_tokens <= 0:
        raise RuntimeError("No valid tokens were evaluated")
    return total_nll / total_tokens


def evaluate_perplexity_on_texts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 64,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    show_progress: bool = False,
    progress_desc: str = "Evaluating perplexity",
) -> dict[str, float]:
    avg_nll = average_nll_from_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return {
        "average_nll": float(avg_nll),
        "perplexity": perplexity_from_average_nll(avg_nll),
    }
