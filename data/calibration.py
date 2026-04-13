from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CalibrationData:
    activations: torch.Tensor
    metadata: dict[str, int | float | str]


@dataclass
class CalibrationTextCorpus:
    texts: list[str]
    metadata: dict[str, Any]


def default_calibration_texts() -> list[str]:
    return [
        "Large language models need efficient pruning for cheaper deployment.",
        "Optimization-based pruning can preserve model behavior better than pure heuristics.",
        "Calibration activations provide a lightweight way to match layer outputs after pruning.",
        "FISTA combines gradient steps with soft-thresholding for sparse optimization.",
        "Adaptive threshold schedules may improve the trade-off between flexibility and sparsity.",
        "Post-training pruning is attractive because full retraining is expensive for large models.",
        "Transformer layers often contain redundant weights that can be removed safely.",
        "Model compression matters for latency, memory footprint, and energy efficiency.",
    ]


def build_default_text_corpus() -> CalibrationTextCorpus:
    texts = default_calibration_texts()
    return CalibrationTextCorpus(
        texts=texts,
        metadata={
            "calibration_source": "default_texts",
            "num_texts": len(texts),
        },
    )


def load_hf_calibration_texts(
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    text_key: str = "text",
    max_texts: int = 64,
    min_chars: int = 20,
    seed: int = 7,
    shuffle: bool = True,
) -> CalibrationTextCorpus:
    if max_texts <= 0:
        raise ValueError(f"max_texts must be positive, got {max_texts}")
    if min_chars < 0:
        raise ValueError(f"min_chars must be non-negative, got {min_chars}")

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    texts: list[str] = []
    inspected_rows = 0
    for row in dataset:
        inspected_rows += 1
        value = row.get(text_key, "")
        if value is None:
            continue
        text = str(value).strip()
        if len(text) < min_chars:
            continue
        texts.append(text)
        if len(texts) >= max_texts:
            break

    if not texts:
        raise ValueError(
            "No calibration texts were collected from the dataset. "
            "Try lowering min_chars or checking the dataset/text_key."
        )

    metadata = {
        "calibration_source": "hf_dataset",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": split,
        "text_key": text_key,
        "num_texts": len(texts),
        "min_chars": min_chars,
        "shuffle": shuffle,
        "seed": seed,
        "inspected_rows": inspected_rows,
    }
    return CalibrationTextCorpus(texts=texts, metadata=metadata)


def load_calibration_text_corpus(
    source: str,
    *,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    text_key: str = "text",
    max_texts: int = 64,
    min_chars: int = 20,
    seed: int = 7,
    shuffle: bool = True,
) -> CalibrationTextCorpus:
    if source == "default_texts":
        return build_default_text_corpus()
    if source == "wikitext103":
        return load_hf_calibration_texts(
            dataset_name="Salesforce/wikitext",
            dataset_config="wikitext-103-raw-v1",
            split=split,
            text_key=text_key,
            max_texts=max_texts,
            min_chars=min_chars,
            seed=seed,
            shuffle=shuffle,
        )
    if source == "hf_dataset":
        return load_hf_calibration_texts(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            text_key=text_key,
            max_texts=max_texts,
            min_chars=min_chars,
            seed=seed,
            shuffle=shuffle,
        )
    raise ValueError(f"Unsupported calibration source: {source}")


def make_synthetic_calibration(
    num_features: int,
    num_samples: int,
    seed: int = 7,
    dtype: torch.dtype = torch.float32,
) -> CalibrationData:
    torch.manual_seed(seed)
    activations = torch.randn(num_features, num_samples, dtype=dtype)
    metadata = {
        "source": "synthetic",
        "num_features": num_features,
        "num_samples": num_samples,
        "seed": seed,
    }
    return CalibrationData(activations=activations, metadata=metadata)
