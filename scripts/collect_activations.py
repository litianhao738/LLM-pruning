import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.formal_mainline import (
    FORMAL_BUNDLE_PATH,
    FORMAL_CALIBRATION_BATCH_SIZE,
    FORMAL_CALIBRATION_DATASET_CONFIG,
    FORMAL_CALIBRATION_DATASET_NAME,
    FORMAL_CALIBRATION_MAX_LENGTH,
    FORMAL_CALIBRATION_MAX_TEXTS,
    FORMAL_CALIBRATION_MIN_CHARS,
    FORMAL_CALIBRATION_SEED,
    FORMAL_CALIBRATION_SOURCE,
    FORMAL_CALIBRATION_SPLIT,
    FORMAL_CALIBRATION_TEXT_KEY,
    FORMAL_LAYER_NAME,
    FORMAL_MODEL_NAME,
)
from data.calibration import load_calibration_text_corpus
from models.hooks import (
    ActivationHook,
    choose_default_prunable_module,
    extract_weight_matrix,
    list_supported_prunable_modules,
    resolve_module,
)
from utils.io_utils import save_tensor_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect real calibration activations from a Hugging Face model layer."
    )
    parser.add_argument("--model-name", type=str, default=FORMAL_MODEL_NAME)
    parser.add_argument("--layer-name", type=str, default=FORMAL_LAYER_NAME)
    parser.add_argument("--max-length", type=int, default=FORMAL_CALIBRATION_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=FORMAL_CALIBRATION_BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-path", type=str, default=str(FORMAL_BUNDLE_PATH))
    parser.add_argument(
        "--calibration-source",
        type=str,
        default=FORMAL_CALIBRATION_SOURCE,
        help="One of: wikitext103, hf_dataset, default_texts",
    )
    parser.add_argument("--calibration-dataset-name", type=str, default=FORMAL_CALIBRATION_DATASET_NAME)
    parser.add_argument(
        "--calibration-dataset-config",
        type=str,
        default=FORMAL_CALIBRATION_DATASET_CONFIG,
    )
    parser.add_argument("--calibration-split", type=str, default=FORMAL_CALIBRATION_SPLIT)
    parser.add_argument("--calibration-text-key", type=str, default=FORMAL_CALIBRATION_TEXT_KEY)
    parser.add_argument("--calibration-max-texts", type=int, default=FORMAL_CALIBRATION_MAX_TEXTS)
    parser.add_argument("--calibration-min-chars", type=int, default=FORMAL_CALIBRATION_MIN_CHARS)
    parser.add_argument("--calibration-seed", type=int, default=FORMAL_CALIBRATION_SEED)
    parser.add_argument(
        "--disable-calibration-shuffle",
        action="store_true",
        help="Disable dataset shuffling before selecting calibration texts.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars during activation collection.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print all supported prunable layer names and exit.",
    )
    return parser


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    args = build_parser().parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()
    model.to(args.device)

    supported_layers = list_supported_prunable_modules(model)
    if args.list_layers:
        print("Supported prunable layers:")
        for name in supported_layers:
            print(name)
        return

    if not supported_layers:
        raise ValueError("No supported prunable modules were found in the loaded model")

    layer_name = args.layer_name or choose_default_prunable_module(model)
    module = resolve_module(model, layer_name)
    hook = ActivationHook(module=module, move_to_cpu=True, flatten_batch=True)

    corpus = load_calibration_text_corpus(
        source=args.calibration_source,
        dataset_name=args.calibration_dataset_name,
        dataset_config=args.calibration_dataset_config,
        split=args.calibration_split,
        text_key=args.calibration_text_key,
        max_texts=args.calibration_max_texts,
        min_chars=args.calibration_min_chars,
        seed=args.calibration_seed,
        shuffle=not args.disable_calibration_shuffle,
    )
    texts = corpus.texts
    text_batches = batched(texts, args.batch_size)
    with torch.no_grad():
        batch_iterator = tqdm(
            text_batches,
            desc="Collecting activations",
            disable=args.disable_progress,
        )
        for text_batch in batch_iterator:
            encoded = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            model(**encoded)

    X = hook.stacked_inputs().to(dtype=torch.float32)
    W = extract_weight_matrix(module)
    hook.close()

    bundle = {
        "W": W,
        "X": X,
        "metadata": {
            "source": "huggingface",
            "model_name": args.model_name,
            "layer_name": layer_name,
            "module_type": module.__class__.__name__,
            "input_dim": int(W.shape[1]),
            "output_dim": int(W.shape[0]),
            "num_samples": int(X.shape[1]),
            "num_texts": len(texts),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            **corpus.metadata,
        },
    }
    save_tensor_bundle(bundle, args.output_path)

    print("Collected calibration activations.")
    print(f"model: {args.model_name}")
    print(f"layer: {layer_name}")
    print(f"module_type: {module.__class__.__name__}")
    print(f"W shape: {tuple(W.shape)}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"calibration_source: {corpus.metadata.get('calibration_source')}")
    if "dataset_name" in corpus.metadata:
        print(f"dataset_name: {corpus.metadata.get('dataset_name')}")
        print(f"dataset_config: {corpus.metadata.get('dataset_config')}")
        print(f"dataset_split: {corpus.metadata.get('dataset_split')}")
    print(f"num_texts: {len(texts)}")
    print(f"saved to: {args.output_path}")


if __name__ == "__main__":
    main()
