import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.perplexity import evaluate_perplexity_on_texts
from models.hooks import apply_weight_matrix, resolve_module
from utils.finetune_masks import apply_parameter_masks, build_module_weight_masks, mask_parameter_grads
from utils.io_utils import load_tensor_bundle, save_json
from utils.single_layer_utils import (
    build_prune_result,
    method_settings,
    select_finetune_and_eval_texts,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune one layer, write the pruned weights back into the model, and fine-tune only that layer."
    )
    parser.add_argument("--model-name", type=str, default="distilgpt2")
    parser.add_argument("--layer-name", type=str, required=True)
    parser.add_argument("--bundle-path", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "magnitude",
            "fista",
            "adaptive_fista",
            "gradient_momentum_fista",
            "gradient_momentum_fista_original",
        ],
        default="gradient_momentum_fista",
    )
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--search-steps", type=int, default=12)
    parser.add_argument("--sparsity-tol", type=float, default=0.02)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--momentum-beta", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--finetune-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--finetune-texts", type=int, default=32)
    parser.add_argument("--eval-texts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--calibration-source", type=str, default="wikitext103")
    parser.add_argument("--calibration-dataset-name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--calibration-dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-text-key", type=str, default="text")
    parser.add_argument("--calibration-min-chars", type=int, default=20)
    parser.add_argument("--disable-calibration-shuffle", action="store_true")
    return parser


def _resolve_output_path(raw: str | None, *, method: str, target_sparsity: float) -> Path:
    if raw is not None:
        return Path(raw)
    return Path("artifacts") / f"prune_then_finetune_{method}_s{target_sparsity:.2f}.json"

def _freeze_except_module(model: torch.nn.Module, target_module: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in target_module.parameters():
        parameter.requires_grad = True


def _run_finetuning(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    parameter_masks: list[tuple[torch.nn.Parameter, torch.Tensor]],
    show_progress: bool,
) -> list[dict[str, float]]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for fine-tuning")

    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    model.train()
    history: list[dict[str, float]] = []

    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Fine-tuning", leave=False)

    for step in iterator:
        batch_texts = [
            texts[(step * batch_size + offset) % len(texts)]
            for offset in range(batch_size)
        ]
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

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**encoded, labels=labels)
        loss = outputs.loss
        loss.backward()
        mask_parameter_grads(parameter_masks)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        apply_parameter_masks(parameter_masks)

        history.append(
            {
                "step": float(step),
                "train_loss": float(loss.item()),
            }
        )

    return history


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    pruning_device = device if device.type == "cuda" else torch.device("cpu")
    output_path = _resolve_output_path(
        args.output_path,
        method=args.method,
        target_sparsity=args.target_sparsity,
    )

    bundle = load_tensor_bundle(args.bundle_path)
    W = bundle["W"].to(device=pruning_device, dtype=torch.float32)
    X = bundle["X"].to(device=pruning_device, dtype=torch.float32)
    bundle_metadata = bundle.get("metadata", {})

    prune_info = build_prune_result(
        method=args.method,
        target_sparsity=args.target_sparsity,
        W=W,
        X=X,
        num_iters=args.iters,
        search_steps=args.search_steps,
        sparsity_tol=args.sparsity_tol,
        show_progress=not args.disable_progress,
        progress_desc=f"{args.method} lambda search",
        settings=method_settings(
            args.method,
            default_r_min=args.r_min,
            default_r_max=args.r_max,
            default_momentum_beta=args.momentum_beta,
        ),
    )
    prune_result = prune_info["prune_result"]

    finetune_texts, eval_texts, corpus_metadata = select_finetune_and_eval_texts(
        source=args.calibration_source,
        dataset_name=args.calibration_dataset_name,
        dataset_config=args.calibration_dataset_config,
        split=args.calibration_split,
        text_key=args.calibration_text_key,
        finetune_texts=args.finetune_texts,
        eval_texts=args.eval_texts,
        min_chars=args.calibration_min_chars,
        seed=args.seed,
        shuffle=not args.disable_calibration_shuffle,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    pruned_model = copy.deepcopy(base_model)
    pruned_model.to(device)

    target_module = resolve_module(pruned_model, args.layer_name)
    apply_weight_matrix(target_module, prune_result.U)
    parameter_masks = build_module_weight_masks([target_module])
    _freeze_except_module(pruned_model, target_module)

    before_metrics = evaluate_perplexity_on_texts(
        model=base_model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        show_progress=not args.disable_progress,
        progress_desc="Eval before pruning",
    )
    after_prune_metrics = evaluate_perplexity_on_texts(
        model=pruned_model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        show_progress=not args.disable_progress,
        progress_desc="Eval after pruning",
    )

    finetune_history = _run_finetuning(
        model=pruned_model,
        tokenizer=tokenizer,
        texts=finetune_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_steps=args.finetune_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        parameter_masks=parameter_masks,
        show_progress=not args.disable_progress,
    )

    after_finetune_metrics = evaluate_perplexity_on_texts(
        model=pruned_model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        show_progress=not args.disable_progress,
        progress_desc="Eval after fine-tuning",
    )

    report = {
        "setup": {
            "model_name": args.model_name,
            "layer_name": args.layer_name,
            "bundle_path": args.bundle_path,
            "method": args.method,
            "target_sparsity": float(args.target_sparsity),
            "iters": int(args.iters),
            "r_min": float(args.r_min),
            "r_max": float(args.r_max),
            "momentum_beta": float(args.momentum_beta),
            "device": str(device),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "finetune_steps": int(args.finetune_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "preserve_pruning_mask": True,
            "finetune_texts": len(finetune_texts),
            "eval_texts": len(eval_texts),
            "seed": int(args.seed),
        },
        "bundle_metadata": bundle_metadata,
        "corpus_metadata": corpus_metadata,
        "pruning": {
            "selected_lambda": prune_info["selected_lambda"],
            "search": prune_info["search"],
            "stats": prune_result.stats,
        },
        "metrics": {
            "before_pruning": before_metrics,
            "after_pruning": after_prune_metrics,
            "after_finetuning": after_finetune_metrics,
        },
        "finetune_history": finetune_history,
    }

    save_json(report, output_path)

    print("[setup]")
    print(f"method: {args.method}")
    print(f"model_name: {args.model_name}")
    print(f"layer_name: {args.layer_name}")
    print(f"target_sparsity: {args.target_sparsity:.6f}")
    if prune_info["selected_lambda"] is not None:
        print(f"selected_lambda: {prune_info['selected_lambda']:.6f}")
    print(f"finetune_steps: {args.finetune_steps}")
    print(f"finetune_texts: {len(finetune_texts)}")
    print(f"eval_texts: {len(eval_texts)}")

    print("\n[metrics]")
    for stage, metrics in report["metrics"].items():
        print(stage)
        print(f"  average_nll: {metrics['average_nll']:.6f}")
        print(f"  perplexity: {metrics['perplexity']:.6f}")

    print("\n[saved]")
    print(f"report_json: {output_path}")


if __name__ == "__main__":
    main()
