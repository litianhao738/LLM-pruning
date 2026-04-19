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
from utils.io_utils import load_tensor_bundle, save_csv_rows, save_json
from utils.single_layer_utils import (
    build_prune_result,
    load_prune_cache,
    method_settings,
    parse_methods,
    select_finetune_and_eval_texts,
    set_seed,
)


DEFAULT_BUNDLE_PATH = Path("artifacts") / "distilgpt2_h0_attn_cproj_wikitext128_bundle_nopad.pt"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "single_layer_prune_then_finetune_compare_nopad"
DEFAULT_PRUNE_CACHE_DIR = Path("artifacts") / "single_layer_mainline"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare single-layer pruning methods after pruning and after fine-tuning "
            "on the same bundle, finetune texts, and evaluation texts."
        )
    )
    parser.add_argument("--model-name", type=str, default="distilgpt2")
    parser.add_argument("--layer-name", type=str, default="transformer.h.0.attn.c_proj")
    parser.add_argument("--bundle-path", type=str, default=str(DEFAULT_BUNDLE_PATH))
    parser.add_argument(
        "--methods",
        type=str,
        default="magnitude,fista,adaptive_fista,gradient_momentum_fista",
        help="Comma-separated methods to compare.",
    )
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--search-steps", type=int, default=12)
    parser.add_argument("--sparsity-tol", type=float, default=0.01)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--momentum-beta", type=float, default=0.5)
    parser.add_argument("--adaptive-r-min", type=float, default=None)
    parser.add_argument("--adaptive-r-max", type=float, default=None)
    parser.add_argument("--gradient-r-min", type=float, default=None)
    parser.add_argument("--gradient-r-max", type=float, default=None)
    parser.add_argument("--gradient-momentum-beta", type=float, default=None)
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
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--prune-cache-dir",
        type=str,
        default=str(DEFAULT_PRUNE_CACHE_DIR),
        help=(
            "Directory containing per-method single-layer pruning caches. "
            "If a matching cache is found, fine-tuning reuses that exact pruning result."
        ),
    )
    parser.add_argument("--calibration-source", type=str, default="wikitext103")
    parser.add_argument("--calibration-dataset-name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--calibration-dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-text-key", type=str, default="text")
    parser.add_argument("--calibration-min-chars", type=int, default=20)
    parser.add_argument("--disable-calibration-shuffle", action="store_true")
    return parser

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
    progress_desc: str,
) -> list[dict[str, float]]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for fine-tuning")
    if not texts:
        raise ValueError("Fine-tuning texts are required")

    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    model.train()
    history: list[dict[str, float]] = []

    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc, leave=False)

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
    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prune_cache_dir = Path(args.prune_cache_dir)

    set_seed(args.seed)
    device = torch.device(args.device)
    pruning_device = device if device.type == "cuda" else torch.device("cpu")

    bundle = load_tensor_bundle(args.bundle_path)
    W = bundle["W"].to(device=pruning_device, dtype=torch.float32)
    X = bundle["X"].to(device=pruning_device, dtype=torch.float32)
    bundle_metadata = bundle.get("metadata", {})

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

    summary_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    finetune_history_rows: list[dict[str, Any]] = []

    for method in methods:
        settings = method_settings(
            method,
            default_r_min=args.r_min,
            default_r_max=args.r_max,
            default_momentum_beta=args.momentum_beta,
            adaptive_r_min=args.adaptive_r_min,
            adaptive_r_max=args.adaptive_r_max,
            gradient_r_min=args.gradient_r_min,
            gradient_r_max=args.gradient_r_max,
            gradient_momentum_beta=args.gradient_momentum_beta,
        )
        prune_info = load_prune_cache(
            cache_root=prune_cache_dir,
            method=method,
            bundle_path=args.bundle_path,
            target_sparsity=args.target_sparsity,
            num_iters=args.iters,
            search_steps=args.search_steps,
            sparsity_tol=args.sparsity_tol,
            settings=settings,
        )
        used_pruning_cache = prune_info is not None
        if prune_info is None:
            prune_info = build_prune_result(
                method=method,
                target_sparsity=args.target_sparsity,
                W=W,
                X=X,
                num_iters=args.iters,
                search_steps=args.search_steps,
                sparsity_tol=args.sparsity_tol,
                show_progress=not args.disable_progress,
                progress_desc=f"{method} lambda search",
                settings=settings,
            )
        prune_result = prune_info["prune_result"]

        model = copy.deepcopy(base_model).to(device)
        target_module = resolve_module(model, args.layer_name)
        apply_weight_matrix(target_module, prune_result.U.to(dtype=torch.float32))
        parameter_masks = build_module_weight_masks([target_module])
        _freeze_except_module(model, target_module)

        after_prune_metrics = evaluate_perplexity_on_texts(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            show_progress=not args.disable_progress,
            progress_desc=f"Eval after pruning ({method})",
        )

        set_seed(args.seed)
        finetune_history = _run_finetuning(
            model=model,
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
            progress_desc=f"Fine-tuning ({method})",
        )

        after_finetune_metrics = evaluate_perplexity_on_texts(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            show_progress=not args.disable_progress,
            progress_desc=f"Eval after fine-tuning ({method})",
        )

        last_history = prune_result.history[-1] if prune_result.history else {}
        summary_rows.append(
            {
                "method": method,
                "target_sparsity": float(args.target_sparsity),
                "actual_sparsity": float(prune_result.stats.get("actual_sparsity", 0.0)),
                "target_gap": abs(
                    float(prune_result.stats.get("actual_sparsity", 0.0)) - float(args.target_sparsity)
                ),
                "selected_lambda": prune_info["selected_lambda"],
                "num_iters": int(prune_result.stats.get("num_iters", 0)),
                "r_min_used": settings["r_min"],
                "r_max_used": settings["r_max"],
                "momentum_beta_used": settings["momentum_beta"],
                "before_average_nll": float(before_metrics["average_nll"]),
                "before_perplexity": float(before_metrics["perplexity"]),
                "after_pruning_average_nll": float(after_prune_metrics["average_nll"]),
                "after_pruning_perplexity": float(after_prune_metrics["perplexity"]),
                "after_finetuning_average_nll": float(after_finetune_metrics["average_nll"]),
                "after_finetuning_perplexity": float(after_finetune_metrics["perplexity"]),
                "delta_pruning_perplexity": float(after_prune_metrics["perplexity"] - before_metrics["perplexity"]),
                "delta_finetuning_perplexity": float(after_finetune_metrics["perplexity"] - before_metrics["perplexity"]),
                "delta_recovery_perplexity": float(after_finetune_metrics["perplexity"] - after_prune_metrics["perplexity"]),
                "pruning_reconstruction_error": prune_result.stats.get("reconstruction_error"),
                # Backward-compatible alias. Reports should prefer pruning_reconstruction_error.
                "reconstruction_error": prune_result.stats.get("reconstruction_error"),
                "objective": prune_result.stats.get("objective"),
                "last_diff_norm": last_history.get("diff_norm"),
                "search_best_gap": (
                    None if prune_info["search"] is None else prune_info["search"]["best_gap"]
                ),
                "search_num_trials": (
                    0 if prune_info["search"] is None else prune_info["search"]["num_trials"]
                ),
                "search_terminated_reason": (
                    "not_applicable"
                    if prune_info["search"] is None
                    else prune_info["search"]["terminated_reason"]
                ),
                "used_pruning_cache": used_pruning_cache,
                "prune_cache_path": (
                    str(prune_info["cache_path"]) if used_pruning_cache else None
                ),
            }
        )

        for entry in finetune_history:
            row = {
                "method": method,
                "target_sparsity": float(args.target_sparsity),
                "r_min_used": settings["r_min"],
                "r_max_used": settings["r_max"],
                "momentum_beta_used": settings["momentum_beta"],
            }
            row.update(entry)
            finetune_history_rows.append(row)

        if prune_info["search"] is not None:
            for trial in prune_info["search"]["trials"]:
                row = {
                    "method": method,
                    "target_sparsity": float(args.target_sparsity),
                    "r_min_used": settings["r_min"],
                    "r_max_used": settings["r_max"],
                    "momentum_beta_used": settings["momentum_beta"],
                }
                row.update(trial)
                search_rows.append(row)

        del target_module
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "setup": {
            "model_name": args.model_name,
            "layer_name": args.layer_name,
            "bundle_path": args.bundle_path,
            "methods": methods,
            "target_sparsity": float(args.target_sparsity),
            "iters": int(args.iters),
            "search_steps": int(args.search_steps),
            "sparsity_tol": float(args.sparsity_tol),
            "r_min": float(args.r_min),
            "r_max": float(args.r_max),
            "momentum_beta": float(args.momentum_beta),
            "adaptive_r_min": args.adaptive_r_min,
            "adaptive_r_max": args.adaptive_r_max,
            "gradient_r_min": args.gradient_r_min,
            "gradient_r_max": args.gradient_r_max,
            "gradient_momentum_beta": args.gradient_momentum_beta,
            "device": str(device),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "finetune_steps": int(args.finetune_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "preserve_pruning_mask": True,
            "reconstruction_error_definition": "pruning_stage_only",
            "prune_cache_dir": str(prune_cache_dir),
            "finetune_texts": len(finetune_texts),
            "eval_texts": len(eval_texts),
            "seed": int(args.seed),
        },
        "bundle_metadata": bundle_metadata,
        "corpus_metadata": corpus_metadata,
        "summary_rows": summary_rows,
        "search_rows": search_rows,
        "finetune_history": finetune_history_rows,
    }

    summary_csv = save_csv_rows(summary_rows, output_dir / "summary.csv")
    search_csv = None
    if search_rows:
        search_csv = save_csv_rows(search_rows, output_dir / "search_trace.csv")
    finetune_csv = save_csv_rows(finetune_history_rows, output_dir / "finetune_history.csv")
    report_json = save_json(report, output_dir / "report.json")

    print("[setup]")
    print(f"model_name: {args.model_name}")
    print(f"layer_name: {args.layer_name}")
    print(f"methods: {', '.join(methods)}")
    print(f"target_sparsity: {args.target_sparsity:.6f}")
    print(f"finetune_texts: {len(finetune_texts)}")
    print(f"eval_texts: {len(eval_texts)}")

    print("\n[before_pruning]")
    print(f"average_nll: {before_metrics['average_nll']:.6f}")
    print(f"perplexity: {before_metrics['perplexity']:.6f}")

    print("\n[summary]")
    for row in summary_rows:
        print(f"[{row['method']}]")
        print(f"actual_sparsity: {float(row['actual_sparsity']):.6f}")
        print(f"after_pruning_perplexity: {float(row['after_pruning_perplexity']):.6f}")
        print(f"after_finetuning_perplexity: {float(row['after_finetuning_perplexity']):.6f}")
        if row["selected_lambda"] is not None:
            print(f"selected_lambda: {float(row['selected_lambda']):.6f}")

    print("\n[saved]")
    print(f"report_json: {report_json}")
    print(f"summary_csv: {summary_csv}")
    print(f"finetune_history_csv: {finetune_csv}")
    if search_csv is not None:
        print(f"search_trace_csv: {search_csv}")


if __name__ == "__main__":
    main()
