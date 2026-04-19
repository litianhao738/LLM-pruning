import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.perplexity import evaluate_perplexity_on_texts
from models.hooks import apply_weight_matrix, resolve_module
from utils.io_utils import load_tensor_bundle, save_csv_rows, save_json
from utils.single_layer_utils import (
    build_prune_result,
    method_settings,
    parse_methods,
    select_eval_texts,
    set_seed,
)


DEFAULT_BUNDLE_PATH = Path("artifacts") / "distilgpt2_h0_attn_cproj_wikitext128_bundle_nopad.pt"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "single_layer_perplexity_compare_nopad"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare single-layer pruning methods on after-pruning perplexity "
            "at matched target sparsity, without fine-tuning."
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
    parser.add_argument("--lambda-low", type=float, default=1e-6)
    parser.add_argument("--lambda-high", type=float, default=1.0)
    parser.add_argument("--bracket-scale", type=float, default=10.0)
    parser.add_argument("--max-bracket-steps", type=int, default=12)
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
    parser.add_argument("--eval-start-index", type=int, default=32)
    parser.add_argument("--eval-texts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calibration-source", type=str, default="wikitext103")
    parser.add_argument("--calibration-dataset-name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--calibration-dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-text-key", type=str, default="text")
    parser.add_argument("--calibration-min-chars", type=int, default=20)
    parser.add_argument("--disable-calibration-shuffle", action="store_true")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    pruning_device = device if device.type == "cuda" else torch.device("cpu")
    bundle = load_tensor_bundle(args.bundle_path)
    W = bundle["W"].to(device=pruning_device, dtype=torch.float64)
    X = bundle["X"].to(device=pruning_device, dtype=torch.float64)
    bundle_metadata = bundle.get("metadata", {})

    eval_texts, corpus_metadata = select_eval_texts(
        source=args.calibration_source,
        dataset_name=args.calibration_dataset_name,
        dataset_config=args.calibration_dataset_config,
        split=args.calibration_split,
        text_key=args.calibration_text_key,
        eval_start_index=args.eval_start_index,
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
            lambda_low=args.lambda_low,
            lambda_high=args.lambda_high,
            bracket_scale=args.bracket_scale,
            max_bracket_steps=args.max_bracket_steps,
        )
        prune_result = prune_info["prune_result"]

        model = copy.deepcopy(base_model)
        target_module = resolve_module(model, args.layer_name)
        apply_weight_matrix(target_module, prune_result.U.to(dtype=torch.float32))

        after_metrics = evaluate_perplexity_on_texts(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            show_progress=not args.disable_progress,
            progress_desc=f"Eval after {method}",
        )

        last_history = prune_result.history[-1] if prune_result.history else {}
        summary_row = {
            "method": method,
            "target_sparsity": float(args.target_sparsity),
            "actual_sparsity": float(prune_result.stats.get("actual_sparsity", 0.0)),
            "target_gap": abs(
                float(prune_result.stats.get("actual_sparsity", 0.0)) - float(args.target_sparsity)
            ),
            "selected_lambda": prune_info["selected_lambda"],
            "num_iters": int(prune_result.stats.get("num_iters", 0)),
            "before_average_nll": float(before_metrics["average_nll"]),
            "before_perplexity": float(before_metrics["perplexity"]),
            "after_pruning_average_nll": float(after_metrics["average_nll"]),
            "after_pruning_perplexity": float(after_metrics["perplexity"]),
            "delta_average_nll": float(after_metrics["average_nll"] - before_metrics["average_nll"]),
            "delta_perplexity": float(after_metrics["perplexity"] - before_metrics["perplexity"]),
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
        }
        summary_rows.append(summary_row)

        if prune_info["search"] is not None:
            for trial in prune_info["search"]["trials"]:
                row = {
                    "method": method,
                    "target_sparsity": float(args.target_sparsity),
                }
                row.update(trial)
                search_rows.append(row)

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
            "lambda_low": float(args.lambda_low),
            "lambda_high": float(args.lambda_high),
            "bracket_scale": float(args.bracket_scale),
            "max_bracket_steps": int(args.max_bracket_steps),
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
            "eval_start_index": int(args.eval_start_index),
            "eval_texts": int(args.eval_texts),
            "seed": int(args.seed),
        },
        "bundle_metadata": bundle_metadata,
        "corpus_metadata": corpus_metadata,
        "summary_rows": summary_rows,
        "search_rows": search_rows,
    }

    report_json = save_json(report, output_dir / "report.json")
    summary_csv = save_csv_rows(summary_rows, output_dir / "summary.csv")
    search_csv = None
    if search_rows:
        search_csv = save_csv_rows(search_rows, output_dir / "search_trace.csv")

    print("[setup]")
    print(f"model_name: {args.model_name}")
    print(f"layer_name: {args.layer_name}")
    print(f"methods: {', '.join(methods)}")
    print(f"target_sparsity: {args.target_sparsity:.6f}")
    print(f"eval_texts: {args.eval_texts}")

    print("\n[before_pruning]")
    print(f"average_nll: {before_metrics['average_nll']:.6f}")
    print(f"perplexity: {before_metrics['perplexity']:.6f}")

    print("\n[summary]")
    for row in summary_rows:
        print(f"[{row['method']}]")
        print(f"actual_sparsity: {float(row['actual_sparsity']):.6f}")
        print(f"after_pruning_perplexity: {float(row['after_pruning_perplexity']):.6f}")
        print(f"delta_perplexity: {float(row['delta_perplexity']):.6f}")
        if row["selected_lambda"] is not None:
            print(f"selected_lambda: {float(row['selected_lambda']):.6f}")

    print("\n[saved]")
    print(f"report_json: {report_json}")
    print(f"summary_csv: {summary_csv}")
    if search_csv is not None:
        print(f"search_trace_csv: {search_csv}")


if __name__ == "__main__":
    main()
