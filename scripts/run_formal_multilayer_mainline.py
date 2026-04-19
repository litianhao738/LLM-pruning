import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.formal_runs import (
    FORMAL_CALIBRATION_DATASET_CONFIG,
    FORMAL_CALIBRATION_DATASET_NAME,
    FORMAL_CALIBRATION_MAX_TEXTS,
    FORMAL_CALIBRATION_MIN_CHARS,
    FORMAL_CALIBRATION_SOURCE,
    FORMAL_CALIBRATION_SPLIT,
    FORMAL_CALIBRATION_TEXT_KEY,
    FORMAL_MODEL_NAME,
    FORMAL_ML_ADAPTIVE_R_MAX,
    FORMAL_ML_ADAPTIVE_R_MIN,
    FORMAL_ML_BATCH_SIZE,
    FORMAL_ML_DEVICE,
    FORMAL_ML_EVAL_TEXTS,
    FORMAL_ML_FINETUNE_STEPS,
    FORMAL_ML_FINETUNE_TEXTS,
    FORMAL_ML_FT_OUTPUT_DIR,
    FORMAL_ML_GM_MOMENTUM_BETA,
    FORMAL_ML_GM_R_MAX,
    FORMAL_ML_GM_R_MIN,
    FORMAL_ML_GRAD_CLIP,
    FORMAL_ML_ITERS,
    FORMAL_ML_LAYER_NAMES,
    FORMAL_ML_LEARNING_RATE,
    FORMAL_ML_MAX_LENGTH,
    FORMAL_ML_METHODS,
    FORMAL_ML_MOMENTUM_BETA,
    FORMAL_ML_OUTPUT_DIR,
    FORMAL_ML_R_MAX,
    FORMAL_ML_R_MIN,
    FORMAL_ML_SEARCH_STEPS,
    FORMAL_ML_SEED,
    FORMAL_ML_SPARSITY_TOL,
    FORMAL_ML_TARGET_SPARSITY,
    FORMAL_ML_WEIGHT_DECAY,
)
from utils.io_utils import save_csv_rows, save_json
from utils.single_layer_utils import method_settings, parse_methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal multi-layer mainline across one or more methods, "
            "using the tuned nopad configuration for the current project."
        )
    )
    parser.add_argument("--device", type=str, default=FORMAL_ML_DEVICE)
    parser.add_argument("--methods", type=str, default=FORMAL_ML_METHODS)
    parser.add_argument("--target-sparsity", type=float, default=FORMAL_ML_TARGET_SPARSITY)
    parser.add_argument("--iters", type=int, default=FORMAL_ML_ITERS)
    parser.add_argument("--search-steps", type=int, default=FORMAL_ML_SEARCH_STEPS)
    parser.add_argument("--sparsity-tol", type=float, default=FORMAL_ML_SPARSITY_TOL)
    parser.add_argument("--r-min", type=float, default=FORMAL_ML_R_MIN)
    parser.add_argument("--r-max", type=float, default=FORMAL_ML_R_MAX)
    parser.add_argument("--momentum-beta", type=float, default=FORMAL_ML_MOMENTUM_BETA)
    parser.add_argument("--adaptive-r-min", type=float, default=FORMAL_ML_ADAPTIVE_R_MIN)
    parser.add_argument("--adaptive-r-max", type=float, default=FORMAL_ML_ADAPTIVE_R_MAX)
    parser.add_argument("--gradient-r-min", type=float, default=FORMAL_ML_GM_R_MIN)
    parser.add_argument("--gradient-r-max", type=float, default=FORMAL_ML_GM_R_MAX)
    parser.add_argument("--gradient-momentum-beta", type=float, default=FORMAL_ML_GM_MOMENTUM_BETA)
    parser.add_argument("--max-texts", type=int, default=FORMAL_CALIBRATION_MAX_TEXTS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_ML_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=FORMAL_ML_MAX_LENGTH)
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-texts", type=int, default=FORMAL_ML_FINETUNE_TEXTS)
    parser.add_argument("--eval-texts", type=int, default=FORMAL_ML_EVAL_TEXTS)
    parser.add_argument("--seed", type=int, default=FORMAL_ML_SEED)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--disable-progress", action="store_true")
    return parser


def _resolve_output_dir(raw: str | None, *, finetune_steps: int) -> Path:
    if raw is not None:
        return Path(raw)
    if finetune_steps > 0:
        return FORMAL_ML_FT_OUTPUT_DIR
    return FORMAL_ML_OUTPUT_DIR


def _run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_stage(rows: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("stage") == stage:
            return row
    return None


def _last_stage(rows: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    matches = [row for row in rows if row.get("stage") == stage]
    if not matches:
        return None
    return matches[-1]


def _build_summary_row(
    *,
    method: str,
    settings: dict[str, float | None],
    method_output_dir: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    initial_metrics = report.get("initial_metrics", {})
    layer_summaries = report.get("layer_summaries", [])
    evaluation_history = report.get("evaluation_history", [])

    final_layer = layer_summaries[-1] if layer_summaries else {}
    after_pruning = _last_stage(evaluation_history, "after_layer") or {}
    after_finetuning = _find_stage(evaluation_history, "after_finetuning") or {}

    return {
        "method": method,
        "target_sparsity": report.get("setup", {}).get("target_sparsity"),
        "num_layers": len(layer_summaries),
        "final_layer_name": final_layer.get("layer_name"),
        "final_actual_sparsity": final_layer.get("actual_sparsity"),
        "final_target_gap": final_layer.get("target_gap"),
        "final_reconstruction_error": final_layer.get("reconstruction_error"),
        "final_selected_lambda": final_layer.get("selected_lambda"),
        "before_average_nll": initial_metrics.get("average_nll"),
        "before_perplexity": initial_metrics.get("perplexity"),
        "after_pruning_average_nll": after_pruning.get("average_nll"),
        "after_pruning_perplexity": after_pruning.get("perplexity"),
        "after_finetuning_average_nll": after_finetuning.get("average_nll"),
        "after_finetuning_perplexity": after_finetuning.get("perplexity"),
        "r_min_used": settings.get("r_min"),
        "r_max_used": settings.get("r_max"),
        "momentum_beta_used": settings.get("momentum_beta"),
        "output_dir": str(method_output_dir),
        "report_path": str(method_output_dir / "report.json"),
        "layer_summary_path": str(method_output_dir / "layer_summary.csv"),
        "model_eval_path": str(method_output_dir / "model_eval.csv"),
    }


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_methods(args.methods)
    output_dir = _resolve_output_dir(args.output_dir, finetune_steps=args.finetune_steps)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

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
        method_output_dir = output_dir / method

        command = [
            sys.executable,
            "scripts/run_multilayer_pruning.py",
            "--model-name",
            FORMAL_MODEL_NAME,
            "--layer-names",
            ",".join(FORMAL_ML_LAYER_NAMES),
            "--method",
            method,
            "--target-sparsity",
            str(args.target_sparsity),
            "--iters",
            str(args.iters),
            "--search-steps",
            str(args.search_steps),
            "--sparsity-tol",
            str(args.sparsity_tol),
            "--device",
            args.device,
            "--max-length",
            str(args.max_length),
            "--batch-size",
            str(args.batch_size),
            "--calibration-source",
            FORMAL_CALIBRATION_SOURCE,
            "--calibration-dataset-name",
            FORMAL_CALIBRATION_DATASET_NAME,
            "--calibration-dataset-config",
            FORMAL_CALIBRATION_DATASET_CONFIG,
            "--calibration-split",
            FORMAL_CALIBRATION_SPLIT,
            "--calibration-text-key",
            FORMAL_CALIBRATION_TEXT_KEY,
            "--calibration-max-texts",
            str(args.max_texts),
            "--calibration-min-chars",
            str(FORMAL_CALIBRATION_MIN_CHARS),
            "--eval-texts",
            str(args.eval_texts),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(method_output_dir),
        ]
        if settings.get("r_min") is not None:
            command.extend(["--r-min", str(settings["r_min"])])
        if settings.get("r_max") is not None:
            command.extend(["--r-max", str(settings["r_max"])])
        if settings.get("momentum_beta") is not None:
            command.extend(["--momentum-beta", str(settings["momentum_beta"])])
        if args.finetune_steps > 0:
            command.extend(
                [
                    "--finetune-steps",
                    str(args.finetune_steps),
                    "--finetune-texts",
                    str(args.finetune_texts),
                    "--learning-rate",
                    str(FORMAL_ML_LEARNING_RATE),
                    "--weight-decay",
                    str(FORMAL_ML_WEIGHT_DECAY),
                    "--grad-clip",
                    str(FORMAL_ML_GRAD_CLIP),
                ]
            )
        if args.disable_progress:
            command.append("--disable-progress")

        _run_command(command)

        report_path = method_output_dir / "report.json"
        report = _load_json(report_path)
        summary_row = _build_summary_row(
            method=method,
            settings=settings,
            method_output_dir=method_output_dir,
            report=report,
        )
        summary_rows.append(summary_row)
        run_entries.append(
            {
                "method": method,
                "settings": settings,
                "output_dir": str(method_output_dir),
                "report": report,
                "summary": summary_row,
            }
        )

    summary_csv = save_csv_rows(summary_rows, output_dir / "summary.csv")
    report_json = save_json(
        {
            "setup": {
                "model_name": FORMAL_MODEL_NAME,
                "layer_names": FORMAL_ML_LAYER_NAMES,
                "methods": methods,
                "target_sparsity": float(args.target_sparsity),
                "iters": int(args.iters),
                "search_steps": int(args.search_steps),
                "sparsity_tol": float(args.sparsity_tol),
                "device": args.device,
                "max_texts": int(args.max_texts),
                "batch_size": int(args.batch_size),
                "max_length": int(args.max_length),
                "finetune_steps": int(args.finetune_steps),
                "finetune_texts": int(args.finetune_texts),
                "eval_texts": int(args.eval_texts),
                "seed": int(args.seed),
            },
            "runs": run_entries,
        },
        output_dir / "report.json",
    )

    print("[setup]")
    print(f"methods: {', '.join(methods)}")
    print(f"output_dir: {output_dir}")
    print(f"finetune_steps: {args.finetune_steps}")

    print("\n[summary]")
    for row in summary_rows:
        print(f"[{row['method']}]")
        print(f"after_pruning_perplexity: {float(row['after_pruning_perplexity']):.6f}")
        if row.get("after_finetuning_perplexity") is not None:
            print(f"after_finetuning_perplexity: {float(row['after_finetuning_perplexity']):.6f}")
        print(f"final_actual_sparsity: {float(row['final_actual_sparsity']):.6f}")

    print("\n[saved]")
    print(f"summary_csv: {summary_csv}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
