import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.calibration import make_synthetic_calibration
from eval.reconstruction import summarize_pruning_result
from pruning import (
    AdaptiveThresholdFISTAPruner,
    FISTAPruner,
    GradientAwareMomentumFISTAPruner,
    MagnitudePruner,
    find_lambda_for_target_sparsity,
)
from utils.io_utils import load_tensor_bundle, save_csv_rows, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare magnitude, FISTA, and adaptive FISTA at matched target sparsity values."
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target-sparsity-grid",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated target sparsities, e.g. 0.3,0.5,0.7",
    )
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--search-steps", type=int, default=12)
    parser.add_argument("--sparsity-tol", type=float, default=0.02)
    parser.add_argument("--lambda-low", type=float, default=1e-6)
    parser.add_argument("--lambda-high", type=float, default=1.0)
    parser.add_argument("--bracket-scale", type=float, default=10.0)
    parser.add_argument("--max-bracket-steps", type=int, default=12)
    parser.add_argument("--r-min", type=float, default=0.5)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--include-gradient-momentum", action="store_true")
    parser.add_argument("--momentum-beta", type=float, default=1.0)
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars during target-sparsity experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summary, histories, and search traces.",
    )
    parser.add_argument(
        "--bundle-path",
        type=str,
        default=None,
        help="Optional torch bundle containing W, X, and metadata from collect_activations.py",
    )
    return parser


def _parse_float_grid(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("target-sparsity-grid must contain at least one value")
    return [float(item) for item in values]


def _resolve_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir is not None:
        return Path(raw_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / f"target_sparsity_compare_{timestamp}"


def _print_block(title: str, metrics: dict[str, object]) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def _load_problem_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if args.bundle_path:
        bundle = load_tensor_bundle(args.bundle_path)
        W = bundle["W"].to(dtype=torch.float32)
        X = bundle["X"].to(dtype=torch.float32)
        metadata = bundle.get("metadata", {})
    else:
        torch.manual_seed(args.seed)
        W = torch.randn(args.rows, args.cols)
        calibration = make_synthetic_calibration(
            num_features=args.cols,
            num_samples=args.samples,
            seed=args.seed,
        )
        X = calibration.activations
        metadata = calibration.metadata
    return W, X, metadata


def _flatten_history(
    *,
    run_id: str,
    method: str,
    target_sparsity: float,
    history: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for entry in history:
        row: dict[str, float | str] = {
            "run_id": run_id,
            "method": method,
            "target_sparsity": float(target_sparsity),
        }
        row.update(entry)
        rows.append(row)
    return rows


def _flatten_search_trace(
    *,
    run_id: str,
    method: str,
    target_sparsity: float,
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(trace):
        row: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "target_sparsity": float(target_sparsity),
            "evaluation_index": index,
        }
        row.update(entry)
        rows.append(row)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    W, X, metadata = _load_problem_data(args)
    target_sparsity_grid = _parse_float_grid(args.target_sparsity_grid)
    output_dir = _resolve_output_dir(args.output_dir)

    print("[experiment]")
    print(f"output_dir: {output_dir}")
    print(f"target_sparsity_grid: {', '.join(f'{value:.6f}' for value in target_sparsity_grid)}")
    if metadata:
        print("[data]")
        for key, value in metadata.items():
            print(f"{key}: {value}")

    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []

    target_iterator = tqdm(
        target_sparsity_grid,
        desc="Target sparsities",
        disable=args.disable_progress,
    )
    for target_sparsity in target_iterator:
        magnitude_result = MagnitudePruner(sparsity=target_sparsity).prune(W=W, X=X)
        magnitude_run_id = f"magnitude_target{target_sparsity:.6f}"
        magnitude_summary = summarize_pruning_result(
            method="magnitude",
            W=W,
            U=magnitude_result.U,
            X=X,
            extra=magnitude_result.stats,
        )
        magnitude_summary["run_id"] = magnitude_run_id
        magnitude_summary["target_sparsity"] = float(target_sparsity)
        magnitude_summary["target_gap"] = abs(
            float(magnitude_summary["actual_sparsity"]) - float(target_sparsity)
        )
        magnitude_summary["search_best_lambda"] = None
        magnitude_summary["search_best_gap"] = magnitude_summary["target_gap"]
        magnitude_summary["search_num_evaluations"] = 0
        magnitude_summary["search_bracket_found"] = None
        magnitude_summary["search_terminated_reason"] = "not_applicable"
        summary_rows.append(magnitude_summary)

        searches = [
            (
                "fista",
                FISTAPruner,
                {},
            ),
            (
                "adaptive_fista",
                AdaptiveThresholdFISTAPruner,
                {
                    "r_min": args.r_min,
                    "r_max": args.r_max,
                },
            ),
        ]
        if args.include_gradient_momentum:
            searches.append(
                (
                    "gradient_momentum_fista",
                    GradientAwareMomentumFISTAPruner,
                    {
                        "r_min": args.r_min,
                        "r_max": args.r_max,
                        "momentum_beta": args.momentum_beta,
                    },
                )
            )

        for method, pruner_cls, pruner_kwargs in searches:
            search_result = find_lambda_for_target_sparsity(
                pruner_cls=pruner_cls,
                W=W,
                X=X,
                target_sparsity=target_sparsity,
                num_iters=args.iters,
                search_steps=args.search_steps,
                sparsity_tol=args.sparsity_tol,
                lambda_low=args.lambda_low,
                lambda_high=args.lambda_high,
                bracket_scale=args.bracket_scale,
                max_bracket_steps=args.max_bracket_steps,
                pruner_kwargs=pruner_kwargs,
                show_progress=not args.disable_progress,
                progress_desc=f"{method} @ target={target_sparsity:.2f}",
            )

            run_id = f"{method}_target{target_sparsity:.6f}"
            summary = summarize_pruning_result(
                method=method,
                W=W,
                U=search_result.best_result.U,
                X=X,
                lambda_=search_result.best_lambda,
                extra=search_result.best_result.stats,
            )
            summary["run_id"] = run_id
            summary["target_sparsity"] = float(target_sparsity)
            summary["target_gap"] = abs(
                float(summary["actual_sparsity"]) - float(target_sparsity)
            )
            summary["search_best_lambda"] = float(search_result.best_lambda)
            summary["search_best_gap"] = float(search_result.best_gap)
            summary["search_num_evaluations"] = len(search_result.trials)
            summary["search_bracket_low"] = float(search_result.bracket_low)
            summary["search_bracket_high"] = float(search_result.bracket_high)
            summary["search_bracket_found"] = bool(search_result.bracket_found)
            summary["search_terminated_reason"] = search_result.terminated_reason
            summary_rows.append(summary)

            history_rows.extend(
                _flatten_history(
                    run_id=run_id,
                    method=method,
                    target_sparsity=target_sparsity,
                    history=search_result.best_result.history,
                )
            )
            search_rows.extend(
                _flatten_search_trace(
                    run_id=run_id,
                    method=method,
                    target_sparsity=target_sparsity,
                    trace=search_result.trials,
                )
            )

    save_json(
        {
            "config": {
                "bundle_path": args.bundle_path,
                "rows": args.rows,
                "cols": args.cols,
                "samples": args.samples,
                "seed": args.seed,
                "iters": args.iters,
                "search_steps": args.search_steps,
                "sparsity_tol": args.sparsity_tol,
                "lambda_low": args.lambda_low,
                "lambda_high": args.lambda_high,
                "bracket_scale": args.bracket_scale,
                "max_bracket_steps": args.max_bracket_steps,
                "r_min": args.r_min,
                "r_max": args.r_max,
                "include_gradient_momentum": args.include_gradient_momentum,
                "momentum_beta": args.momentum_beta,
                "target_sparsity_grid": target_sparsity_grid,
            },
            "data_metadata": metadata,
            "summary_rows": summary_rows,
        },
        output_dir / "summary.json",
    )
    save_csv_rows(summary_rows, output_dir / "summary.csv")
    save_json(history_rows, output_dir / "histories.json")
    save_csv_rows(history_rows, output_dir / "histories.csv")
    save_json(search_rows, output_dir / "search_trace.json")
    save_csv_rows(search_rows, output_dir / "search_trace.csv")

    print("\n[summary]")
    for row in summary_rows:
        compact = {
            "method": row["method"],
            "target_sparsity": row["target_sparsity"],
            "actual_sparsity": row["actual_sparsity"],
            "reconstruction_error": row["reconstruction_error"],
            "target_gap": row["target_gap"],
        }
        if row.get("search_best_lambda") is not None:
            compact["best_lambda"] = row["search_best_lambda"]
        _print_block(str(row["run_id"]), compact)

    print("\n[saved]")
    print(f"summary_json: {output_dir / 'summary.json'}")
    print(f"summary_csv: {output_dir / 'summary.csv'}")
    print(f"histories_json: {output_dir / 'histories.json'}")
    print(f"histories_csv: {output_dir / 'histories.csv'}")
    print(f"search_trace_json: {output_dir / 'search_trace.json'}")
    print(f"search_trace_csv: {output_dir / 'search_trace.csv'}")


if __name__ == "__main__":
    main()
