import argparse
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import save_csv_rows, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch sweep multi-layer pruning hyperparameters by repeatedly invoking "
            "scripts/run_multilayer_pruning.py and aggregating the final metrics."
        )
    )
    parser.add_argument("--model-name", type=str, default="distilgpt2")
    parser.add_argument(
        "--layer-names",
        type=str,
        default="transformer.h.0.attn.c_proj,transformer.h.1.attn.c_proj",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="fista,adaptive_fista,gradient_momentum_fista,magnitude",
        help="Comma-separated methods to include in the sweep.",
    )
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--search-steps", type=int, default=12)
    parser.add_argument("--sparsity-tol", type=float, default=0.02)
    parser.add_argument("--r-min-grid", type=str, default="0.9,1.0")
    parser.add_argument("--r-max-grid", type=str, default="1.0,1.1")
    parser.add_argument("--momentum-beta-grid", type=str, default="0.001,0.01")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--calibration-source", type=str, default="wikitext103")
    parser.add_argument("--calibration-dataset-name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--calibration-dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-text-key", type=str, default="text")
    parser.add_argument("--calibration-max-texts", type=int, default=128)
    parser.add_argument("--calibration-min-chars", type=int, default=20)
    parser.add_argument("--eval-texts", type=int, default=32)
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-texts", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument(
        "--disable-child-progress",
        action="store_true",
        help="Hide progress bars inside each run while keeping the top-level sweep progress bar.",
    )
    parser.add_argument("--disable-calibration-shuffle", action="store_true")
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke run_multilayer_pruning.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/multilayer_param_sweep",
    )
    return parser


def _parse_str_grid(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("grid argument must contain at least one value")
    return values


def _parse_float_grid(raw: str) -> list[float]:
    return [float(item) for item in _parse_str_grid(raw)]


def _method_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = _parse_str_grid(args.methods)
    configs: list[dict[str, Any]] = []
    r_min_grid = _parse_float_grid(args.r_min_grid)
    r_max_grid = _parse_float_grid(args.r_max_grid)
    momentum_beta_grid = _parse_float_grid(args.momentum_beta_grid)

    for method in methods:
        if method == "magnitude":
            configs.append({"method": method, "r_min": None, "r_max": None, "momentum_beta": None})
            continue
        if method == "fista":
            configs.append({"method": method, "r_min": None, "r_max": None, "momentum_beta": None})
            continue
        if method == "adaptive_fista":
            for r_min, r_max in itertools.product(r_min_grid, r_max_grid):
                if r_min > r_max:
                    continue
                configs.append(
                    {
                        "method": method,
                        "r_min": float(r_min),
                        "r_max": float(r_max),
                        "momentum_beta": None,
                    }
                )
            continue
        if method == "gradient_momentum_fista":
            for r_min, r_max, beta in itertools.product(r_min_grid, r_max_grid, momentum_beta_grid):
                if r_min > r_max:
                    continue
                configs.append(
                    {
                        "method": method,
                        "r_min": float(r_min),
                        "r_max": float(r_max),
                        "momentum_beta": float(beta),
                    }
                )
            continue
        raise ValueError(f"Unsupported method: {method}")

    if not configs:
        raise ValueError("No valid sweep configurations were generated")
    return configs


def _float_tag(value: float | None) -> str:
    if value is None:
        return "na"
    return str(value).replace("-", "m").replace(".", "p")


def _run_label(config: dict[str, Any]) -> str:
    method = str(config["method"])
    if method in {"magnitude", "fista"}:
        return method
    if method == "adaptive_fista":
        return f"{method}_r{_float_tag(config['r_min'])}_{_float_tag(config['r_max'])}"
    return (
        f"{method}_r{_float_tag(config['r_min'])}_{_float_tag(config['r_max'])}"
        f"_b{_float_tag(config['momentum_beta'])}"
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def _build_command(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    command = [
        args.python_executable,
        "scripts/run_multilayer_pruning.py",
        "--model-name",
        args.model_name,
        "--layer-names",
        args.layer_names,
        "--method",
        str(config["method"]),
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
        args.calibration_source,
        "--calibration-dataset-name",
        args.calibration_dataset_name,
        "--calibration-dataset-config",
        args.calibration_dataset_config,
        "--calibration-split",
        args.calibration_split,
        "--calibration-text-key",
        args.calibration_text_key,
        "--calibration-max-texts",
        str(args.calibration_max_texts),
        "--calibration-min-chars",
        str(args.calibration_min_chars),
        "--eval-texts",
        str(args.eval_texts),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    if config["r_min"] is not None:
        command.extend(["--r-min", str(config["r_min"])])
    if config["r_max"] is not None:
        command.extend(["--r-max", str(config["r_max"])])
    if config["momentum_beta"] is not None:
        command.extend(["--momentum-beta", str(config["momentum_beta"])])
    if args.finetune_steps > 0:
        command.extend(
            [
                "--finetune-steps",
                str(args.finetune_steps),
                "--finetune-texts",
                str(args.finetune_texts),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--grad-clip",
                str(args.grad_clip),
            ]
        )
    if args.disable_progress or args.disable_child_progress:
        command.append("--disable-progress")
    if args.disable_calibration_shuffle:
        command.append("--disable-calibration-shuffle")
    return command


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _summarize_run(
    *,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    layer_rows = _read_csv_rows(output_dir / "layer_summary.csv")
    eval_rows = _read_csv_rows(output_dir / "model_eval.csv")
    report_rows = _read_csv_rows(output_dir / "search_summary.csv") if (output_dir / "search_summary.csv").exists() else []

    final_layer = layer_rows[-1]
    eval_before = next(row for row in eval_rows if row["stage"] == "before_pruning")
    eval_after = eval_rows[-1]

    row = {
        "run_label": _run_label(config),
        "method": config["method"],
        "r_min": config["r_min"],
        "r_max": config["r_max"],
        "momentum_beta": config["momentum_beta"],
        "final_layer_index": int(final_layer["layer_index"]),
        "final_layer_name": final_layer["layer_name"],
        "final_actual_sparsity": _to_float(final_layer.get("actual_sparsity")),
        "final_target_gap": _to_float(final_layer.get("target_gap")),
        "final_reconstruction_error": _to_float(final_layer.get("reconstruction_error")),
        "final_selected_lambda": _to_float(final_layer.get("selected_lambda")),
        "before_nll": _to_float(eval_before.get("average_nll")),
        "before_perplexity": _to_float(eval_before.get("perplexity")),
        "final_stage": eval_after["stage"],
        "final_nll": _to_float(eval_after.get("average_nll")),
        "final_perplexity": _to_float(eval_after.get("perplexity")),
        "delta_perplexity": (
            None
            if _to_float(eval_after.get("perplexity")) is None or _to_float(eval_before.get("perplexity")) is None
            else _to_float(eval_after.get("perplexity")) - _to_float(eval_before.get("perplexity"))
        ),
        "num_layers": len(layer_rows),
        "search_rows": len(report_rows),
        "output_dir": str(output_dir),
    }
    if any(item["stage"] == "after_finetuning" for item in eval_rows):
        after_finetune = next(item for item in eval_rows if item["stage"] == "after_finetuning")
        row["after_finetuning_nll"] = _to_float(after_finetune.get("average_nll"))
        row["after_finetuning_perplexity"] = _to_float(after_finetune.get("perplexity"))
    else:
        row["after_finetuning_nll"] = None
        row["after_finetuning_perplexity"] = None
    return row


def main() -> None:
    args = build_parser().parse_args()
    configs = _method_configs(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    print("[setup]")
    print(f"methods: {args.methods}")
    print(f"num_runs: {len(configs)}")
    print(f"output_dir: {output_root}")

    iterator = enumerate(configs, start=1)
    progress_bar = None
    if not args.disable_progress:
        progress_bar = tqdm(total=len(configs), desc="Multi-layer sweep", leave=True)

    try:
        for index, config in iterator:
            run_label = _run_label(config)
            run_dir = output_root / run_label
            command = _build_command(args=args, config=config, output_dir=run_dir)

            print(f"\n[run {index}/{len(configs)}] {run_label}")
            print("$ " + " ".join(command))
            process = subprocess.Popen(command, cwd=PROJECT_ROOT)
            started_at = time.time()
            try:
                while True:
                    return_code = process.poll()
                    if progress_bar is not None:
                        elapsed_seconds = int(time.time() - started_at)
                        progress_bar.set_postfix_str(
                            f"run={run_label} elapsed={elapsed_seconds}s"
                        )
                        progress_bar.refresh()
                    if return_code is not None:
                        if return_code != 0:
                            raise subprocess.CalledProcessError(return_code, command)
                        break
                    time.sleep(1.0)
            except BaseException:
                _terminate_process(process)
                raise

            row = _summarize_run(config=config, output_dir=run_dir)
            summary_rows.append(row)
            run_rows.append(
                {
                    "run_label": run_label,
                    "method": config["method"],
                    "r_min": config["r_min"],
                    "r_max": config["r_max"],
                    "momentum_beta": config["momentum_beta"],
                    "output_dir": str(run_dir),
                    "command": " ".join(command),
                }
            )
            if progress_bar is not None:
                progress_bar.set_postfix(
                    run=run_label,
                    final_ppl=(
                        f"{row['after_finetuning_perplexity']:.4f}"
                        if args.finetune_steps > 0 and row["after_finetuning_perplexity"] is not None
                        else f"{row['final_perplexity']:.4f}"
                    ),
                )
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    sort_key = "after_finetuning_perplexity" if args.finetune_steps > 0 else "final_perplexity"
    sorted_rows = sorted(
        summary_rows,
        key=lambda row: (
            float("inf") if row.get(sort_key) is None else float(row[sort_key]),
            float("inf") if row.get("final_target_gap") is None else float(row["final_target_gap"]),
        ),
    )

    report = {
        "setup": {
            "model_name": args.model_name,
            "layer_names": args.layer_names,
            "methods": args.methods,
            "target_sparsity": float(args.target_sparsity),
            "iters": int(args.iters),
            "search_steps": int(args.search_steps),
            "sparsity_tol": float(args.sparsity_tol),
            "r_min_grid": args.r_min_grid,
            "r_max_grid": args.r_max_grid,
            "momentum_beta_grid": args.momentum_beta_grid,
            "device": args.device,
            "finetune_steps": int(args.finetune_steps),
            "seed": int(args.seed),
            "sort_key": sort_key,
        },
        "runs": run_rows,
        "summary": sorted_rows,
    }

    summary_csv = save_csv_rows(sorted_rows, output_root / "summary.csv")
    runs_csv = save_csv_rows(run_rows, output_root / "runs.csv")
    report_json = save_json(report, output_root / "report.json")

    print("\n[best]")
    best = sorted_rows[0]
    print(f"run_label: {best['run_label']}")
    print(f"method: {best['method']}")
    print(f"{sort_key}: {best[sort_key]}")
    print(f"final_target_gap: {best['final_target_gap']}")

    print("\n[saved]")
    print(f"summary_csv: {summary_csv}")
    print(f"runs_csv: {runs_csv}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
