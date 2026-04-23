import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "latest_all_results_summary.csv"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _single_layer_settings(setup: dict[str, Any], method: str) -> tuple[Any, Any, Any]:
    if method == "adaptive_fista":
        return setup.get("adaptive_r_min"), setup.get("adaptive_r_max"), None
    if method == "gradient_momentum_fista":
        return (
            setup.get("gradient_r_min"),
            setup.get("gradient_r_max"),
            setup.get("gradient_momentum_beta"),
        )
    return None, None, None


def _single_layer_rows(report_path: Path, experiment_name: str, finetune_enabled: bool) -> list[dict[str, Any]]:
    report = _load_json(report_path)
    setup = report["setup"]
    bundle_metadata = report.get("bundle_metadata", {})
    corpus_metadata = report.get("corpus_metadata", {})

    rows: list[dict[str, Any]] = []
    for summary in report["summary_rows"]:
        r_min_used, r_max_used, momentum_beta_used = _single_layer_settings(
            setup,
            summary["method"],
        )
        rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_scope": "single_layer",
                "finetune_enabled": finetune_enabled,
                "model_name": setup["model_name"],
                "layer_name": setup["layer_name"],
                "layer_names": setup["layer_name"],
                "method": summary["method"],
                "target_sparsity": summary["target_sparsity"],
                "actual_sparsity": summary["actual_sparsity"],
                "target_gap": summary["target_gap"],
                "selected_lambda": summary["selected_lambda"],
                "num_layers": 1,
                "num_iters": summary["num_iters"],
                "before_average_nll": summary["before_average_nll"],
                "before_perplexity": summary["before_perplexity"],
                "after_pruning_average_nll": summary["after_pruning_average_nll"],
                "after_pruning_perplexity": summary["after_pruning_perplexity"],
                "after_finetuning_average_nll": summary.get("after_finetuning_average_nll"),
                "after_finetuning_perplexity": summary.get("after_finetuning_perplexity"),
                "pruning_reconstruction_error": summary["pruning_reconstruction_error"],
                "objective": summary.get("objective"),
                "search_num_trials": summary.get("search_num_trials"),
                "search_terminated_reason": summary.get("search_terminated_reason"),
                "r_min_used": summary.get("r_min_used", r_min_used) or r_min_used,
                "r_max_used": summary.get("r_max_used", r_max_used) or r_max_used,
                "momentum_beta_used": (
                    summary.get("momentum_beta_used", momentum_beta_used) or momentum_beta_used
                ),
                "device": setup["device"],
                "max_length": setup["max_length"],
                "batch_size": setup["batch_size"],
                "seed": setup["seed"],
                "bundle_path": setup["bundle_path"],
                "bundle_num_texts": bundle_metadata.get("num_texts"),
                "bundle_num_samples": bundle_metadata.get("num_samples"),
                "bundle_num_valid_tokens": bundle_metadata.get("num_valid_tokens"),
                "bundle_padding_fraction_excluded": bundle_metadata.get("padding_fraction_excluded"),
                "corpus_num_texts": corpus_metadata.get("num_texts"),
                "corpus_skip_texts": corpus_metadata.get("skip_texts"),
                "requested_eval_start_index": corpus_metadata.get("requested_eval_start_index"),
                "effective_eval_start_index": corpus_metadata.get("eval_start_index"),
                "eval_texts": corpus_metadata.get("eval_texts", setup.get("eval_texts")),
                "finetune_texts": corpus_metadata.get("finetune_texts", setup.get("finetune_texts")),
                "used_pruning_cache": summary.get("used_pruning_cache"),
                "output_report_path": str(report_path.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def _multilayer_rows(report_path: Path, experiment_name: str, finetune_enabled: bool) -> list[dict[str, Any]]:
    report = _load_json(report_path)
    setup = report["setup"]

    rows: list[dict[str, Any]] = []
    for run in report["runs"]:
        summary = run["summary"]
        run_report = run["report"]
        run_setup = run_report["setup"]
        corpus_metadata = run_report.get("corpus_metadata", {})
        rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_scope": "multilayer",
                "finetune_enabled": finetune_enabled,
                "model_name": setup["model_name"],
                "layer_name": summary["final_layer_name"],
                "layer_names": ",".join(setup["layer_names"]),
                "method": summary["method"],
                "target_sparsity": summary["target_sparsity"],
                "actual_sparsity": summary["final_actual_sparsity"],
                "target_gap": summary["final_target_gap"],
                "selected_lambda": summary["final_selected_lambda"],
                "num_layers": summary["num_layers"],
                "num_iters": setup["iters"],
                "before_average_nll": summary["before_average_nll"],
                "before_perplexity": summary["before_perplexity"],
                "after_pruning_average_nll": summary["after_pruning_average_nll"],
                "after_pruning_perplexity": summary["after_pruning_perplexity"],
                "after_finetuning_average_nll": summary.get("after_finetuning_average_nll"),
                "after_finetuning_perplexity": summary.get("after_finetuning_perplexity"),
                "pruning_reconstruction_error": summary["final_pruning_reconstruction_error"],
                "objective": None,
                "search_num_trials": None,
                "search_terminated_reason": None,
                "r_min_used": summary.get("r_min_used"),
                "r_max_used": summary.get("r_max_used"),
                "momentum_beta_used": summary.get("momentum_beta_used"),
                "device": setup["device"],
                "max_length": setup["max_length"],
                "batch_size": setup["batch_size"],
                "seed": setup["seed"],
                "bundle_path": None,
                "bundle_num_texts": None,
                "bundle_num_samples": None,
                "bundle_num_valid_tokens": None,
                "bundle_padding_fraction_excluded": None,
                "corpus_num_texts": corpus_metadata.get("num_texts"),
                "corpus_skip_texts": None,
                "requested_eval_start_index": None,
                "effective_eval_start_index": None,
                "eval_texts": run_setup.get("eval_texts"),
                "finetune_texts": run_setup.get("finetune_texts"),
                "used_pruning_cache": None,
                "output_report_path": str(report_path.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        _single_layer_rows(
            PROJECT_ROOT / "artifacts" / "single_layer_mainline" / "report.json",
            experiment_name="single_layer_mainline",
            finetune_enabled=False,
        )
    )
    rows.extend(
        _single_layer_rows(
            PROJECT_ROOT / "artifacts" / "single_layer_finetune_mainline" / "report.json",
            experiment_name="single_layer_finetune_mainline",
            finetune_enabled=True,
        )
    )
    rows.extend(
        _multilayer_rows(
            PROJECT_ROOT / "artifacts" / "multilayer_mainline" / "report.json",
            experiment_name="multilayer_mainline",
            finetune_enabled=False,
        )
    )
    rows.extend(
        _multilayer_rows(
            PROJECT_ROOT / "artifacts" / "multilayer_finetune_mainline" / "report.json",
            experiment_name="multilayer_finetune_mainline",
            finetune_enabled=True,
        )
    )
    return rows


def main() -> None:
    rows = build_rows()
    _write_rows(rows, DEFAULT_OUTPUT_PATH)
    print(f"wrote {len(rows)} rows to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
