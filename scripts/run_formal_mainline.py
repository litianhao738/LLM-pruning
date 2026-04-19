import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.formal_mainline import (
    FORMAL_ADAPTIVE_R_MAX,
    FORMAL_ADAPTIVE_R_MIN,
    FORMAL_BUNDLE_PATH,
    FORMAL_CALIBRATION_BATCH_SIZE,
    FORMAL_CALIBRATION_DATASET_CONFIG,
    FORMAL_CALIBRATION_DATASET_NAME,
    FORMAL_EVAL_START_INDEX,
    FORMAL_EVAL_TEXTS,
    FORMAL_CALIBRATION_MAX_LENGTH,
    FORMAL_CALIBRATION_MAX_TEXTS,
    FORMAL_CALIBRATION_MIN_CHARS,
    FORMAL_CALIBRATION_SEED,
    FORMAL_CALIBRATION_SOURCE,
    FORMAL_CALIBRATION_SPLIT,
    FORMAL_CALIBRATION_TEXT_KEY,
    FORMAL_ITERS,
    FORMAL_GM_MOMENTUM_BETA,
    FORMAL_GM_R_MAX,
    FORMAL_GM_R_MIN,
    FORMAL_LAYER_NAME,
    FORMAL_METHODS,
    FORMAL_MODEL_NAME,
    FORMAL_MOMENTUM_BETA,
    FORMAL_OUTPUT_DIR,
    FORMAL_R_MAX,
    FORMAL_R_MIN,
    FORMAL_SEARCH_STEPS,
    FORMAL_SPARSITY_TOL,
    FORMAL_TARGET_SPARSITY,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current single-layer formal mainline: nopad activation collection "
            "plus perplexity-first comparison on distilgpt2 transformer.h.0.attn.c_proj."
        )
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-texts", type=int, default=FORMAL_CALIBRATION_MAX_TEXTS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_CALIBRATION_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=FORMAL_CALIBRATION_MAX_LENGTH)
    parser.add_argument("--iters", type=int, default=FORMAL_ITERS)
    parser.add_argument("--search-steps", type=int, default=FORMAL_SEARCH_STEPS)
    parser.add_argument("--sparsity-tol", type=float, default=FORMAL_SPARSITY_TOL)
    parser.add_argument("--methods", type=str, default=FORMAL_METHODS)
    parser.add_argument("--target-sparsity", type=float, default=FORMAL_TARGET_SPARSITY)
    parser.add_argument("--eval-start-index", type=int, default=FORMAL_EVAL_START_INDEX)
    parser.add_argument("--eval-texts", type=int, default=FORMAL_EVAL_TEXTS)
    parser.add_argument("--bundle-path", type=str, default=str(FORMAL_BUNDLE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(FORMAL_OUTPUT_DIR))
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser


def _run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = build_parser().parse_args()
    python_executable = sys.executable
    bundle_path = Path(args.bundle_path)

    should_collect = not args.skip_collect and (args.force_collect or not bundle_path.exists())

    if should_collect:
        collect_command = [
            python_executable,
            "scripts/collect_activations.py",
            "--model-name",
            FORMAL_MODEL_NAME,
            "--layer-name",
            FORMAL_LAYER_NAME,
            "--device",
            args.device,
            "--max-length",
            str(args.max_length),
            "--batch-size",
            str(args.batch_size),
            "--output-path",
            str(bundle_path),
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
            "--calibration-seed",
            str(FORMAL_CALIBRATION_SEED),
        ]
        if args.disable_progress:
            collect_command.append("--disable-progress")
        _run_command(collect_command)
    else:
        print(f"Using existing bundle: {bundle_path}")

    compare_command = [
        python_executable,
        "scripts/run_single_layer_perplexity_compare.py",
        "--model-name",
        FORMAL_MODEL_NAME,
        "--layer-name",
        FORMAL_LAYER_NAME,
        "--bundle-path",
        str(bundle_path),
        "--methods",
        args.methods,
        "--target-sparsity",
        str(args.target_sparsity),
        "--iters",
        str(args.iters),
        "--search-steps",
        str(args.search_steps),
        "--sparsity-tol",
        str(args.sparsity_tol),
        "--r-min",
        str(FORMAL_R_MIN),
        "--r-max",
        str(FORMAL_R_MAX),
        "--momentum-beta",
        str(FORMAL_MOMENTUM_BETA),
        "--adaptive-r-min",
        str(FORMAL_ADAPTIVE_R_MIN),
        "--adaptive-r-max",
        str(FORMAL_ADAPTIVE_R_MAX),
        "--gradient-r-min",
        str(FORMAL_GM_R_MIN),
        "--gradient-r-max",
        str(FORMAL_GM_R_MAX),
        "--gradient-momentum-beta",
        str(FORMAL_GM_MOMENTUM_BETA),
        "--device",
        args.device,
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
        "--eval-start-index",
        str(args.eval_start_index),
        "--eval-texts",
        str(args.eval_texts),
        "--output-dir",
        args.output_dir,
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
        "--calibration-min-chars",
        str(FORMAL_CALIBRATION_MIN_CHARS),
    ]
    if args.disable_progress:
        compare_command.append("--disable-progress")
    _run_command(compare_command)


if __name__ == "__main__":
    main()
