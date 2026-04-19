import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.formal_runs import (
    FORMAL_FT_ADAPTIVE_R_MAX,
    FORMAL_FT_ADAPTIVE_R_MIN,
    FORMAL_FT_BATCH_SIZE,
    FORMAL_FT_BUNDLE_PATH,
    FORMAL_FT_CALIBRATION_DATASET_CONFIG,
    FORMAL_FT_CALIBRATION_DATASET_NAME,
    FORMAL_FT_CALIBRATION_MIN_CHARS,
    FORMAL_FT_CALIBRATION_SOURCE,
    FORMAL_FT_CALIBRATION_SPLIT,
    FORMAL_FT_CALIBRATION_TEXT_KEY,
    FORMAL_FT_DEVICE,
    FORMAL_FT_EVAL_TEXTS,
    FORMAL_FT_FINETUNE_STEPS,
    FORMAL_FT_FINETUNE_TEXTS,
    FORMAL_FT_GM_MOMENTUM_BETA,
    FORMAL_FT_GM_R_MAX,
    FORMAL_FT_GM_R_MIN,
    FORMAL_FT_GRAD_CLIP,
    FORMAL_FT_ITERS,
    FORMAL_FT_LAYER_NAME,
    FORMAL_FT_LEARNING_RATE,
    FORMAL_FT_MAX_LENGTH,
    FORMAL_FT_METHODS,
    FORMAL_FT_MODEL_NAME,
    FORMAL_FT_OUTPUT_DIR,
    FORMAL_FT_R_MAX,
    FORMAL_FT_R_MIN,
    FORMAL_FT_SEARCH_STEPS,
    FORMAL_FT_SEED,
    FORMAL_FT_SPARSITY_TOL,
    FORMAL_FT_TARGET_SPARSITY,
    FORMAL_FT_WEIGHT_DECAY,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal single-layer prune-then-fine-tune comparison on "
            "distilgpt2 + h.0.attn.c_proj."
        )
    )
    parser.add_argument("--bundle-path", type=str, default=str(FORMAL_FT_BUNDLE_PATH))
    parser.add_argument("--device", type=str, default=FORMAL_FT_DEVICE)
    parser.add_argument("--methods", type=str, default=FORMAL_FT_METHODS)
    parser.add_argument("--finetune-steps", type=int, default=FORMAL_FT_FINETUNE_STEPS)
    parser.add_argument("--finetune-texts", type=int, default=FORMAL_FT_FINETUNE_TEXTS)
    parser.add_argument("--eval-texts", type=int, default=FORMAL_FT_EVAL_TEXTS)
    parser.add_argument("--output-dir", type=str, default=str(FORMAL_FT_OUTPUT_DIR))
    parser.add_argument("--disable-progress", action="store_true")
    return parser


def _run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = build_parser().parse_args()
    python_executable = sys.executable

    command = [
        python_executable,
        "scripts/run_single_layer_prune_then_finetune_compare.py",
        "--model-name",
        FORMAL_FT_MODEL_NAME,
        "--layer-name",
        FORMAL_FT_LAYER_NAME,
        "--bundle-path",
        args.bundle_path,
        "--methods",
        args.methods,
        "--target-sparsity",
        str(FORMAL_FT_TARGET_SPARSITY),
        "--iters",
        str(FORMAL_FT_ITERS),
        "--search-steps",
        str(FORMAL_FT_SEARCH_STEPS),
        "--sparsity-tol",
        str(FORMAL_FT_SPARSITY_TOL),
        "--r-min",
        str(FORMAL_FT_R_MIN),
        "--r-max",
        str(FORMAL_FT_R_MAX),
        "--adaptive-r-min",
        str(FORMAL_FT_ADAPTIVE_R_MIN),
        "--adaptive-r-max",
        str(FORMAL_FT_ADAPTIVE_R_MAX),
        "--gradient-r-min",
        str(FORMAL_FT_GM_R_MIN),
        "--gradient-r-max",
        str(FORMAL_FT_GM_R_MAX),
        "--gradient-momentum-beta",
        str(FORMAL_FT_GM_MOMENTUM_BETA),
        "--device",
        args.device,
        "--max-length",
        str(FORMAL_FT_MAX_LENGTH),
        "--batch-size",
        str(FORMAL_FT_BATCH_SIZE),
        "--finetune-steps",
        str(args.finetune_steps),
        "--learning-rate",
        str(FORMAL_FT_LEARNING_RATE),
        "--weight-decay",
        str(FORMAL_FT_WEIGHT_DECAY),
        "--grad-clip",
        str(FORMAL_FT_GRAD_CLIP),
        "--finetune-texts",
        str(args.finetune_texts),
        "--eval-texts",
        str(args.eval_texts),
        "--seed",
        str(FORMAL_FT_SEED),
        "--output-dir",
        args.output_dir,
        "--calibration-source",
        FORMAL_FT_CALIBRATION_SOURCE,
        "--calibration-dataset-name",
        FORMAL_FT_CALIBRATION_DATASET_NAME,
        "--calibration-dataset-config",
        FORMAL_FT_CALIBRATION_DATASET_CONFIG,
        "--calibration-split",
        FORMAL_FT_CALIBRATION_SPLIT,
        "--calibration-text-key",
        FORMAL_FT_CALIBRATION_TEXT_KEY,
        "--calibration-min-chars",
        str(FORMAL_FT_CALIBRATION_MIN_CHARS),
    ]
    if args.disable_progress:
        command.append("--disable-progress")

    _run_command(command)


if __name__ == "__main__":
    main()
