import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.formal_multilayer_magnitude import (
    FORMAL_ML_MAG_BATCH_SIZE,
    FORMAL_ML_MAG_CALIBRATION_DATASET_CONFIG,
    FORMAL_ML_MAG_CALIBRATION_DATASET_NAME,
    FORMAL_ML_MAG_CALIBRATION_MAX_TEXTS,
    FORMAL_ML_MAG_CALIBRATION_MIN_CHARS,
    FORMAL_ML_MAG_CALIBRATION_SOURCE,
    FORMAL_ML_MAG_CALIBRATION_SPLIT,
    FORMAL_ML_MAG_CALIBRATION_TEXT_KEY,
    FORMAL_ML_MAG_DEVICE,
    FORMAL_ML_MAG_EVAL_TEXTS,
    FORMAL_ML_MAG_FINETUNE_TEXTS,
    FORMAL_ML_MAG_GRAD_CLIP,
    FORMAL_ML_MAG_ITERS,
    FORMAL_ML_MAG_LAYER_NAMES,
    FORMAL_ML_MAG_LEARNING_RATE,
    FORMAL_ML_MAG_MAX_LENGTH,
    FORMAL_ML_MAG_METHOD,
    FORMAL_ML_MAG_MODEL_NAME,
    FORMAL_ML_MAG_MOMENTUM_BETA,
    FORMAL_ML_MAG_OUTPUT_DIR,
    FORMAL_ML_MAG_R_MAX,
    FORMAL_ML_MAG_R_MIN,
    FORMAL_ML_MAG_SEARCH_STEPS,
    FORMAL_ML_MAG_SPARSITY_TOL,
    FORMAL_ML_MAG_TARGET_SPARSITY,
    FORMAL_ML_MAG_WEIGHT_DECAY,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the formal sequential multi-layer pruning extension with magnitude pruning, with optional post-pruning fine-tuning."
    )
    parser.add_argument("--device", type=str, default=FORMAL_ML_MAG_DEVICE)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-texts", type=int, default=FORMAL_ML_MAG_FINETUNE_TEXTS)
    parser.add_argument("--disable-progress", action="store_true")
    return parser


def _run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _resolve_output_dir(raw: str | None, finetune_steps: int) -> str:
    if raw is not None:
        return raw
    if finetune_steps > 0:
        return str(FORMAL_ML_MAG_OUTPUT_DIR.parent / f"{FORMAL_ML_MAG_OUTPUT_DIR.name}_ft{finetune_steps}")
    return str(FORMAL_ML_MAG_OUTPUT_DIR)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = _resolve_output_dir(args.output_dir, args.finetune_steps)
    command = [
        sys.executable,
        "scripts/run_multilayer_pruning.py",
        "--model-name",
        FORMAL_ML_MAG_MODEL_NAME,
        "--layer-names",
        ",".join(FORMAL_ML_MAG_LAYER_NAMES),
        "--method",
        FORMAL_ML_MAG_METHOD,
        "--target-sparsity",
        str(FORMAL_ML_MAG_TARGET_SPARSITY),
        "--iters",
        str(FORMAL_ML_MAG_ITERS),
        "--search-steps",
        str(FORMAL_ML_MAG_SEARCH_STEPS),
        "--sparsity-tol",
        str(FORMAL_ML_MAG_SPARSITY_TOL),
        "--r-min",
        str(FORMAL_ML_MAG_R_MIN),
        "--r-max",
        str(FORMAL_ML_MAG_R_MAX),
        "--momentum-beta",
        str(FORMAL_ML_MAG_MOMENTUM_BETA),
        "--device",
        args.device,
        "--max-length",
        str(FORMAL_ML_MAG_MAX_LENGTH),
        "--batch-size",
        str(FORMAL_ML_MAG_BATCH_SIZE),
        "--calibration-source",
        FORMAL_ML_MAG_CALIBRATION_SOURCE,
        "--calibration-dataset-name",
        FORMAL_ML_MAG_CALIBRATION_DATASET_NAME,
        "--calibration-dataset-config",
        FORMAL_ML_MAG_CALIBRATION_DATASET_CONFIG,
        "--calibration-split",
        FORMAL_ML_MAG_CALIBRATION_SPLIT,
        "--calibration-text-key",
        FORMAL_ML_MAG_CALIBRATION_TEXT_KEY,
        "--calibration-max-texts",
        str(FORMAL_ML_MAG_CALIBRATION_MAX_TEXTS),
        "--calibration-min-chars",
        str(FORMAL_ML_MAG_CALIBRATION_MIN_CHARS),
        "--eval-texts",
        str(FORMAL_ML_MAG_EVAL_TEXTS),
        "--output-dir",
        output_dir,
    ]
    if args.finetune_steps > 0:
        command.extend(
            [
                "--finetune-steps",
                str(args.finetune_steps),
                "--finetune-texts",
                str(args.finetune_texts),
                "--learning-rate",
                str(FORMAL_ML_MAG_LEARNING_RATE),
                "--weight-decay",
                str(FORMAL_ML_MAG_WEIGHT_DECAY),
                "--grad-clip",
                str(FORMAL_ML_MAG_GRAD_CLIP),
            ]
        )
    if args.disable_progress:
        command.append("--disable-progress")
    _run_command(command)


if __name__ == "__main__":
    main()
