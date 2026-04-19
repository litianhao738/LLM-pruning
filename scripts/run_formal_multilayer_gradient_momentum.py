import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.formal_runs import (
    FORMAL_ML_DEVICE,
    FORMAL_ML_EVAL_TEXTS,
    FORMAL_ML_FINETUNE_TEXTS,
    FORMAL_ML_GM_METHOD,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for running the formal multi-layer mainline "
            "with gradient_momentum_fista only."
        )
    )
    parser.add_argument("--device", type=str, default=FORMAL_ML_DEVICE)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-texts", type=int, default=FORMAL_ML_FINETUNE_TEXTS)
    parser.add_argument("--eval-texts", type=int, default=FORMAL_ML_EVAL_TEXTS)
    parser.add_argument("--disable-progress", action="store_true")
    return parser


def _run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = build_parser().parse_args()
    command = [
        sys.executable,
        "scripts/run_formal_multilayer_mainline.py",
        "--methods",
        FORMAL_ML_GM_METHOD,
        "--device",
        args.device,
        "--finetune-steps",
        str(args.finetune_steps),
        "--finetune-texts",
        str(args.finetune_texts),
        "--eval-texts",
        str(args.eval_texts),
    ]
    if args.output_dir is not None:
        command.extend(["--output-dir", args.output_dir])
    if args.disable_progress:
        command.append("--disable-progress")
    _run_command(command)


if __name__ == "__main__":
    main()
