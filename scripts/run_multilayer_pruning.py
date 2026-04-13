import argparse
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

from data.calibration import load_calibration_text_corpus
from eval.reconstruction import reconstruction_error
from eval.perplexity import evaluate_perplexity_on_texts
from models.hooks import ActivationHook, apply_weight_matrix, extract_weight_matrix, resolve_module
from pruning import GradientAwareMomentumFISTAPruner, MagnitudePruner, find_lambda_for_target_sparsity
from utils.io_utils import save_csv_rows, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential layer-wise multi-layer pruning: prune one layer, write it back, then recalibrate the next layer on the updated model."
    )
    parser.add_argument("--model-name", type=str, default="distilgpt2")
    parser.add_argument("--layer-names", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "gradient_momentum_fista"],
        default="gradient_momentum_fista",
    )
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--search-steps", type=int, default=12)
    parser.add_argument("--sparsity-tol", type=float, default=0.02)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--momentum-beta", type=float, default=0.5)
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
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-texts", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-texts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--disable-calibration-shuffle", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def _parse_layer_names(raw: str) -> list[str]:
    layer_names = [item.strip() for item in raw.split(",") if item.strip()]
    if not layer_names:
        raise ValueError("layer-names must contain at least one layer name")
    return layer_names


def _resolve_output_dir(raw: str | None, *, method: str, target_sparsity: float) -> Path:
    if raw is not None:
        return Path(raw)
    return Path("artifacts") / f"multilayer_{method}_s{target_sparsity:.2f}"


def _split_texts(
    texts: list[str],
    calibration_count: int,
    finetune_count: int,
    eval_count: int,
) -> tuple[list[str], list[str], list[str]]:
    required = calibration_count + finetune_count + eval_count
    if len(texts) < required:
        raise ValueError(f"Not enough texts for split: need {required}, got {len(texts)}")
    calibration_texts = texts[:calibration_count]
    finetune_start = calibration_count
    finetune_end = finetune_start + finetune_count
    finetune_texts = texts[finetune_start:finetune_end]
    eval_texts = texts[finetune_end : finetune_end + eval_count]
    return calibration_texts, finetune_texts, eval_texts


def _freeze_except_modules(model: torch.nn.Module, target_modules: list[torch.nn.Module]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in target_modules:
        for parameter in module.parameters():
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
    show_progress: bool,
) -> list[dict[str, float]]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for fine-tuning")
    if not texts:
        raise ValueError("Fine-tuning texts are required when finetune-steps > 0")

    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    model.train()
    history: list[dict[str, float]] = []

    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Multi-layer fine-tuning", leave=False)

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
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        history.append(
            {
                "step": float(step),
                "train_loss": float(loss.item()),
            }
        )

    return history


def _collect_layer_problem(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    layer_name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = resolve_module(model, layer_name)
    hook = ActivationHook(module=module, move_to_cpu=True, flatten_batch=True)

    try:
        model.eval()
        with torch.no_grad():
            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc=f"Collect {layer_name}", leave=False)
            for start in iterator:
                batch_texts = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded)
        X = hook.stacked_inputs().to(dtype=torch.float32)
    finally:
        hook.close()

    W = extract_weight_matrix(module)
    return W, X


def _build_prune_result(args: argparse.Namespace, W: torch.Tensor, X: torch.Tensor) -> tuple[Any, float | None, dict[str, Any] | None]:
    if args.method == "magnitude":
        result = MagnitudePruner(sparsity=args.target_sparsity).prune(W=W, X=X)
        return result, None, None

    search = find_lambda_for_target_sparsity(
        pruner_cls=GradientAwareMomentumFISTAPruner,
        W=W,
        X=X,
        target_sparsity=args.target_sparsity,
        num_iters=args.iters,
        search_steps=args.search_steps,
        sparsity_tol=args.sparsity_tol,
        pruner_kwargs={
            "r_min": args.r_min,
            "r_max": args.r_max,
            "momentum_beta": args.momentum_beta,
        },
        show_progress=not args.disable_progress,
        progress_desc="gradient_momentum_fista lambda search",
    )
    search_info = {
        "target_sparsity": float(search.target_sparsity),
        "best_lambda": float(search.best_lambda),
        "best_actual_sparsity": float(search.best_actual_sparsity),
        "best_gap": float(search.best_gap),
        "bracket_low": float(search.bracket_low),
        "bracket_high": float(search.bracket_high),
        "bracket_found": bool(search.bracket_found),
        "terminated_reason": search.terminated_reason,
        "num_trials": len(search.trials),
    }
    return search.best_result, float(search.best_lambda), search_info


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    layer_names = _parse_layer_names(args.layer_names)
    output_dir = _resolve_output_dir(
        args.output_dir,
        method=args.method,
        target_sparsity=args.target_sparsity,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_finetune_texts = args.finetune_texts if args.finetune_steps > 0 else 0
    corpus = load_calibration_text_corpus(
        args.calibration_source,
        dataset_name=args.calibration_dataset_name,
        dataset_config=args.calibration_dataset_config,
        split=args.calibration_split,
        text_key=args.calibration_text_key,
        max_texts=args.calibration_max_texts + requested_finetune_texts + args.eval_texts,
        min_chars=args.calibration_min_chars,
        seed=args.seed,
        shuffle=not args.disable_calibration_shuffle,
    )
    calibration_texts, finetune_texts, eval_texts = _split_texts(
        corpus.texts,
        calibration_count=args.calibration_max_texts,
        finetune_count=requested_finetune_texts,
        eval_count=args.eval_texts,
    )

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    initial_metrics = evaluate_perplexity_on_texts(
        model=model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        show_progress=not args.disable_progress,
        progress_desc="Eval before multi-layer pruning",
    )

    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    finetune_history_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = [
        {
            "stage": "before_pruning",
            "layer_index": -1,
            "layer_name": "",
            **initial_metrics,
        }
    ]
    search_rows: list[dict[str, Any]] = []

    layer_iterator = enumerate(layer_names)
    if not args.disable_progress:
        layer_iterator = tqdm(list(layer_iterator), desc="Pruning layers")

    for layer_index, layer_name in layer_iterator:
        W, X = _collect_layer_problem(
            model=model,
            tokenizer=tokenizer,
            texts=calibration_texts,
            layer_name=layer_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            show_progress=not args.disable_progress,
        )

        prune_result, selected_lambda, search_info = _build_prune_result(args, W, X)

        target_module = resolve_module(model, layer_name)
        apply_weight_matrix(target_module, prune_result.U)

        layer_metrics = evaluate_perplexity_on_texts(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            show_progress=not args.disable_progress,
            progress_desc=f"Eval after {layer_name}",
        )

        summary_row = {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "method": args.method,
            "target_sparsity": float(args.target_sparsity),
            "target_gap": abs(float(prune_result.stats.get("actual_sparsity", 0.0)) - float(args.target_sparsity)),
            "selected_lambda": selected_lambda,
            "reconstruction_error": reconstruction_error(W=W, U=prune_result.U, X=X),
            "post_layer_average_nll": float(layer_metrics["average_nll"]),
            "post_layer_perplexity": float(layer_metrics["perplexity"]),
        }
        summary_row.update(prune_result.stats)
        summary_rows.append(summary_row)

        eval_rows.append(
            {
                "stage": "after_layer",
                "layer_index": layer_index,
                "layer_name": layer_name,
                **layer_metrics,
            }
        )

        for entry in prune_result.history:
            row = {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "method": args.method,
                "target_sparsity": float(args.target_sparsity),
            }
            row.update(entry)
            history_rows.append(row)

        if search_info is not None:
            row = {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "method": args.method,
            }
            row.update(search_info)
            search_rows.append(row)

    after_finetune_metrics = None
    if args.finetune_steps > 0:
        target_modules = [resolve_module(model, layer_name) for layer_name in layer_names]
        _freeze_except_modules(model, target_modules)
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
            show_progress=not args.disable_progress,
        )
        for entry in finetune_history:
            row = {
                "method": args.method,
                "target_sparsity": float(args.target_sparsity),
            }
            row.update(entry)
            finetune_history_rows.append(row)

        after_finetune_metrics = evaluate_perplexity_on_texts(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            show_progress=not args.disable_progress,
            progress_desc="Eval after multi-layer fine-tuning",
        )
        eval_rows.append(
            {
                "stage": "after_finetuning",
                "layer_index": len(layer_names) - 1,
                "layer_name": ",".join(layer_names),
                **after_finetune_metrics,
            }
        )

    report = {
        "setup": {
            "model_name": args.model_name,
            "layer_names": layer_names,
            "method": args.method,
            "target_sparsity": float(args.target_sparsity),
            "iters": int(args.iters),
            "search_steps": int(args.search_steps),
            "sparsity_tol": float(args.sparsity_tol),
            "r_min": float(args.r_min),
            "r_max": float(args.r_max),
            "momentum_beta": float(args.momentum_beta),
            "device": str(device),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "calibration_texts": len(calibration_texts),
            "finetune_steps": int(args.finetune_steps),
            "finetune_texts": len(finetune_texts),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "eval_texts": len(eval_texts),
            "seed": int(args.seed),
        },
        "corpus_metadata": corpus.metadata,
        "initial_metrics": initial_metrics,
        "layer_summaries": summary_rows,
        "evaluation_history": eval_rows,
        "lambda_search": search_rows,
        "finetune_history": finetune_history_rows,
    }

    summary_csv = save_csv_rows(summary_rows, output_dir / "layer_summary.csv")
    eval_csv = save_csv_rows(eval_rows, output_dir / "model_eval.csv")
    histories_csv = save_csv_rows(history_rows, output_dir / "histories.csv")
    finetune_csv = None
    if finetune_history_rows:
        finetune_csv = save_csv_rows(finetune_history_rows, output_dir / "finetune_history.csv")
    report_json = save_json(report, output_dir / "report.json")
    search_csv = None
    if search_rows:
        search_csv = save_csv_rows(search_rows, output_dir / "search_summary.csv")

    print("[setup]")
    print(f"model_name: {args.model_name}")
    print(f"method: {args.method}")
    print(f"layer_names: {', '.join(layer_names)}")
    print(f"target_sparsity: {args.target_sparsity:.6f}")
    print(f"calibration_texts: {len(calibration_texts)}")
    print(f"finetune_steps: {args.finetune_steps}")
    print(f"finetune_texts: {len(finetune_texts)}")
    print(f"eval_texts: {len(eval_texts)}")

    print("\n[initial_metrics]")
    print(f"average_nll: {initial_metrics['average_nll']:.6f}")
    print(f"perplexity: {initial_metrics['perplexity']:.6f}")

    print("\n[layer_summary]")
    for row in summary_rows:
        print(f"[{row['layer_name']}]")
        print(f"actual_sparsity: {float(row['actual_sparsity']):.6f}")
        print(f"reconstruction_error: {float(row['reconstruction_error']):.6f}")
        print(f"post_layer_perplexity: {float(row['post_layer_perplexity']):.6f}")
        if row.get("selected_lambda") is not None:
            print(f"selected_lambda: {float(row['selected_lambda']):.6f}")

    if after_finetune_metrics is not None:
        print("\n[after_finetuning]")
        print(f"average_nll: {after_finetune_metrics['average_nll']:.6f}")
        print(f"perplexity: {after_finetune_metrics['perplexity']:.6f}")

    print("\n[saved]")
    print(f"layer_summary_csv: {summary_csv}")
    print(f"model_eval_csv: {eval_csv}")
    print(f"histories_csv: {histories_csv}")
    if finetune_csv is not None:
        print(f"finetune_history_csv: {finetune_csv}")
    if search_csv is not None:
        print(f"search_summary_csv: {search_csv}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
