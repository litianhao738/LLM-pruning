# FISTA-Based Layer-Wise Pruning for LLMs

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Collect the corrected `nopad` activation bundle:

```bash
python scripts/collect_activations.py --model-name distilgpt2 --layer-name transformer.h.0.attn.c_proj --device cuda --calibration-source wikitext103 --calibration-dataset-name Salesforce/wikitext --calibration-dataset-config wikitext-103-raw-v1 --calibration-split train --calibration-text-key text --calibration-max-texts 128 --calibration-min-chars 20 --calibration-seed 7 --max-length 64 --batch-size 4 --output-path artifacts/distilgpt2_h0_attn_cproj_wikitext128_bundle_nopad.pt
```

Run the current 4-method single-layer comparison:

```bash
python scripts/run_single_layer_perplexity_compare.py --model-name distilgpt2 --layer-name transformer.h.0.attn.c_proj --bundle-path artifacts/distilgpt2_h0_attn_cproj_wikitext128_bundle_nopad.pt --methods magnitude,fista,adaptive_fista,gradient_momentum_fista --target-sparsity 0.5 --iters 1000 --search-steps 12 --sparsity-tol 0.01 --adaptive-r-min 0.9 --adaptive-r-max 1.1 --gradient-r-min 0.9 --gradient-r-max 1.1 --gradient-momentum-beta 0.001 --device cuda --max-length 64 --batch-size 4 --eval-start-index 32 --eval-texts 32 --output-dir artifacts/single_layer_mainline
```

Run single-layer pruning followed by masked fine-tuning:

```bash
python scripts/run_single_layer_prune_then_finetune_compare.py --model-name distilgpt2 --layer-name transformer.h.0.attn.c_proj --bundle-path artifacts/distilgpt2_h0_attn_cproj_wikitext128_bundle_nopad.pt --methods magnitude,fista,adaptive_fista,gradient_momentum_fista --target-sparsity 0.5 --iters 1000 --search-steps 12 --sparsity-tol 0.01 --r-min 0.1 --r-max 1.5 --adaptive-r-min 0.9 --adaptive-r-max 1.1 --gradient-r-min 0.9 --gradient-r-max 1.1 --gradient-momentum-beta 0.001 --device cuda --max-length 64 --batch-size 4 --finetune-steps 20 --learning-rate 1e-4 --weight-decay 0.0 --grad-clip 1.0 --finetune-texts 32 --eval-texts 32 --seed 7 --output-dir artifacts/single_layer_finetune_mainline
```

Run the current 4-method multi-layer comparison:

```bash
python scripts/run_formal_multilayer_mainline.py --methods magnitude,fista,adaptive_fista,gradient_momentum_fista --device cuda --output-dir artifacts/multilayer_mainline
```

Run multi-layer pruning followed by masked fine-tuning:

```bash
python scripts/run_formal_multilayer_mainline.py --methods magnitude,fista,adaptive_fista,gradient_momentum_fista --device cuda --finetune-steps 20 --output-dir artifacts/multilayer_finetune_mainline
```

Minimal convenience wrappers:

```bash
python main.py
python scripts/run_formal_prune_then_finetune.py --device cuda
python scripts/run_formal_multilayer_mainline.py --device cuda
```

## What This Project Does

This repository studies structured comparisons between four layer-wise pruning methods on `distilgpt2`:

- `magnitude`
- `fista`
- `adaptive_fista`
- `gradient_momentum_fista`

The workflow is:

1. Collect real activations from a target layer.
2. Prune weights using one method at a matched target sparsity.
3. Evaluate with perplexity and reconstruction error.
4. Optionally fine-tune while **preserving the pruning mask**.
5. Repeat the same process for sequential multi-layer pruning.

The project uses a corrected `nopad` activation pipeline, so padding tokens are excluded from the activation matrix `X`.

## Current Mainline

The current mainline is:

- model: `distilgpt2`
- single-layer target: `transformer.h.0.attn.c_proj`
- multi-layer targets:
  - `transformer.h.0.attn.c_proj`
  - `transformer.h.1.attn.c_proj`
- dataset: `Salesforce/wikitext`, config `wikitext-103-raw-v1`
- calibration texts: `128`
- target sparsity: `0.5`
- iterations: `1000`
- search steps: `12`
- sparsity tolerance: `0.01`
- fine-tune steps: `20`
- fine-tune mask policy: **keep pruned weights at zero**

The default activation bundle is:

- `artifacts/distilgpt2_h0_attn_cproj_wikitext128_bundle_nopad.pt`

## Where the Parameters Come From

The parameters are not arbitrary. They come from three sources.

### 1. Fixed experiment setup

These are chosen to define one stable evaluation line:

- `distilgpt2`
- `wikitext-103-raw-v1`
- target sparsity `0.5`
- `128` calibration texts
- `32` evaluation texts
- `1000` pruning iterations

These defaults live in:

- [configs/formal_runs.py](./configs/formal_runs.py)

### 2. Automatically searched parameters

For `fista`, `adaptive_fista`, and `gradient_momentum_fista`, the regularization weight `lambda` is **not fixed by hand**. It is selected automatically by binary/bracketed search to match the target sparsity as closely as possible.

This search logic lives in:

- [pruning/search.py](./pruning/search.py)

### 3. Method-specific tuned parameters

Some methods need extra hyperparameters beyond `lambda`.

Current tuned defaults:

- `adaptive_fista`
  - `adaptive_r_min = 0.9`
  - `adaptive_r_max = 1.1`
- `gradient_momentum_fista`
  - `gradient_r_min = 0.9`
  - `gradient_r_max = 1.1`
  - `gradient_momentum_beta = 0.001`

These values were chosen from empirical comparisons in this repository. They are the current best-performing stable settings on the corrected `nopad` line.

## Code Structure

The code is organized into a few small layers.

### Config

- [configs/formal_runs.py](./configs/formal_runs.py)
  - central defaults for the current mainline
- [configs/formal_mainline.py](./configs/formal_mainline.py)
  - thin compatibility export

### Data and model plumbing

- [data/calibration.py](./data/calibration.py)
  - loads calibration, fine-tune, and evaluation texts
- [models/hooks.py](./models/hooks.py)
  - collects activations and applies the `nopad` filtering fix

### Pruning methods

- [pruning/magnitude.py](./pruning/magnitude.py)
- [pruning/fista.py](./pruning/fista.py)
- [pruning/adaptive_fista.py](./pruning/adaptive_fista.py)
- [pruning/gradient_momentum_fista.py](./pruning/gradient_momentum_fista.py)
- [pruning/search.py](./pruning/search.py)

### Shared utilities

- [utils/single_layer_utils.py](./utils/single_layer_utils.py)
  - shared method construction and lambda-search wiring
- [utils/finetune_masks.py](./utils/finetune_masks.py)
  - keeps pruned weights fixed at zero during fine-tuning

### Main experiment scripts

- [scripts/collect_activations.py](./scripts/collect_activations.py)
- [scripts/run_single_layer_perplexity_compare.py](./scripts/run_single_layer_perplexity_compare.py)
- [scripts/run_single_layer_prune_then_finetune_compare.py](./scripts/run_single_layer_prune_then_finetune_compare.py)
- [scripts/run_multilayer_pruning.py](./scripts/run_multilayer_pruning.py)
- [scripts/run_formal_mainline.py](./scripts/run_formal_mainline.py)
- [scripts/run_formal_prune_then_finetune.py](./scripts/run_formal_prune_then_finetune.py)
- [scripts/run_formal_multilayer_mainline.py](./scripts/run_formal_multilayer_mainline.py)

## Main Outputs

Typical output files are:

- single-layer no fine-tuning
  - `summary.csv`
  - `report.json`
  - `search_trace.csv`
- single-layer fine-tuning
  - `summary.csv`
  - `report.json`
  - `search_trace.csv`
  - `finetune_history.csv`
- multi-layer
  - `summary.csv`
  - per-method `layer_summary.csv`
  - per-method `model_eval.csv`
  - per-method `report.json`
  - optional `finetune_history.csv`

The most important metrics are:

- `actual_sparsity`
- `target_gap`
- `after_pruning_perplexity`
- `after_finetuning_perplexity`
- `reconstruction_error`

## Notes

- Fine-tuning now preserves the pruning mask. Pruned weights are kept at zero.
- `scripts/run_target_sparsity_compare.py` is still available, but it is now a legacy/diagnostic script rather than the main entry point.
- `scripts/run_formal_multilayer_magnitude.py` and `scripts/run_formal_multilayer_gradient_momentum.py` are compatibility wrappers. The unified multi-layer entry point is `scripts/run_formal_multilayer_mainline.py`.

## Related Files

For a short project snapshot, see:

- [PROJECT_STATUS.md](./PROJECT_STATUS.md)

For a code map, see:

- [docs/CODE_LAYOUT.md](./docs/CODE_LAYOUT.md)
