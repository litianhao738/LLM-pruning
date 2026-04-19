# Project Status

## Current Mainline

The current project mainline uses:

- corrected `nopad` activation collection
- `perplexity-first` evaluation
- masked fine-tuning, meaning pruned weights stay at zero during fine-tuning
- four main methods:
  - `magnitude`
  - `fista`
  - `adaptive_fista`
  - `gradient_momentum_fista`

Method-specific tuned settings:

- `adaptive_fista`
  - `adaptive_r_min = 0.9`
  - `adaptive_r_max = 1.1`
- `gradient_momentum_fista`
  - `gradient_r_min = 0.9`
  - `gradient_r_max = 1.1`
  - `gradient_momentum_beta = 0.001`

## Current Best Results

These are the latest headline results on the current 4-method line.

### Single-layer no fine-tuning

- best: `adaptive_fista`
- very close second: `fista`
- current ordering:
  1. `adaptive_fista`
  2. `fista`
  3. `gradient_momentum_fista`
  4. `magnitude`

### Single-layer masked fine-tuning

- best: `adaptive_fista`
- very close second: `fista`
- current ordering:
  1. `adaptive_fista`
  2. `fista`
  3. `gradient_momentum_fista`
  4. `magnitude`

### Multi-layer no fine-tuning

- best: `adaptive_fista`
- very close second: `fista`
- current ordering:
  1. `adaptive_fista`
  2. `fista`
  3. `gradient_momentum_fista`
  4. `magnitude`

### Multi-layer masked fine-tuning

- best: `adaptive_fista`
- very close second: `fista`
- current ordering:
  1. `adaptive_fista`
  2. `fista`
  3. `gradient_momentum_fista`
  4. `magnitude`

The latest compact summary is:

- [artifacts/latest_4method_mainline_summary.md](./artifacts/latest_4method_mainline_summary.md)

## Current Entry Points

These are the scripts you should use now.

- [scripts/collect_activations.py](./scripts/collect_activations.py)
  - collect the corrected `nopad` activation bundle
- [scripts/run_single_layer_perplexity_compare.py](./scripts/run_single_layer_perplexity_compare.py)
  - single-layer no-finetune comparison
- [scripts/run_single_layer_prune_then_finetune_compare.py](./scripts/run_single_layer_prune_then_finetune_compare.py)
  - single-layer masked fine-tuning comparison
- [scripts/run_multilayer_pruning.py](./scripts/run_multilayer_pruning.py)
  - core sequential multi-layer runner
- [scripts/run_formal_mainline.py](./scripts/run_formal_mainline.py)
  - convenience wrapper for the single-layer no-finetune mainline
- [scripts/run_formal_prune_then_finetune.py](./scripts/run_formal_prune_then_finetune.py)
  - convenience wrapper for the single-layer masked fine-tune mainline
- [scripts/run_formal_multilayer_mainline.py](./scripts/run_formal_multilayer_mainline.py)
  - unified multi-layer convenience wrapper

## Current Output Directories

The current mainline outputs are:

- `artifacts/distilgpt2_h0_attn_cproj_wikitext128_bundle_nopad.pt`
- `artifacts/single_layer_mainline/`
- `artifacts/single_layer_finetune_mainline/`
- `artifacts/multilayer_mainline/`
- `artifacts/multilayer_finetune_mainline/`

Older or non-headline outputs may still exist under:

- `artifacts/legacy/`
- `artifacts/archive_nopad/`
- `artifacts/tuning/`
- `artifacts/sweeps/`
- `artifacts/smoke/`

## What Changed Recently

- padding tokens are excluded from collected activations
- single-layer and multi-layer experiments are both on the corrected `nopad` line
- fine-tuning now preserves the pruning mask
- `adaptive_fista` was upgraded from a purely iteration-driven schedule to a sparsity-gap-driven thresholding rule
- `debiased_fista` was removed from the mainline because it did not provide enough practical benefit

## What Is Still Kept For Compatibility

- [scripts/run_formal_multilayer_magnitude.py](./scripts/run_formal_multilayer_magnitude.py)
- [scripts/run_formal_multilayer_gradient_momentum.py](./scripts/run_formal_multilayer_gradient_momentum.py)

These wrappers still exist, but the preferred multi-layer entry point is:

- [scripts/run_formal_multilayer_mainline.py](./scripts/run_formal_multilayer_mainline.py)

## Recommended Reading Order

1. [README.md](./README.md)
2. [PROJECT_STATUS.md](./PROJECT_STATUS.md)
3. [docs/CODE_LAYOUT.md](./docs/CODE_LAYOUT.md)
4. [artifacts/latest_4method_mainline_summary.md](./artifacts/latest_4method_mainline_summary.md)
