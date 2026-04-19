# Code Layout

## Current Entry Points

- [main.py](</d:/7503/main.py:1>)
  Compatibility shortcut for the older formal single-layer mainline wrapper.
- [scripts/collect_activations.py](</d:/7503/scripts/collect_activations.py:1>)
  Collects real activations and exports `W/X` bundles. This is the source of truth for the nopad fix.
- [scripts/run_single_layer_perplexity_compare.py](</d:/7503/scripts/run_single_layer_perplexity_compare.py:1>)
  Current single-layer headline experiment.
- [scripts/run_single_layer_prune_then_finetune_compare.py](</d:/7503/scripts/run_single_layer_prune_then_finetune_compare.py:1>)
  Current single-layer prune-then-fine-tune comparison entry point.
- [scripts/run_prune_then_finetune.py](</d:/7503/scripts/run_prune_then_finetune.py:1>)
  Single-method single-layer prune then fine-tune runner.
- [scripts/run_multilayer_pruning.py](</d:/7503/scripts/run_multilayer_pruning.py:1>)
  Main multi-layer runner.
- [scripts/run_multilayer_param_sweep.py](</d:/7503/scripts/run_multilayer_param_sweep.py:1>)
  Batch sweep driver for multi-layer hyperparameter scans.

## Code Folders

- `pruning/`
  Core pruning methods and lambda search.
- `models/`
  Hooking and model weight extraction / writeback helpers.
- `eval/`
  Reconstruction and perplexity metrics.
- `data/`
  Calibration text loading.
- `utils/`
  Shared math and IO helpers.
  This now also includes [utils/single_layer_utils.py](</d:/7503/utils/single_layer_utils.py:1>) for shared single-layer pruning / text-split helpers.
- `configs/`
  Centralized formal defaults. The main configuration source is now [configs/formal_runs.py](</d:/7503/configs/formal_runs.py:1>).

## Formal Config Structure

- [configs/formal_runs.py](</d:/7503/configs/formal_runs.py:1>)
  Holds the shared formal defaults for:
  single-layer compare, prune-then-fine-tune, and multi-layer runs.
- [configs/formal_mainline.py](</d:/7503/configs/formal_mainline.py:1>)
  Thin compatibility export layer for older imports.

## Wrapper Scripts

- [scripts/run_formal_mainline.py](</d:/7503/scripts/run_formal_mainline.py:1>)
- [scripts/run_formal_prune_then_finetune.py](</d:/7503/scripts/run_formal_prune_then_finetune.py:1>)
- [scripts/run_formal_multilayer_mainline.py](</d:/7503/scripts/run_formal_multilayer_mainline.py:1>)
- [scripts/run_formal_multilayer_magnitude.py](</d:/7503/scripts/run_formal_multilayer_magnitude.py:1>)
- [scripts/run_formal_multilayer_gradient_momentum.py](</d:/7503/scripts/run_formal_multilayer_gradient_momentum.py:1>)

These are still convenience wrappers, but they now read from one central formal config instead of separate duplicated config files. The unified multi-layer entry point is `run_formal_multilayer_mainline.py`; the method-specific multi-layer wrappers are now just compatibility shims.

## Current Mainline

- Data: nopad activation collection
- Single-layer headline metric: perplexity
- Main optimization baseline: `fista`
- Tuned variants:
  `adaptive_fista` near `r_min=0.9, r_max=1.1`
  `gradient_momentum_fista` near `momentum_beta=0.001, r_min=0.9, r_max=1.1`

## Remaining Messy Areas

- `artifacts/` is cleaner now, but it still includes some historical per-method multi-layer result folders from before the unified launcher landed.
- README still contains historical experiment notes in addition to the current layout.
