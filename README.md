# Adaptive Threshold FISTA for LLM Pruning

## Overview

This repository implements an **LLM layer-wise pruning** experiment framework.
The current codebase focuses on real-layer activation collection from `distilgpt2`,
single-layer pruning, post-pruning fine-tuning, and sequential multi-layer pruning.

Install the required packages with:

```bash
pip install -r requirements.txt
```
The project is organized into three levels:

1. Mainline
   - `magnitude`
   - `fista`
2. Strong extensions
   - `adaptive_fista`
   - `gradient_momentum_fista`
3. Bonus
   - `prune-then-fine-tune`
   - `sequential multi-layer pruning`

---

## 1. Code Modules

### 1.1 Mainline

The mainline contains the two baseline methods:

- `magnitude`
  - heuristic pruning baseline
- `fista`
  - optimization-based pruning baseline

Mainline-related files:

- `scripts/collect_activations.py`
- `models/hooks.py`
- `data/calibration.py`
- `pruning/magnitude.py`
- `pruning/fista.py`
- `pruning/search.py`
- `scripts/run_target_sparsity_compare.py`
- `scripts/run_formal_mainline.py`
- `main.py`

### 1.2 Strong Extensions

The strong extensions are:

- `adaptive_fista`
  - Adaptive-Threshold FISTA
- `gradient_momentum_fista`
  - Gradient-Aware Adaptive Momentum FISTA

Extension-related files:

- `pruning/adaptive_fista.py`
- `pruning/gradient_momentum_fista.py`
- `pruning/search.py`
- `scripts/run_target_sparsity_compare.py`

### 1.3 Bonus

The current bonus modules are:

- `prune-then-fine-tune`
- `sequential multi-layer pruning`

Bonus-related files:

- `scripts/run_prune_then_finetune.py`
- `scripts/run_formal_prune_then_finetune.py`
- `configs/formal_prune_then_finetune.py`
- `eval/perplexity.py`
- `scripts/run_multilayer_pruning.py`
- `scripts/run_formal_multilayer_gradient_momentum.py`
- `scripts/run_formal_multilayer_magnitude.py`
- `configs/formal_multilayer_gradient_momentum.py`
- `configs/formal_multilayer_magnitude.py`

---

## 2. Current Default Settings

The current locked experiment setup is:

- model: `distilgpt2`
- main layer: `transformer.h.0.attn.c_proj`
- calibration dataset: `Salesforce/wikitext`
- dataset config: `wikitext-103-raw-v1`
- calibration texts: `128`
- target sparsity grid: `0.3, 0.5, 0.7`
- iterations: `100`
- `r_min = 0.1`
- `r_max = 1.5`
- `momentum_beta = 0.5`

Default bundle path:

- `artifacts\distilgpt2_h0_attn_cproj_wikitext128_bundle.pt`

Important note:

- `python main.py` currently runs:
  - `magnitude`
  - `fista`
  - `adaptive_fista`
  - `gradient_momentum_fista`
- In other words, the current formal mainline entrypoint already includes the gradient-momentum extension.

---

## 3. Run Commands

This section is organized by experiment module.

### 3.1 Mainline

Simplest entrypoint:

```bash
python main.py
```

Equivalent formal entrypoint:

```bash
python scripts\run_formal_mainline.py
```

Manual two-step run:

First collect the bundle:

```bash
python scripts\collect_activations.py --model-name distilgpt2 --layer-name transformer.h.0.attn.c_proj --calibration-source wikitext103 --calibration-max-texts 128 --batch-size 4 --output-path artifacts\distilgpt2_h0_attn_cproj_wikitext128_bundle.pt
```

Then run the single-layer comparison:

```bash
python scripts\run_target_sparsity_compare.py --bundle-path artifacts\distilgpt2_h0_attn_cproj_wikitext128_bundle.pt --target-sparsity-grid 0.3,0.5,0.7 --iters 100 --r-min 0.1 --r-max 1.5 --include-gradient-momentum --momentum-beta 0.5 --output-dir artifacts\distilgpt2_h0_attn_cproj_wikitext128_i100_r01_15_gm05
```

### 3.2 Single-Layer Pruning Followed by Fine-Tuning

`gradient_momentum_fista`:

```bash
python scripts\run_prune_then_finetune.py --model-name distilgpt2 --layer-name transformer.h.0.attn.c_proj --bundle-path artifacts\distilgpt2_h0_attn_cproj_wikitext128_bundle.pt --method gradient_momentum_fista --target-sparsity 0.5 --iters 100 --r-min 0.1 --r-max 1.5 --momentum-beta 0.5 --finetune-steps 20 --finetune-texts 32 --eval-texts 32 --output-path artifacts\formal_prune_then_finetune_gm_s050.json
```

`magnitude`:

```bash
python scripts\run_prune_then_finetune.py --model-name distilgpt2 --layer-name transformer.h.0.attn.c_proj --bundle-path artifacts\distilgpt2_h0_attn_cproj_wikitext128_bundle.pt --method magnitude --target-sparsity 0.5 --finetune-steps 20 --finetune-texts 32 --eval-texts 32 --output-path artifacts\formal_prune_then_finetune_magnitude_s050.json
```

If you want the default formal single-layer fine-tuning run:

```bash
python scripts\run_formal_prune_then_finetune.py
```

### 3.3 Multi-Layer Pruning

The current default multi-layer setup uses:

- `transformer.h.0.attn.c_proj`
- `transformer.h.1.attn.c_proj`

Formal `gradient_momentum_fista` entrypoint:

```bash
python scripts\run_formal_multilayer_gradient_momentum.py
```

Formal `magnitude` entrypoint:

```bash
python scripts\run_formal_multilayer_magnitude.py
```

Generic `gradient_momentum_fista` entrypoint:

```bash
python scripts\run_multilayer_pruning.py --model-name distilgpt2 --layer-names transformer.h.0.attn.c_proj,transformer.h.1.attn.c_proj --method gradient_momentum_fista --target-sparsity 0.5 --iters 100 --r-min 0.1 --r-max 1.5 --momentum-beta 0.5 --calibration-source wikitext103 --calibration-max-texts 128 --eval-texts 32 --output-dir artifacts\formal_multilayer_gradient_momentum_s050
```

Generic `magnitude` entrypoint:

```bash
python scripts\run_multilayer_pruning.py --model-name distilgpt2 --layer-names transformer.h.0.attn.c_proj,transformer.h.1.attn.c_proj --method magnitude --target-sparsity 0.5 --calibration-source wikitext103 --calibration-max-texts 128 --eval-texts 32 --output-dir artifacts\formal_multilayer_magnitude_s050
```

### 3.4 Multi-Layer Pruning Followed by Fine-Tuning

Here, fine-tuning is applied only after all target layers have been pruned.

Formal `gradient_momentum_fista`:

```bash
python scripts\run_formal_multilayer_gradient_momentum.py --finetune-steps 20
```

Formal `magnitude`:

```bash
python scripts\run_formal_multilayer_magnitude.py --finetune-steps 20
```

Generic `gradient_momentum_fista`:

```bash
python scripts\run_multilayer_pruning.py --model-name distilgpt2 --layer-names transformer.h.0.attn.c_proj,transformer.h.1.attn.c_proj --method gradient_momentum_fista --target-sparsity 0.5 --iters 100 --r-min 0.1 --r-max 1.5 --momentum-beta 0.5 --calibration-source wikitext103 --calibration-max-texts 128 --finetune-steps 20 --finetune-texts 32 --learning-rate 1e-4 --weight-decay 0.0 --grad-clip 1.0 --eval-texts 32 --output-dir artifacts\formal_multilayer_gradient_momentum_s050_ft20
```

Generic `magnitude`:

```bash
python scripts\run_multilayer_pruning.py --model-name distilgpt2 --layer-names transformer.h.0.attn.c_proj,transformer.h.1.attn.c_proj --method magnitude --target-sparsity 0.5 --calibration-source wikitext103 --calibration-max-texts 128 --finetune-steps 20 --finetune-texts 32 --learning-rate 1e-4 --weight-decay 0.0 --grad-clip 1.0 --eval-texts 32 --output-dir artifacts\formal_multilayer_magnitude_s050_ft20
```

---

## 4. Output Files

### 4.1 Single-Layer Comparison

Typical outputs:

- `summary.csv`
- `summary.json`
- `histories.csv`
- `histories.json`
- `search_trace.csv`
- `search_trace.json`

### 4.2 Single-Layer Pruning Followed by Fine-Tuning

Typical output:

- `formal_prune_then_finetune_*.json`

This JSON usually contains:

- `before_pruning`
- `after_pruning`
- `after_finetuning`
- `average_nll`
- `perplexity`
- `finetune_history`

### 4.3 Multi-Layer Pruning / Multi-Layer Pruning Followed by Fine-Tuning

Typical outputs:

- `layer_summary.csv`
- `model_eval.csv`
- `histories.csv`
- `search_summary.csv`
- `report.json`

If post-pruning fine-tuning is enabled, there is one additional file:

- `finetune_history.csv`

---

## 5. How To Read the Metrics

### 5.1 Single-Layer Pruning Metrics

- `actual_sparsity`
  - the actual sparsity level of the pruned weights
- `target_gap`
  - the difference between actual sparsity and requested sparsity; smaller is better
- `reconstruction_error`
  - layer reconstruction error; smaller is better
- `best_lambda`
  - the regularization strength selected by automatic search

### 5.2 Fine-Tuning Metrics

- `average_nll`
  - average negative log-likelihood; smaller is better
- `perplexity`
  - smaller is better

### 5.3 How To Read the Multi-Layer Pipeline

In `model_eval.csv`, focus on:

- `before_pruning`
- `after_layer`
- `after_finetuning`

A reasonable trend is:

1. performance drops slightly after pruning one layer
2. it drops further after pruning another layer
3. it recovers after fine-tuning

---

## 6. Parameter Sources

The parameters in this repository are not arbitrary. They come from four main sources.

### 6.1 Model and Data Sources

- `model = distilgpt2`
- `dataset = Salesforce/wikitext`
- `dataset_config = wikitext-103-raw-v1`
- `layer = transformer.h.0.attn.c_proj`

### 6.2 Experimentally Chosen Parameters

- `target sparsity = 0.3, 0.5, 0.7`
- `iters = 100`
- `r_min = 0.1`
- `r_max = 1.5`
- `calibration_max_texts = 128`
- `momentum_beta = 0.5`

### 6.3 Automatically Chosen Parameters

- `lambda`

For each target sparsity, `lambda` is selected automatically rather than fixed by hand.

### 6.4 Main Experimental Code Paths

- `scripts/collect_activations.py`
- `scripts/run_target_sparsity_compare.py`
- `pruning/search.py`
- `pruning/adaptive_fista.py`
- `pruning/gradient_momentum_fista.py`
- `scripts/run_prune_then_finetune.py`
- `scripts/run_multilayer_pruning.py`

---

## 7. Current Conclusions

- In the single-layer setup, `magnitude` is still the strongest baseline
- `adaptive_fista` mainly improves sparsity matching
- `gradient_momentum_fista` is stronger than `fista` and `adaptive_fista`, but usually still does not beat `magnitude`
- Single-layer prune-then-fine-tune is working as expected: performance drops first, then recovers
- Sequential multi-layer pruning is working
- Multi-layer pruning followed by fine-tuning is also working, and the recovery trend is reasonable
