# Report Analysis Table (Current 4-Method Mainline)

## Key Takeaways

- `adaptive_fista` is currently the best method on all four mainline result lines.
- `fista` is extremely close to `adaptive_fista` in every setting and remains the most stable baseline.
- `gradient_momentum_fista` is consistently weaker than the top two on the current corrected line.
- `magnitude` is clearly the weakest baseline across all four lines.

## Consolidated Table

| Line | Primary Metric | Rank | Method | After-pruning PPL | After-finetuning PPL | Pruning Reconstruction Error | Actual Sparsity |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Single-layer no fine-tuning | After-pruning perplexity | 1 | adaptive_fista | 92.004454 |  | 1610.934523 | 0.503086 |
| Single-layer no fine-tuning | After-pruning perplexity | 2 | fista | 92.005115 |  | 1616.946401 | 0.503167 |
| Single-layer no fine-tuning | After-pruning perplexity | 3 | gradient_momentum_fista | 92.406288 |  | 1715.882973 | 0.492883 |
| Single-layer no fine-tuning | After-pruning perplexity | 4 | magnitude | 93.826074 |  | 2359.709048 | 0.500000 |
| Single-layer masked fine-tuning | After-finetuning perplexity | 1 | adaptive_fista | 92.004454 | 88.112521 | 1610.934523 | 0.503086 |
| Single-layer masked fine-tuning | After-finetuning perplexity | 2 | fista | 92.005115 | 88.113851 | 1616.946401 | 0.503167 |
| Single-layer masked fine-tuning | After-finetuning perplexity | 3 | gradient_momentum_fista | 92.421778 | 88.736564 | 1837.997073 | 0.506438 |
| Single-layer masked fine-tuning | After-finetuning perplexity | 4 | magnitude | 93.826074 | 90.111668 | 2359.709048 | 0.500000 |
| Multi-layer no fine-tuning | After-pruning perplexity | 1 | adaptive_fista | 109.641519 |  | 12858.388672 | 0.493371 |
| Multi-layer no fine-tuning | After-pruning perplexity | 2 | fista | 109.649047 |  | 12758.550781 | 0.492164 |
| Multi-layer no fine-tuning | After-pruning perplexity | 3 | gradient_momentum_fista | 109.886512 |  | 25040.335938 | 0.495260 |
| Multi-layer no fine-tuning | After-pruning perplexity | 4 | magnitude | 113.436202 |  | 31919.441406 | 0.500000 |
| Multi-layer masked fine-tuning | After-finetuning perplexity | 1 | adaptive_fista | 121.580329 | 113.789798 | 12858.388672 | 0.493371 |
| Multi-layer masked fine-tuning | After-finetuning perplexity | 2 | fista | 121.586783 | 113.792673 | 12758.550781 | 0.492164 |
| Multi-layer masked fine-tuning | After-finetuning perplexity | 3 | gradient_momentum_fista | 121.708670 | 114.007366 | 25040.335938 | 0.495260 |
| Multi-layer masked fine-tuning | After-finetuning perplexity | 4 | magnitude | 124.097336 | 116.108732 | 31919.441406 | 0.500000 |
