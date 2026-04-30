# Reproducibility Notes

This document records the exact commands and expected outputs for the branch `add-minority-continuation-analysis`. It is meant to make the project easy to verify from a fresh checkout and to make the reported results traceable to files in the repository.

## Branch Verification

Start by confirming that the local checkout is aligned with the GitHub feature branch.

```bash
git fetch origin add-minority-continuation-analysis
git rev-parse HEAD
git ls-remote origin add-minority-continuation-analysis
```

The hash printed by `git rev-parse HEAD` should match the hash printed by `git ls-remote`. If the hashes differ, pull the branch before running the experiments.

```bash
git pull --rebase origin add-minority-continuation-analysis
```

## Environment

The project is written for Python 3.10 or newer and uses TransformerLens for GPT-2 family model access. The recommended setup uses uv.

```bash
uv venv
source .venv/bin/activate
uv sync
```

If uv is not available, pip also works.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

CPU execution is supported. GPT-2 Medium is slower on CPU because it downloads and runs a larger model. A CUDA GPU is recommended for comfortable reruns, but the published GPT-2 and GPT-2 Medium result files are already included in the repository.

## Experiment Commands

The core experiment script runs the residual stream comparison, the SAE feature analysis, and the original next token priming analysis.

```bash
python src/experiments.py
```

The minority continuation scripts test whether plausible alternative heads receive meaningful probability mass after the same modifier.

```bash
python src/minority_continuation_analysis.py --model gpt2
python src/minority_continuation_analysis.py --model gpt2-medium
```

The contextual override scripts test whether sentence level context can shift the model from a dominant continuation to a minority continuation.

```bash
python src/contextual_override.py --model gpt2
python src/contextual_override.py --model gpt2-medium
```

## Expected Result Summaries

The exact floating point values can vary slightly across hardware, PyTorch versions, or TransformerLens versions. The qualitative results should remain stable: minority heads beat target heads in 6 of 14 analyzable compounds for both GPT-2 Small and GPT-2 Medium, and contextual override rescues 11 of 14 tested minority continuations in GPT-2 Small and 14 of 14 in GPT-2 Medium.

### Minority Continuation: GPT-2 Small

```json
{
  "n_items": 15,
  "n_analyzable": 14,
  "target_mean_probability": 0.195764531923591,
  "best_minority_mean_probability": 0.03422491519595496,
  "median_target_vs_best_minority_ratio": 4.6517337526678295,
  "median_log10_target_margin_over_best_minority": 0.66327778306868,
  "n_items_where_minority_beats_target": 6,
  "median_best_minority_vs_best_unrelated_ratio": 1797.680092875999
}
```

### Minority Continuation: GPT-2 Medium

```json
{
  "n_items": 15,
  "n_analyzable": 14,
  "target_mean_probability": 0.19892528857682398,
  "best_minority_mean_probability": 0.044531891220581855,
  "median_target_vs_best_minority_ratio": 1.840025203094858,
  "median_log10_target_margin_over_best_minority": 0.2647472855096811,
  "n_items_where_minority_beats_target": 6,
  "median_best_minority_vs_best_unrelated_ratio": 1579.209602645084
}
```

### Contextual Override: GPT-2 Small

```json
{
  "n_items": 14,
  "n_rescue_success": 11,
  "rescue_success_rate": 0.7857142857142857,
  "mean_override_strength": 1.9903861223431545,
  "median_override_strength": 2.0704939668435425,
  "target_prob_change_pct": -59.58543469625468,
  "minority_prob_change_pct": 1360.0672405779653,
  "wilcoxon_p": 0.00006103515625
}
```

### Contextual Override: GPT-2 Medium

```json
{
  "n_items": 14,
  "n_rescue_success": 14,
  "rescue_success_rate": 1.0,
  "mean_override_strength": 2.2452391938596326,
  "median_override_strength": 2.305989431097795,
  "target_prob_change_pct": -72.36256610224731,
  "minority_prob_change_pct": 1720.3605859489378,
  "wilcoxon_p": 0.00006103515625
}
```

## Output Files

The experiment outputs are written under `results` and `results/plots`.

```text
results/minority_continuations_gpt2.json
results/minority_continuations_gpt2_medium.json
results/contextual_override_gpt2.json
results/contextual_override_gpt2_medium.json
results/plots/minority_continuation_gpt2.png
results/plots/minority_continuation_gpt2_medium.png
results/plots/contextual_override_gpt2.png
results/plots/contextual_override_gpt2_medium.png
```

The scripts set Python, NumPy, and PyTorch seeds. Small numerical differences can still occur across systems, but the result pattern should remain the same.
