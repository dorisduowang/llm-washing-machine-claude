"""
Contextual Override Experiment for Compound Concepts
=====================================================

Building on the minority-continuation analysis, this experiment tests
whether disambiguating sentence context can rescue minority continuations
from being suppressed by a dominant compound head.

Research question:
    When "The washing" strongly primes "machine", can a steering context
    like "She poured the washing" shift the model toward "powder" instead?

This addresses REPORT.md Limitation 2 (context sensitivity) and connects
to the broader question of whether compound knowledge in LLMs reflects
shallow bigram co-occurrence or deeper contextual understanding.

Design:
    For each modifier we construct three context types:
      - neutral:     minimal context ("The {modifier}")
      - reinforcing: context biasing toward the dominant head
                     ("He repaired the {modifier}")
      - rescuing:    context biasing toward a specific minority head
                     ("She poured the washing" → "powder")

    We then measure whether P(minority_head) increases and P(target_head)
    decreases under the rescuing context relative to the neutral baseline.

Outputs:
    - results/contextual_override_<model>.json
    - results/plots/contextual_override_<model>.png

"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformer_lens as tl

SEED = 42
EPS = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# ─────────────────────────────────────────────
# 1. DATASET: hand-crafted contextual prompts
# ─────────────────────────────────────────────

OVERRIDE_ITEMS: list[dict[str, Any]] = [
    {
        "modifier": "washing",
        "target_head": "machine",
        "rescued_minority": "powder",
        "neutral_prompts":      ["The washing", "A washing", "This washing"],
        "reinforcing_prompts":  ["He repaired the washing", "She loaded the washing", "The broken washing"],
        "rescuing_prompts":     ["She poured the washing", "He bought more washing", "They ran out of washing"],
    },
    {
        "modifier": "washing",
        "target_head": "machine",
        "rescued_minority": "instructions",
        "neutral_prompts":      ["The washing", "A washing", "This washing"],
        "reinforcing_prompts":  ["He repaired the washing", "She loaded the washing", "The broken washing"],
        "rescuing_prompts":     ["Please read the washing", "He followed the washing", "She printed the washing"],
    },
    {
        "modifier": "coffee",
        "target_head": "shop",
        "rescued_minority": "table",
        "neutral_prompts":      ["The coffee", "A coffee", "This coffee"],
        "reinforcing_prompts":  ["She entered the coffee", "He found a coffee", "The new coffee"],
        "rescuing_prompts":     ["She wiped the coffee", "He sat at the coffee", "The wooden coffee"],
    },
    {
        "modifier": "coffee",
        "target_head": "shop",
        "rescued_minority": "beans",
        "neutral_prompts":      ["The coffee", "A coffee", "This coffee"],
        "reinforcing_prompts":  ["She entered the coffee", "He found a coffee", "The new coffee"],
        "rescuing_prompts":     ["She ground the coffee", "He roasted the coffee", "They imported coffee"],
    },
    {
        "modifier": "swimming",
        "target_head": "pool",
        "rescued_minority": "lessons",
        "neutral_prompts":      ["The swimming", "A swimming", "This swimming"],
        "reinforcing_prompts":  ["He cleaned the swimming", "She dove into the swimming", "The heated swimming"],
        "rescuing_prompts":     ["She signed up for swimming", "He booked the swimming", "The children started swimming"],
    },
    {
        "modifier": "swimming",
        "target_head": "pool",
        "rescued_minority": "team",
        "neutral_prompts":      ["The swimming", "A swimming", "This swimming"],
        "reinforcing_prompts":  ["He cleaned the swimming", "She dove into the swimming", "The heated swimming"],
        "rescuing_prompts":     ["She joined the swimming", "He coached the swimming", "The school swimming"],
    },
    {
        "modifier": "parking",
        "target_head": "lot",
        "rescued_minority": "ticket",
        "neutral_prompts":      ["The parking", "A parking", "This parking"],
        "reinforcing_prompts":  ["He crossed the parking", "She drove into the parking", "The empty parking"],
        "rescuing_prompts":     ["She received a parking", "He paid the parking", "The officer wrote a parking"],
    },
    {
        "modifier": "parking",
        "target_head": "lot",
        "rescued_minority": "garage",
        "neutral_prompts":      ["The parking", "A parking", "This parking"],
        "reinforcing_prompts":  ["He crossed the parking", "She drove into the parking", "The empty parking"],
        "rescuing_prompts":     ["She drove into the underground parking", "He parked inside the parking", "The multi-story parking"],
    },
    {
        "modifier": "fire",
        "target_head": "truck",
        "rescued_minority": "alarm",
        "neutral_prompts":      ["The fire", "A fire", "This fire"],
        "reinforcing_prompts":  ["He drove the fire", "She saw the fire", "The red fire"],
        "rescuing_prompts":     ["She triggered the fire", "He tested the fire", "The broken fire"],
    },
    {
        "modifier": "fire",
        "target_head": "truck",
        "rescued_minority": "station",
        "neutral_prompts":      ["The fire", "A fire", "This fire"],
        "reinforcing_prompts":  ["He drove the fire", "She saw the fire", "The red fire"],
        "rescuing_prompts":     ["She visited the fire", "He worked at the fire", "The nearest fire"],
    },
    {
        "modifier": "ice",
        "target_head": "cream",
        "rescued_minority": "hockey",
        "neutral_prompts":      ["The ice", "A ice", "This ice"],
        "reinforcing_prompts":  ["She scooped the ice", "He ordered ice", "The melted ice"],
        "rescuing_prompts":     ["She played ice", "He watched the ice", "The professional ice"],
    },
    {
        "modifier": "school",
        "target_head": "bus",
        "rescued_minority": "board",
        "neutral_prompts":      ["The school", "A school", "This school"],
        "reinforcing_prompts":  ["She waited for the school", "He drove the school", "The yellow school"],
        "rescuing_prompts":     ["She addressed the school", "He joined the school", "The elected school"],
    },
    {
        "modifier": "hot",
        "target_head": "dog",
        "rescued_minority": "water",
        "neutral_prompts":      ["The hot", "A hot", "This hot"],
        "reinforcing_prompts":  ["She ate a hot", "He grilled a hot", "The leftover hot"],
        "rescuing_prompts":     ["She boiled the hot", "He poured the hot", "The scalding hot"],
    },
    {
        "modifier": "dark",
        "target_head": "horse",
        "rescued_minority": "room",
        "neutral_prompts":      ["The dark", "A dark", "This dark"],
        "reinforcing_prompts":  ["She bet on the dark", "He was the dark", "The political dark"],
        "rescuing_prompts":     ["She entered the dark", "He sat in the dark", "The cold dark"],
    },
]


# ─────────────────────────────────────────────
# 2. MODEL HELPERS
# ─────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_token_id(model: tl.HookedTransformer, word: str) -> int | None:
    ids = model.to_tokens(f" {word}", prepend_bos=False)[0].tolist()
    return int(ids[0]) if len(ids) == 1 else None


def measure_prob(
    model: tl.HookedTransformer,
    prompt: str,
    token_id: int,
    device: str,
) -> dict[str, Any]:
    tokens = model.to_tokens(prompt).to(device)
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1], dim=-1)
    prob = float(probs[token_id].item())
    rank = int((probs > probs[token_id]).sum().item() + 1)

    top5_vals, top5_ids = torch.topk(probs, 5)
    top5 = [
        {"token": model.to_string([int(tid)]).strip(), "prob": float(v)}
        for v, tid in zip(top5_vals.tolist(), top5_ids.tolist())
    ]
    return {"prompt": prompt, "probability": prob, "rank": rank, "top5": top5}


# ─────────────────────────────────────────────
# 3. CORE EXPERIMENT
# ─────────────────────────────────────────────

def run_item(
    model: tl.HookedTransformer,
    item: dict[str, Any],
    device: str,
) -> dict[str, Any] | None:
    target_id = get_token_id(model, item["target_head"])
    minority_id = get_token_id(model, item["rescued_minority"])
    if target_id is None or minority_id is None:
        return None

    conditions = {}
    for cond_name in ("neutral", "reinforcing", "rescuing"):
        prompts = item[f"{cond_name}_prompts"]
        target_results = [measure_prob(model, p, target_id, device) for p in prompts]
        minority_results = [measure_prob(model, p, minority_id, device) for p in prompts]

        mean_target = float(np.mean([r["probability"] for r in target_results]))
        mean_minority = float(np.mean([r["probability"] for r in minority_results]))

        conditions[cond_name] = {
            "target_prob": mean_target,
            "minority_prob": mean_minority,
            "target_rank": float(np.median([r["rank"] for r in target_results])),
            "minority_rank": float(np.median([r["rank"] for r in minority_results])),
            "target_details": target_results,
            "minority_details": minority_results,
        }

    # Key metrics: how much does rescuing context shift the balance?
    neutral_ratio = conditions["neutral"]["target_prob"] / max(conditions["neutral"]["minority_prob"], EPS)
    rescue_ratio = conditions["rescuing"]["target_prob"] / max(conditions["rescuing"]["minority_prob"], EPS)
    reinforce_ratio = conditions["reinforcing"]["target_prob"] / max(conditions["reinforcing"]["minority_prob"], EPS)

    rescue_success = conditions["rescuing"]["minority_prob"] > conditions["rescuing"]["target_prob"]

    # "Override strength": how many log-units did the rescuing context
    # shift the target/minority ratio compared to neutral?
    override_strength = math.log10(neutral_ratio + EPS) - math.log10(rescue_ratio + EPS)

    return {
        "modifier": item["modifier"],
        "target_head": item["target_head"],
        "rescued_minority": item["rescued_minority"],
        "conditions": conditions,
        "neutral_target_minority_ratio": neutral_ratio,
        "rescue_target_minority_ratio": rescue_ratio,
        "reinforce_target_minority_ratio": reinforce_ratio,
        "override_strength_log10": override_strength,
        "rescue_success": rescue_success,
    }


# ─────────────────────────────────────────────
# 4. SUMMARY STATISTICS
# ─────────────────────────────────────────────

def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    override_strengths = [r["override_strength_log10"] for r in results]
    rescue_successes = [r["rescue_success"] for r in results]

    neutral_target_probs = [r["conditions"]["neutral"]["target_prob"] for r in results]
    rescue_target_probs = [r["conditions"]["rescuing"]["target_prob"] for r in results]
    neutral_minority_probs = [r["conditions"]["neutral"]["minority_prob"] for r in results]
    rescue_minority_probs = [r["conditions"]["rescuing"]["minority_prob"] for r in results]

    # Paired test: does rescuing context significantly increase minority prob?
    from scipy import stats
    if len(results) >= 5:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            rescue_minority_probs, neutral_minority_probs, alternative="greater"
        )
    else:
        wilcoxon_stat, wilcoxon_p = None, None

    return {
        "n_items": len(results),
        "n_rescue_success": sum(rescue_successes),
        "rescue_success_rate": sum(rescue_successes) / max(len(rescue_successes), 1),
        "mean_override_strength": float(np.mean(override_strengths)),
        "median_override_strength": float(np.median(override_strengths)),
        "mean_neutral_target_prob": float(np.mean(neutral_target_probs)),
        "mean_rescue_target_prob": float(np.mean(rescue_target_probs)),
        "mean_neutral_minority_prob": float(np.mean(neutral_minority_probs)),
        "mean_rescue_minority_prob": float(np.mean(rescue_minority_probs)),
        "target_prob_change_pct": float(
            (np.mean(rescue_target_probs) - np.mean(neutral_target_probs))
            / max(np.mean(neutral_target_probs), EPS) * 100
        ),
        "minority_prob_change_pct": float(
            (np.mean(rescue_minority_probs) - np.mean(neutral_minority_probs))
            / max(np.mean(neutral_minority_probs), EPS) * 100
        ),
        "wilcoxon_stat": float(wilcoxon_stat) if wilcoxon_stat is not None else None,
        "wilcoxon_p": float(wilcoxon_p) if wilcoxon_p is not None else None,
    }


# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(results: list[dict[str, Any]], output_path: Path) -> None:
    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        "Contextual Override: Can Disambiguating Context Rescue Minority Continuations?",
        fontsize=13, fontweight="bold",
    )

    labels = [f"{r['modifier']} → {r['rescued_minority']}\n(vs {r['target_head']})" for r in results]
    n = len(results)

    # ── Panel 1: Minority probability across three conditions ──
    ax = axes[0]
    x = np.arange(n)
    w = 0.25
    neutral_min = [r["conditions"]["neutral"]["minority_prob"] for r in results]
    reinforce_min = [r["conditions"]["reinforcing"]["minority_prob"] for r in results]
    rescue_min = [r["conditions"]["rescuing"]["minority_prob"] for r in results]

    ax.barh(x - w, neutral_min, height=w, label="Neutral", color="#7f8c8d")
    ax.barh(x, reinforce_min, height=w, label="Reinforcing", color="#e74c3c")
    ax.barh(x + w, rescue_min, height=w, label="Rescuing", color="#27ae60")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("P(minority head)")
    ax.set_title("Minority Head Probability\nby Context Type")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # ── Panel 2: Override strength ──
    ax = axes[1]
    strengths = [r["override_strength_log10"] for r in results]
    colors = ["#27ae60" if s > 0 else "#e74c3c" for s in strengths]
    ax.barh(x, strengths, color=colors, edgecolor="black", alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Override Strength (log₁₀ scale)")
    ax.set_title("How Much Does Rescuing Context\nShift the Balance?")
    ax.invert_yaxis()

    # ── Panel 3: Target vs minority probability under rescue ──
    ax = axes[2]
    rescue_tgt = [r["conditions"]["rescuing"]["target_prob"] for r in results]
    rescue_mn = [r["conditions"]["rescuing"]["minority_prob"] for r in results]
    ax.barh(x - 0.15, rescue_tgt, height=0.3, label="Target head", color="#3498db")
    ax.barh(x + 0.15, rescue_mn, height=0.3, label="Minority head", color="#27ae60")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Probability under rescuing context")
    ax.set_title("Target vs Minority\nUnder Rescuing Context")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seeds(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model} on {args.device}")
    model = tl.HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()

    results = []
    for item in OVERRIDE_ITEMS:
        print(f"  Testing: {item['modifier']} → {item['rescued_minority']} (vs {item['target_head']})")
        result = run_item(model, item, args.device)
        if result is not None:
            results.append(result)
            status = "RESCUED" if result["rescue_success"] else "still dominated"
            print(f"    Override strength: {result['override_strength_log10']:.2f} — {status}")
        else:
            print(f"    Skipped (multi-token)")

    summary = summarize(results)

    safe_name = args.model.replace("/", "_").replace("-", "_")
    output = {"model": args.model, "summary": summary, "results": results}

    out_path = args.results_dir / f"contextual_override_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_path = args.results_dir / "plots" / f"contextual_override_{safe_name}.png"
    plot_results(results, plot_path)

    print("\n" + "=" * 60)
    print("CONTEXTUAL OVERRIDE SUMMARY")
    print("=" * 60)
    print(f"Items tested: {summary['n_items']}")
    print(f"Rescue successes: {summary['n_rescue_success']} / {summary['n_items']}"
          f" ({summary['rescue_success_rate']:.0%})")
    print(f"Mean override strength: {summary['mean_override_strength']:.3f}")
    print(f"Minority prob change (neutral → rescue): "
          f"{summary['minority_prob_change_pct']:+.1f}%")
    print(f"Target prob change (neutral → rescue): "
          f"{summary['target_prob_change_pct']:+.1f}%")
    if summary["wilcoxon_p"] is not None:
        print(f"Wilcoxon (rescue > neutral minority): "
              f"W={summary['wilcoxon_stat']:.0f}, p={summary['wilcoxon_p']:.4f}")
    print(f"\nSaved to: {out_path}")
    print(f"Plot: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
