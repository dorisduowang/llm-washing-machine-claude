"""
Minority-continuation analysis for compound concepts.

The original next-token experiment shows that GPT-2 strongly predicts the
head of familiar compounds such as "machine" after "washing". This script
tests a stricter question from the project roadmap: does a high-PMI compound
head crowd out other plausible heads after the same modifier, such as
"washing method" or "washing powder"?

Outputs:
  - results/minority_continuations_<model>.json
  - results/plots/minority_continuation_<model>.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
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
DEFAULT_DATASET = REPO_ROOT / "datasets" / "compound_concepts" / "minority_continuations.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class HeadScore:
    head: str
    token_id: int
    probability: float
    rank: int
    prompts: list[dict[str, Any]]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_single_token_id(model: tl.HookedTransformer, word: str) -> int | None:
    """Return the GPT-style token id for a leading-space word, or None."""
    token_ids = model.to_tokens(f" {word}", prepend_bos=False)[0].tolist()
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def score_head(
    model: tl.HookedTransformer,
    prompts: list[str],
    head: str,
    token_id: int,
    device: str,
) -> HeadScore:
    prompt_scores: list[dict[str, Any]] = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(device)
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1], dim=-1)
        probability = float(probs[token_id].item())
        rank = int((probs > probs[token_id]).sum().item() + 1)
        prompt_scores.append(
            {
                "prompt": prompt,
                "probability": probability,
                "rank": rank,
            }
        )

    return HeadScore(
        head=head,
        token_id=token_id,
        probability=float(np.mean([p["probability"] for p in prompt_scores])),
        rank=int(round(float(np.median([p["rank"] for p in prompt_scores])))),
        prompts=prompt_scores,
    )


def top_predictions(
    model: tl.HookedTransformer,
    prompt: str,
    device: str,
    top_k: int,
) -> list[dict[str, Any]]:
    tokens = model.to_tokens(prompt).to(device)
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1], dim=-1)
    values, token_ids = torch.topk(probs, top_k)
    return [
        {
            "token": model.to_string([int(token_id)]),
            "token_id": int(token_id),
            "probability": float(value),
        }
        for value, token_id in zip(values.tolist(), token_ids.tolist())
    ]


def analyze_item(
    model: tl.HookedTransformer,
    item: dict[str, Any],
    prompt_templates: list[str],
    device: str,
    top_k: int,
) -> dict[str, Any]:
    modifier = item["modifier"]
    prompts = [template.format(modifier=modifier) for template in prompt_templates]

    skipped_heads = []
    target_token_id = get_single_token_id(model, item["target_head"])
    if target_token_id is None:
        skipped_heads.append({"head": item["target_head"], "reason": "multi_token"})
        target_score = None
    else:
        target_score = score_head(model, prompts, item["target_head"], target_token_id, device)

    minority_scores = []
    for head in item["minority_heads"]:
        token_id = get_single_token_id(model, head)
        if token_id is None:
            skipped_heads.append({"head": head, "reason": "multi_token"})
            continue
        minority_scores.append(score_head(model, prompts, head, token_id, device))

    unrelated_scores = []
    for head in item["unrelated_heads"]:
        token_id = get_single_token_id(model, head)
        if token_id is None:
            skipped_heads.append({"head": head, "reason": "multi_token"})
            continue
        unrelated_scores.append(score_head(model, prompts, head, token_id, device))

    best_minority = max(minority_scores, key=lambda s: s.probability, default=None)
    best_unrelated = max(unrelated_scores, key=lambda s: s.probability, default=None)

    if target_score is not None and best_minority is not None:
        target_vs_best_minority_ratio = target_score.probability / max(best_minority.probability, EPS)
        log10_target_margin = math.log10(target_score.probability + EPS) - math.log10(
            best_minority.probability + EPS
        )
        minority_heads_beating_target = [
            s.head for s in minority_scores if s.probability > target_score.probability
        ]
    else:
        target_vs_best_minority_ratio = None
        log10_target_margin = None
        minority_heads_beating_target = []

    if best_minority is not None and best_unrelated is not None:
        minority_vs_unrelated_ratio = best_minority.probability / max(best_unrelated.probability, EPS)
    else:
        minority_vs_unrelated_ratio = None

    return {
        "compound": item["compound"],
        "modifier": modifier,
        "category": item.get("category"),
        "prompts": prompts,
        "target": target_score.__dict__ if target_score else None,
        "minority_heads": [score.__dict__ for score in minority_scores],
        "unrelated_heads": [score.__dict__ for score in unrelated_scores],
        "best_minority_head": best_minority.__dict__ if best_minority else None,
        "best_unrelated_head": best_unrelated.__dict__ if best_unrelated else None,
        "target_vs_best_minority_ratio": target_vs_best_minority_ratio,
        "log10_target_margin_over_best_minority": log10_target_margin,
        "minority_heads_beating_target": minority_heads_beating_target,
        "minority_vs_unrelated_ratio": minority_vs_unrelated_ratio,
        "top_predictions_after_first_prompt": top_predictions(model, prompts[0], device, top_k),
        "skipped_heads": skipped_heads,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    analyzable = [r for r in results if r["target"] and r["best_minority_head"]]
    target_probs = np.array([r["target"]["probability"] for r in analyzable])
    minority_probs = np.array([r["best_minority_head"]["probability"] for r in analyzable])
    margins = np.array([r["log10_target_margin_over_best_minority"] for r in analyzable])
    ratios = np.array([r["target_vs_best_minority_ratio"] for r in analyzable])
    minority_control_ratios = np.array(
        [
            r["minority_vs_unrelated_ratio"]
            for r in analyzable
            if r["minority_vs_unrelated_ratio"] is not None
        ]
    )

    return {
        "n_items": len(results),
        "n_analyzable": len(analyzable),
        "target_mean_probability": float(target_probs.mean()) if len(target_probs) else None,
        "best_minority_mean_probability": float(minority_probs.mean()) if len(minority_probs) else None,
        "median_target_vs_best_minority_ratio": float(np.median(ratios)) if len(ratios) else None,
        "median_log10_target_margin_over_best_minority": float(np.median(margins)) if len(margins) else None,
        "n_items_where_minority_beats_target": int(
            sum(bool(r["minority_heads_beating_target"]) for r in analyzable)
        ),
        "median_best_minority_vs_best_unrelated_ratio": float(np.median(minority_control_ratios))
        if len(minority_control_ratios)
        else None,
    }


def plot_results(results: list[dict[str, Any]], output_path: Path) -> None:
    analyzable = [r for r in results if r["target"] and r["best_minority_head"]]
    if not analyzable:
        return

    labels = [r["compound"] for r in analyzable]
    target_probs = np.array([r["target"]["probability"] for r in analyzable])
    minority_probs = np.array([r["best_minority_head"]["probability"] for r in analyzable])
    best_minority_labels = [r["best_minority_head"]["head"] for r in analyzable]
    order = np.argsort(target_probs / np.maximum(minority_probs, EPS))[::-1]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(order))
    ax.barh(y - 0.18, target_probs[order], height=0.36, label="target compound head")
    ax.barh(y + 0.18, minority_probs[order], height=0.36, label="best minority head")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{labels[i]}\nvs {best_minority_labels[i]}" for i in order],
        fontsize=8,
    )
    ax.set_xlabel("Mean next-token probability across prompts")
    ax.set_title("Target Compound Head vs. Strongest Minority Continuation")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="TransformerLens model name, e.g. gpt2 or gpt2-medium")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seeds(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)

    print(f"Loading model: {args.model} on {args.device}")
    model = tl.HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()

    results = [
        analyze_item(
            model=model,
            item=item,
            prompt_templates=dataset["prompt_templates"],
            device=args.device,
            top_k=args.top_k,
        )
        for item in dataset["items"]
    ]
    summary = summarize(results)

    safe_model_name = args.model.replace("/", "_").replace("-", "_")
    output = {
        "model": args.model,
        "device": args.device,
        "dataset": repo_relative_path(args.dataset),
        "summary": summary,
        "results": results,
    }
    output_path = args.results_dir / f"minority_continuations_{safe_model_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_path = plots_dir / f"minority_continuation_{safe_model_name}.png"
    plot_results(results, plot_path)

    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
