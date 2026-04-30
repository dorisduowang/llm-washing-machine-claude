"""
Validation experiment: Run key analyses on GPT-2-medium (355M, 24 layers)
to check if findings from GPT-2 (124M, 12 layers) generalize.

Focus on:
1. Next-token prediction for key compounds
2. R² reconstruction at final layer
"""

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import transformer_lens as tl

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda:0"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

KEY_COMPOUNDS = [
    ("washing machine", "washing", "machine", 4, "red machine"),
    ("swimming pool", "swimming", "pool", 4, "deep pool"),
    ("hot dog", "hot", "dog", 1, "big dog"),
    ("guinea pig", "guinea", "pig", 2, "small pig"),
    ("coffee table", "coffee", "table", 5, "wooden table"),
    ("parking lot", "parking", "lot", 4, "empty lot"),
    ("living room", "living", "room", 3, "large room"),
    ("mountain cabin", "mountain", "cabin", 5, "small cabin"),
]

TEMPLATES = [
    "The {compound} was",
    "She bought a {compound} for",
    "I saw a {compound} in the",
    "There is a {compound} near the",
]

ISOLATION_TEMPLATES = [
    "The {word} was very",
    "I noticed the {word} seemed",
    "That {word} is quite",
    "A {word} appeared in the",
]


def get_token_ids(model, word):
    tokens = model.to_tokens(f" {word}", prepend_bos=False)
    return tokens[0].tolist()


def main():
    print("Loading GPT-2-medium...")
    model = tl.HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"Model: {n_layers} layers, d_model={model.cfg.d_model}")

    results = []

    for compound, word1, word2, comp_rating, control in KEY_COMPOUNDS:
        w2_tokens = get_token_ids(model, word2)
        if len(w2_tokens) != 1:
            continue
        w2_id = w2_tokens[0]

        # Next-token prediction
        compound_probs = []
        control_word1 = control.split()[0]
        control_probs = []

        for template in TEMPLATES:
            # Compound
            prompt = template.split("{compound}")[0] + word1
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1], dim=-1)
            compound_probs.append(probs[w2_id].item())

            # Control
            prompt = template.split("{compound}")[0] + control_word1
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1], dim=-1)
            control_probs.append(probs[w2_id].item())

        # Residual stream direction analysis (final layer only)
        compound_states = []
        word1_states = []
        word2_states = []

        for template in TEMPLATES:
            sentence = template.format(compound=compound)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            token_strs = [model.to_string([t]) for t in tokens[0]]
            w2_str = f" {word2}"
            w2_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w2_str and pos > 0 and token_strs[pos-1].strip() == word1:
                    w2_pos = pos
                    break
            if w2_pos is not None:
                h = cache[f"blocks.{n_layers-1}.hook_resid_post"][0, w2_pos].cpu().numpy()
                compound_states.append(h)

        for template in ISOLATION_TEMPLATES:
            sentence = template.format(word=word1)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            token_strs = [model.to_string([t]) for t in tokens[0]]
            w1_str = f" {word1}"
            for pos, ts in enumerate(token_strs):
                if ts == w1_str:
                    h = cache[f"blocks.{n_layers-1}.hook_resid_post"][0, pos].cpu().numpy()
                    word1_states.append(h)
                    break

        for template in ISOLATION_TEMPLATES:
            sentence = template.format(word=word2)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            token_strs = [model.to_string([t]) for t in tokens[0]]
            w2_str = f" {word2}"
            for pos, ts in enumerate(token_strs):
                if ts == w2_str:
                    h = cache[f"blocks.{n_layers-1}.hook_resid_post"][0, pos].cpu().numpy()
                    word2_states.append(h)
                    break

        # Compute R²
        r_squared = None
        if compound_states and word1_states and word2_states:
            compound_dir = np.mean(compound_states, axis=0)
            word1_dir = np.mean(word1_states, axis=0)
            word2_dir = np.mean(word2_states, axis=0)
            X = np.stack([word1_dir, word2_dir], axis=0).T
            y = compound_dir
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        p_comp = np.mean(compound_probs)
        p_ctrl = np.mean(control_probs)
        boost = p_comp / max(p_ctrl, 1e-10)

        result = {
            "compound": compound,
            "compositionality": comp_rating,
            "p_word2_compound": float(p_comp),
            "p_word2_control": float(p_ctrl),
            "boost_ratio": float(boost),
            "r_squared_final": float(r_squared) if r_squared else None,
        }
        results.append(result)

        print(f"  {compound:<22} P(w2|w1)={p_comp:.4f} boost={boost:.1f}x "
              f"R²={r_squared:.3f}" if r_squared else f"  {compound:<22} P(w2|w1)={p_comp:.4f}")

    with open(RESULTS_DIR / "validation_gpt2medium.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compare with GPT-2 results
    print("\n--- Comparison: GPT-2 vs GPT-2-medium ---")
    with open(RESULTS_DIR / "exp1_next_token.json") as f:
        gpt2_results = json.load(f)

    gpt2_lookup = {r["compound"]: r for r in gpt2_results}

    print(f"\n{'Compound':<22} {'GPT-2 Boost':<14} {'GPT-2-M Boost':<14} {'GPT-2 R²':<10} {'GPT-2-M R²':<10}")
    print("-" * 70)

    with open(RESULTS_DIR / "exp2_residual_directions.json") as f:
        gpt2_dir = json.load(f)
    gpt2_dir_lookup = {r["compound"]: r for r in gpt2_dir}

    for r in results:
        name = r["compound"]
        gpt2_boost = gpt2_lookup[name]["boost_ratio"] if name in gpt2_lookup else None
        gpt2_r2 = None
        if name in gpt2_dir_lookup:
            last = gpt2_dir_lookup[name]["layer_metrics"][-1]
            if last:
                gpt2_r2 = last["reconstruction_r2"]

        gpt2_boost_str = f"{gpt2_boost:.1f}" if gpt2_boost else "N/A"
        medium_boost_str = f"{r['boost_ratio']:.1f}"
        gpt2_r2_str = f"{gpt2_r2:.3f}" if gpt2_r2 else "N/A"
        medium_r2_str = f"{r['r_squared_final']:.3f}" if r['r_squared_final'] else "N/A"

        print(f"{name:<22} {gpt2_boost_str:<14} {medium_boost_str:<14} {gpt2_r2_str:<10} {medium_r2_str:<10}")

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
