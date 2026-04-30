"""
Experiment: Where is "Washing Machine" Stored in LLMs?

Three experiments testing how compound concepts are represented:
1. Next-token prediction analysis
2. Residual stream direction analysis
3. Layer-wise probing for compound concept emergence

Uses GPT-2 (124M) via TransformerLens for interpretability access.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import transformer_lens as tl

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────

# Two-token compound nouns with compositionality ratings and control phrases
# Each entry: (compound, word1, word2, compositionality, control_phrase)
# Control phrases use the same word2 but a different, non-compound word1
COMPOUNDS = [
    ("washing machine", "washing", "machine", 4, "red machine"),
    ("coffee table", "coffee", "table", 5, "wooden table"),
    ("swimming pool", "swimming", "pool", 4, "deep pool"),
    ("parking lot", "parking", "lot", 4, "empty lot"),
    ("hot dog", "hot", "dog", 1, "big dog"),
    ("shooting star", "shooting", "star", 2, "bright star"),
    ("living room", "living", "room", 3, "large room"),
    ("driving license", "driving", "license", 4, "new license"),
    ("guinea pig", "guinea", "pig", 2, "small pig"),
    ("brick house", "brick", "house", 5, "old house"),
    ("mountain cabin", "mountain", "cabin", 5, "small cabin"),
    ("garden hose", "garden", "hose", 5, "long hose"),
    ("water bottle", "water", "bottle", 5, "glass bottle"),
    ("steel bridge", "steel", "bridge", 5, "old bridge"),
    ("chocolate cake", "chocolate", "cake", 5, "big cake"),
    ("door handle", "door", "handle", 5, "metal handle"),
    ("blueberry", "blue", "berry", 4, "small berry"),
    ("snowman", "snow", "man", 4, "tall man"),
    ("sunflower", "sun", "flower", 4, "red flower"),
]

# Sentence templates for generating diverse contexts
TEMPLATES = [
    "The {compound} was",
    "She bought a {compound} for",
    "I saw a {compound} in the",
    "There is a {compound} near the",
    "He fixed the {compound} with",
    "A new {compound} arrived",
    "The old {compound} needed",
    "We need a {compound} to",
]

# Additional contexts for word-in-isolation analysis
ISOLATION_TEMPLATES = [
    "The {word} was very",
    "I noticed the {word} seemed",
    "That {word} is quite",
    "A {word} appeared in the",
]


def load_model(model_name="gpt2"):
    """Load GPT-2 model via TransformerLens with hooks."""
    print(f"Loading model: {model_name}")
    model = tl.HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model.eval()
    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def get_token_ids(model, word):
    """Get the token ID for a word (with leading space)."""
    tokens = model.to_tokens(f" {word}", prepend_bos=False)
    return tokens[0].tolist()


# ── Experiment 1: Next-Token Prediction Analysis ─────────────────────────

def experiment1_next_token_prediction(model):
    """
    Test whether 'washing' alone makes 'machine' more likely.
    Compare P(word2 | word1) across compositionality levels.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Next-Token Prediction Analysis")
    print("="*70)

    results = []

    for compound, word1, word2, comp_rating, control in tqdm(COMPOUNDS, desc="Compounds"):
        # Get target token id for word2
        word2_tokens = get_token_ids(model, word2)
        if len(word2_tokens) != 1:
            print(f"  Skipping {compound}: word2 '{word2}' is multi-token")
            continue
        word2_id = word2_tokens[0]

        # 1. P(word2 | "The word1")  — compound context
        compound_probs = []
        compound_ranks = []
        for template in TEMPLATES:
            prompt = template.split("{compound}")[0] + word1
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            p_word2 = probs[word2_id].item()
            rank = (probs > probs[word2_id]).sum().item() + 1
            compound_probs.append(p_word2)
            compound_ranks.append(rank)

        # 2. P(word2 | "The control_word1") — control context
        control_word1 = control.split()[0]
        control_probs = []
        control_ranks = []
        for template in TEMPLATES:
            prompt = template.split("{compound}")[0] + control_word1
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            p_word2 = probs[word2_id].item()
            rank = (probs > probs[word2_id]).sum().item() + 1
            control_probs.append(p_word2)
            control_ranks.append(rank)

        # 3. P(word2 | unconditional) — just BOS token
        bos_tokens = model.to_tokens("The")
        with torch.no_grad():
            logits = model(bos_tokens)
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits, dim=-1)
        p_word2_uncond = probs[word2_id].item()
        rank_uncond = (probs > probs[word2_id]).sum().item() + 1

        # 4. Get top-5 predictions after word1 for interpretability
        tokens = model.to_tokens(f"The {word1}")
        with torch.no_grad():
            logits = model(tokens)
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits, dim=-1)
        top5_ids = torch.topk(probs, 5).indices.tolist()
        top5 = [(model.to_string([tid]).strip(), probs[tid].item()) for tid in top5_ids]

        result = {
            "compound": compound,
            "word1": word1,
            "word2": word2,
            "compositionality": comp_rating,
            "p_word2_compound_mean": np.mean(compound_probs),
            "p_word2_compound_std": np.std(compound_probs),
            "rank_compound_mean": np.mean(compound_ranks),
            "p_word2_control_mean": np.mean(control_probs),
            "p_word2_control_std": np.std(control_probs),
            "rank_control_mean": np.mean(control_ranks),
            "p_word2_unconditional": p_word2_uncond,
            "rank_unconditional": rank_uncond,
            "boost_ratio": np.mean(compound_probs) / max(np.mean(control_probs), 1e-10),
            "top5_after_word1": top5,
        }
        results.append(result)

        print(f"  {compound:20s} | P(w2|w1)={result['p_word2_compound_mean']:.4f} "
              f"rank={result['rank_compound_mean']:.0f} | "
              f"P(w2|ctrl)={result['p_word2_control_mean']:.4f} "
              f"rank={result['rank_control_mean']:.0f} | "
              f"boost={result['boost_ratio']:.1f}x | "
              f"top5: {[t[0] for t in top5]}")

    # Save results
    with open(RESULTS_DIR / "exp1_next_token.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_experiment1(results):
    """Visualize Experiment 1 results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    compounds = [r["compound"] for r in results]
    comp_ratings = [r["compositionality"] for r in results]
    p_compound = [r["p_word2_compound_mean"] for r in results]
    p_control = [r["p_word2_control_mean"] for r in results]
    boost_ratios = [r["boost_ratio"] for r in results]

    # Sort by boost ratio for bar chart
    sorted_idx = np.argsort(boost_ratios)[::-1]

    # Plot 1: Boost ratio bar chart
    ax = axes[0]
    bars = ax.barh(range(len(compounds)),
                   [boost_ratios[i] for i in sorted_idx],
                   color=['#e74c3c' if comp_ratings[i] <= 2 else
                          '#f39c12' if comp_ratings[i] == 3 else
                          '#2ecc71' for i in sorted_idx])
    ax.set_yticks(range(len(compounds)))
    ax.set_yticklabels([compounds[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel("Boost Ratio: P(w2|w1) / P(w2|control)")
    ax.set_title("How Much Does Word1 Boost Word2 Prediction?")
    ax.axvline(x=1, color='k', linestyle='--', alpha=0.5)
    ax.invert_yaxis()

    # Plot 2: Compound prob vs control prob scatter
    ax = axes[1]
    scatter = ax.scatter(p_control, p_compound, c=comp_ratings,
                        cmap='RdYlGn', s=80, edgecolors='k', linewidth=0.5,
                        vmin=1, vmax=5)
    ax.plot([0, max(max(p_compound), max(p_control))],
            [0, max(max(p_compound), max(p_control))],
            'k--', alpha=0.3, label='y=x')
    for i, c in enumerate(compounds):
        if boost_ratios[i] > 5 or p_compound[i] > 0.05:
            ax.annotate(c, (p_control[i], p_compound[i]),
                       fontsize=6, ha='left', va='bottom')
    ax.set_xlabel("P(word2 | control word)")
    ax.set_ylabel("P(word2 | compound word1)")
    ax.set_title("Compound vs Control Next-Token Probability")
    plt.colorbar(scatter, ax=ax, label="Compositionality Rating")

    # Plot 3: Compositionality vs boost ratio
    ax = axes[2]
    ax.scatter(comp_ratings, boost_ratios, s=80, c='steelblue', edgecolors='k')
    for i, c in enumerate(compounds):
        ax.annotate(c, (comp_ratings[i], boost_ratios[i]),
                   fontsize=6, ha='left', va='bottom', rotation=15)
    r, p = stats.spearmanr(comp_ratings, boost_ratios)
    ax.set_xlabel("Compositionality Rating (1=idiomatic, 5=compositional)")
    ax.set_ylabel("Boost Ratio")
    ax.set_title(f"Compositionality vs Prediction Boost\n(Spearman r={r:.3f}, p={p:.3f})")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_next_token_prediction.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp1_next_token_prediction.png'}")


# ── Experiment 2: Residual Stream Direction Analysis ─────────────────────

def experiment2_residual_directions(model):
    """
    Compare directions for compound concepts vs. constituent words.
    Test: is the compound direction a linear combination of constituent directions?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Residual Stream Direction Analysis")
    print("="*70)

    n_layers = model.cfg.n_layers
    results = []

    for compound, word1, word2, comp_rating, control in tqdm(COMPOUNDS, desc="Compounds"):
        # Verify both words are single tokens
        w1_toks = get_token_ids(model, word1)
        w2_toks = get_token_ids(model, word2)
        if len(w1_toks) != 1 or len(w2_toks) != 1:
            print(f"  Skipping {compound}: multi-token constituent")
            continue

        # Collect hidden states across contexts
        compound_states = {l: [] for l in range(n_layers)}
        word1_states = {l: [] for l in range(n_layers)}
        word2_states = {l: [] for l in range(n_layers)}
        control_states = {l: [] for l in range(n_layers)}

        # Compound contexts: hidden state at word2 position
        for template in TEMPLATES:
            sentence = template.format(compound=compound)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            # Find position of word2 in tokenized sentence
            token_strs = [model.to_string([t]) for t in tokens[0]]
            # word2 with space prefix
            w2_str = f" {word2}"
            w2_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w2_str:
                    # Check that previous token is word1
                    if pos > 0 and token_strs[pos-1].strip() == word1:
                        w2_pos = pos
                        break
            if w2_pos is None:
                continue

            for layer in range(n_layers):
                h = cache[f"blocks.{layer}.hook_resid_post"][0, w2_pos].cpu().numpy()
                compound_states[layer].append(h)

        # Word1 in isolation contexts
        for template in ISOLATION_TEMPLATES:
            sentence = template.format(word=word1)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            token_strs = [model.to_string([t]) for t in tokens[0]]
            w1_str = f" {word1}"
            w1_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w1_str:
                    w1_pos = pos
                    break
            if w1_pos is None:
                continue
            for layer in range(n_layers):
                h = cache[f"blocks.{layer}.hook_resid_post"][0, w1_pos].cpu().numpy()
                word1_states[layer].append(h)

        # Word2 in isolation contexts
        for template in ISOLATION_TEMPLATES:
            sentence = template.format(word=word2)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            token_strs = [model.to_string([t]) for t in tokens[0]]
            w2_str = f" {word2}"
            w2_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w2_str:
                    w2_pos = pos
                    break
            if w2_pos is None:
                continue
            for layer in range(n_layers):
                h = cache[f"blocks.{layer}.hook_resid_post"][0, w2_pos].cpu().numpy()
                word2_states[layer].append(h)

        # Control contexts: hidden state at word2 position in non-compound context
        control_word1 = control.split()[0]
        for template in TEMPLATES:
            sentence = template.format(compound=control)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            token_strs = [model.to_string([t]) for t in tokens[0]]
            w2_str = f" {word2}"
            w2_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w2_str:
                    w2_pos = pos
                    break
            if w2_pos is None:
                continue
            for layer in range(n_layers):
                h = cache[f"blocks.{layer}.hook_resid_post"][0, w2_pos].cpu().numpy()
                control_states[layer].append(h)

        # Compute metrics per layer
        layer_metrics = []
        for layer in range(n_layers):
            if not compound_states[layer] or not word1_states[layer] or not word2_states[layer]:
                layer_metrics.append(None)
                continue

            # Mean directions
            compound_dir = np.mean(compound_states[layer], axis=0)
            word1_dir = np.mean(word1_states[layer], axis=0)
            word2_dir = np.mean(word2_states[layer], axis=0)

            # Cosine similarities
            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

            sim_compound_w1 = cosine_sim(compound_dir, word1_dir)
            sim_compound_w2 = cosine_sim(compound_dir, word2_dir)
            sim_w1_w2 = cosine_sim(word1_dir, word2_dir)

            # Linear reconstruction: compound = alpha*word1 + beta*word2 + bias
            X = np.stack([word1_dir, word2_dir], axis=0).T  # d_model x 2
            y = compound_dir
            # Least squares: (X^T X)^{-1} X^T y
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ coeffs
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            except np.linalg.LinAlgError:
                coeffs = np.array([0, 0])
                r_squared = 0

            # Control comparison
            if control_states[layer]:
                control_dir = np.mean(control_states[layer], axis=0)
                sim_control_w2 = cosine_sim(control_dir, word2_dir)
                sim_compound_control = cosine_sim(compound_dir, control_dir)
            else:
                sim_control_w2 = None
                sim_compound_control = None

            # Uniqueness: residual after removing word1 and word2 components
            residual = compound_dir - y_pred
            residual_norm_ratio = np.linalg.norm(residual) / (np.linalg.norm(compound_dir) + 1e-10)

            layer_metrics.append({
                "layer": layer,
                "sim_compound_word1": float(sim_compound_w1),
                "sim_compound_word2": float(sim_compound_w2),
                "sim_word1_word2": float(sim_w1_w2),
                "reconstruction_r2": float(r_squared),
                "alpha_word1": float(coeffs[0]),
                "beta_word2": float(coeffs[1]),
                "residual_norm_ratio": float(residual_norm_ratio),
                "sim_control_word2": float(sim_control_w2) if sim_control_w2 is not None else None,
                "sim_compound_control": float(sim_compound_control) if sim_compound_control is not None else None,
            })

        result = {
            "compound": compound,
            "word1": word1,
            "word2": word2,
            "compositionality": comp_rating,
            "n_compound_contexts": len(compound_states[0]),
            "n_word1_contexts": len(word1_states[0]),
            "n_word2_contexts": len(word2_states[0]),
            "layer_metrics": layer_metrics,
        }
        results.append(result)

        # Print summary for last layer
        last = layer_metrics[-1]
        if last:
            print(f"  {compound:20s} | cos(C,w1)={last['sim_compound_word1']:.3f} "
                  f"cos(C,w2)={last['sim_compound_word2']:.3f} "
                  f"R²={last['reconstruction_r2']:.3f} "
                  f"residual={last['residual_norm_ratio']:.3f}")

    with open(RESULTS_DIR / "exp2_residual_directions.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_experiment2(results):
    """Visualize Experiment 2 results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    n_layers = 12  # GPT-2

    # Plot 1: R² across layers (all compounds)
    ax = axes[0, 0]
    for r in results:
        layers = []
        r2_vals = []
        for m in r["layer_metrics"]:
            if m is not None:
                layers.append(m["layer"])
                r2_vals.append(m["reconstruction_r2"])
        color = plt.cm.RdYlGn(r["compositionality"] / 5.0)
        ax.plot(layers, r2_vals, alpha=0.6, color=color, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (compound ≈ α·word1 + β·word2)")
    ax.set_title("Linear Reconstruction Quality Across Layers")
    ax.set_ylim(-0.1, 1.1)
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(1, 5))
    plt.colorbar(sm, ax=ax, label="Compositionality")

    # Plot 2: Cosine similarity (compound vs word2) across layers
    ax = axes[0, 1]
    for r in results:
        layers = []
        sims = []
        for m in r["layer_metrics"]:
            if m is not None:
                layers.append(m["layer"])
                sims.append(m["sim_compound_word2"])
        color = plt.cm.RdYlGn(r["compositionality"] / 5.0)
        ax.plot(layers, sims, alpha=0.6, color=color, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity: Compound Direction vs Word2 Direction")

    # Plot 3: Residual norm ratio across layers
    ax = axes[1, 0]
    for r in results:
        layers = []
        residuals = []
        for m in r["layer_metrics"]:
            if m is not None:
                layers.append(m["layer"])
                residuals.append(m["residual_norm_ratio"])
        color = plt.cm.RdYlGn(r["compositionality"] / 5.0)
        ax.plot(layers, residuals, alpha=0.6, color=color, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||residual|| / ||compound||")
    ax.set_title("Unique Component (Not Explained by Constituents)")

    # Plot 4: Final layer R² vs compositionality
    ax = axes[1, 1]
    comps = []
    r2_finals = []
    labels = []
    for r in results:
        last = r["layer_metrics"][-1]
        if last:
            comps.append(r["compositionality"])
            r2_finals.append(last["reconstruction_r2"])
            labels.append(r["compound"])
    ax.scatter(comps, r2_finals, s=80, c='steelblue', edgecolors='k')
    for i, label in enumerate(labels):
        ax.annotate(label, (comps[i], r2_finals[i]), fontsize=6,
                   ha='left', va='bottom', rotation=15)
    r_corr, p_val = stats.spearmanr(comps, r2_finals)
    ax.set_xlabel("Compositionality Rating")
    ax.set_ylabel("Final Layer R²")
    ax.set_title(f"Compositionality vs Reconstruction Quality\n(Spearman r={r_corr:.3f}, p={p_val:.3f})")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_residual_directions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp2_residual_directions.png'}")


# ── Experiment 3: Layer-wise Probing ─────────────────────────────────────

def experiment3_layerwise_probing(model):
    """
    Train probes to test where compound concept representations emerge.

    Probe 1: At the word2 position, can we predict what word1 was?
             (Tests token erasure: if accuracy drops, word1 info is being erased)
    Probe 2: Can we distinguish compound vs. non-compound contexts at the word2 position?
             (Tests whether compounds create a unique representation)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Layer-wise Probing for Compound Concept Emergence")
    print("="*70)

    n_layers = model.cfg.n_layers

    # Collect data: hidden states at the word2 position
    # Label 1: which word1 preceded (for erasure analysis)
    # Label 2: compound (1) vs control (0) context

    compound_data = []  # (hidden_states_per_layer, word1_label, compound_label=1, compound_name)
    control_data = []   # (hidden_states_per_layer, word1_label, compound_label=0, compound_name)

    word1_to_idx = {}
    idx_counter = 0

    for compound, word1, word2, comp_rating, control in tqdm(COMPOUNDS, desc="Collecting activations"):
        w1_toks = get_token_ids(model, word1)
        w2_toks = get_token_ids(model, word2)
        if len(w1_toks) != 1 or len(w2_toks) != 1:
            continue

        if word1 not in word1_to_idx:
            word1_to_idx[word1] = idx_counter
            idx_counter += 1
        w1_idx = word1_to_idx[word1]

        control_word1 = control.split()[0]
        if control_word1 not in word1_to_idx:
            word1_to_idx[control_word1] = idx_counter
            idx_counter += 1
        ctrl_w1_idx = word1_to_idx[control_word1]

        # Compound contexts
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
            if w2_pos is None:
                continue

            states = {}
            for layer in range(n_layers):
                states[layer] = cache[f"blocks.{layer}.hook_resid_post"][0, w2_pos].cpu().numpy()
            compound_data.append((states, w1_idx, 1, compound, comp_rating))

        # Control contexts
        for template in TEMPLATES:
            sentence = template.format(compound=control)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            token_strs = [model.to_string([t]) for t in tokens[0]]
            w2_str = f" {word2}"
            w2_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w2_str:
                    w2_pos = pos
                    break
            if w2_pos is None:
                continue

            states = {}
            for layer in range(n_layers):
                states[layer] = cache[f"blocks.{layer}.hook_resid_post"][0, w2_pos].cpu().numpy()
            control_data.append((states, ctrl_w1_idx, 0, control, comp_rating))

    print(f"\nCollected {len(compound_data)} compound samples, {len(control_data)} control samples")
    print(f"Number of unique word1 labels: {len(word1_to_idx)}")

    # ── Probe 1: Word1 identity prediction (erasure analysis) ──
    # Only on compound data: can we tell what word1 was from the word2 position?
    print("\nProbe 1: Word1 Identity Prediction (Token Erasure)")

    probe1_results = {"layer": [], "accuracy": [], "accuracy_std": []}

    for layer in range(n_layers):
        X = np.array([d[0][layer] for d in compound_data])
        y = np.array([d[1] for d in compound_data])

        if len(np.unique(y)) < 2:
            probe1_results["layer"].append(layer)
            probe1_results["accuracy"].append(0)
            probe1_results["accuracy_std"].append(0)
            continue

        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        scores = cross_val_score(clf, X, y, cv=min(5, len(X)), scoring='accuracy')
        probe1_results["layer"].append(layer)
        probe1_results["accuracy"].append(scores.mean())
        probe1_results["accuracy_std"].append(scores.std())

        print(f"  Layer {layer:2d}: accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

    # ── Probe 2: Compound vs. Control Classification ──
    print("\nProbe 2: Compound vs. Control Context Classification")

    all_data = compound_data + control_data
    probe2_results = {"layer": [], "accuracy": [], "accuracy_std": []}

    for layer in range(n_layers):
        X = np.array([d[0][layer] for d in all_data])
        y = np.array([d[2] for d in all_data])

        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        scores = cross_val_score(clf, X, y, cv=min(5, len(X)), scoring='accuracy')
        probe2_results["layer"].append(layer)
        probe2_results["accuracy"].append(scores.mean())
        probe2_results["accuracy_std"].append(scores.std())

        print(f"  Layer {layer:2d}: accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

    # ── Probe 3: Per-compositionality-level compound vs. control ──
    print("\nProbe 3: Compound vs. Control by Compositionality Level")

    comp_levels = sorted(set(r for _, _, _, _, r in compound_data))
    probe3_results = {level: {"layer": [], "accuracy": []} for level in comp_levels}

    for level in comp_levels:
        level_compound = [d for d in compound_data if d[4] == level]
        level_control = [d for d in control_data if d[4] == level]
        level_data = level_compound + level_control

        if len(level_data) < 10:
            continue

        for layer in range(n_layers):
            X = np.array([d[0][layer] for d in level_data])
            y = np.array([d[2] for d in level_data])

            if len(np.unique(y)) < 2:
                continue

            clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
            cv_folds = min(3, min(np.sum(y==0), np.sum(y==1)))
            if cv_folds < 2:
                continue
            scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy')
            probe3_results[level]["layer"].append(layer)
            probe3_results[level]["accuracy"].append(scores.mean())

    results = {
        "probe1_word1_identity": probe1_results,
        "probe2_compound_vs_control": probe2_results,
        "probe3_by_compositionality": {str(k): v for k, v in probe3_results.items()},
        "n_compound_samples": len(compound_data),
        "n_control_samples": len(control_data),
        "n_word1_classes": len(word1_to_idx),
    }

    with open(RESULTS_DIR / "exp3_probing.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_experiment3(results):
    """Visualize Experiment 3 results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Word1 identity probe accuracy across layers (erasure)
    ax = axes[0]
    p1 = results["probe1_word1_identity"]
    ax.errorbar(p1["layer"], p1["accuracy"], yerr=p1["accuracy_std"],
                marker='o', capsize=3, color='steelblue', linewidth=2)
    chance = 1.0 / results["n_word1_classes"]
    ax.axhline(y=chance, color='red', linestyle='--', alpha=0.5, label=f'Chance ({chance:.2f})')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe 1: Can We Predict Word1 from Word2 Position?\n(Token Erasure Test)")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Plot 2: Compound vs control classification
    ax = axes[1]
    p2 = results["probe2_compound_vs_control"]
    ax.errorbar(p2["layer"], p2["accuracy"], yerr=p2["accuracy_std"],
                marker='s', capsize=3, color='darkorange', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (0.50)')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe 2: Can We Distinguish Compound vs. Control Context?")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Plot 3: By compositionality level
    ax = axes[2]
    colors = {1: '#e74c3c', 2: '#e67e22', 3: '#f1c40f', 4: '#2ecc71', 5: '#27ae60'}
    for level_str, data in results["probe3_by_compositionality"].items():
        level = int(level_str)
        if data["layer"]:
            ax.plot(data["layer"], data["accuracy"],
                   marker='o', color=colors.get(level, 'gray'),
                   label=f'Comp={level}', linewidth=1.5, markersize=4)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe 3: Compound vs. Control by Compositionality")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_layerwise_probing.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp3_layerwise_probing.png'}")


# ── Experiment 4: Attention Pattern Analysis ─────────────────────────────

def experiment4_attention_patterns(model):
    """
    Analyze how attention at the word2 position relates to word1.
    If compound concepts are assembled via attention, the word2 position
    should attend heavily to the word1 position specifically in compound contexts.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Attention Pattern Analysis")
    print("="*70)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []

    for compound, word1, word2, comp_rating, control in tqdm(COMPOUNDS, desc="Attention analysis"):
        w1_toks = get_token_ids(model, word1)
        w2_toks = get_token_ids(model, word2)
        if len(w1_toks) != 1 or len(w2_toks) != 1:
            continue

        compound_attn_to_w1 = np.zeros((n_layers, n_heads))
        control_attn_to_w1_pos = np.zeros((n_layers, n_heads))
        n_compound = 0
        n_control = 0

        # Compound contexts
        for template in TEMPLATES:
            sentence = template.format(compound=compound)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            token_strs = [model.to_string([t]) for t in tokens[0]]
            w2_str = f" {word2}"
            w2_pos = None
            w1_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w2_str and pos > 0 and token_strs[pos-1].strip() == word1:
                    w2_pos = pos
                    w1_pos = pos - 1
                    break
            if w2_pos is None:
                continue

            for layer in range(n_layers):
                attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, w2_pos, w1_pos]
                compound_attn_to_w1[layer] += attn.cpu().numpy()
            n_compound += 1

        # Control contexts
        for template in TEMPLATES:
            sentence = template.format(compound=control)
            tokens = model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            token_strs = [model.to_string([t]) for t in tokens[0]]
            w2_str = f" {word2}"
            w2_pos = None
            for pos, ts in enumerate(token_strs):
                if ts == w2_str:
                    w2_pos = pos
                    break
            if w2_pos is None or w2_pos < 1:
                continue

            ctrl_w1_pos = w2_pos - 1  # word just before word2 in control

            for layer in range(n_layers):
                attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, w2_pos, ctrl_w1_pos]
                control_attn_to_w1_pos[layer] += attn.cpu().numpy()
            n_control += 1

        if n_compound > 0:
            compound_attn_to_w1 /= n_compound
        if n_control > 0:
            control_attn_to_w1_pos /= n_control

        # Compute per-layer average attention difference
        attn_diff = compound_attn_to_w1 - control_attn_to_w1_pos

        results.append({
            "compound": compound,
            "compositionality": comp_rating,
            "compound_attn_to_word1": compound_attn_to_w1.tolist(),
            "control_attn_to_prev": control_attn_to_w1_pos.tolist(),
            "attn_diff_mean_per_layer": attn_diff.mean(axis=1).tolist(),
            "n_compound": n_compound,
            "n_control": n_control,
        })

        mean_diff = attn_diff.mean(axis=1)
        max_layer = np.argmax(np.abs(mean_diff))
        print(f"  {compound:20s} | max attn diff at layer {max_layer}: {mean_diff[max_layer]:.4f}")

    with open(RESULTS_DIR / "exp4_attention.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_experiment4(results):
    """Visualize attention pattern analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Attention to word1 across layers (compound vs control)
    ax = axes[0]
    all_compound_attn = []
    all_control_attn = []
    for r in results:
        compound_mean = np.array(r["compound_attn_to_word1"]).mean(axis=1)
        control_mean = np.array(r["control_attn_to_prev"]).mean(axis=1)
        all_compound_attn.append(compound_mean)
        all_control_attn.append(control_mean)

    all_compound_attn = np.array(all_compound_attn)
    all_control_attn = np.array(all_control_attn)

    layers = range(all_compound_attn.shape[1])
    ax.plot(layers, all_compound_attn.mean(axis=0), 'b-o', label='Compound (w2→w1)', linewidth=2)
    ax.fill_between(layers,
                    all_compound_attn.mean(axis=0) - all_compound_attn.std(axis=0),
                    all_compound_attn.mean(axis=0) + all_compound_attn.std(axis=0),
                    alpha=0.2, color='blue')
    ax.plot(layers, all_control_attn.mean(axis=0), 'r-s', label='Control (w2→prev)', linewidth=2)
    ax.fill_between(layers,
                    all_control_attn.mean(axis=0) - all_control_attn.std(axis=0),
                    all_control_attn.mean(axis=0) + all_control_attn.std(axis=0),
                    alpha=0.2, color='red')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Attention from Word2 to Word1/Previous Word")
    ax.legend()

    # Plot 2: Attention difference by compositionality
    ax = axes[1]
    for r in results:
        diff = r["attn_diff_mean_per_layer"]
        color = plt.cm.RdYlGn(r["compositionality"] / 5.0)
        ax.plot(range(len(diff)), diff, alpha=0.5, color=color, linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Difference (Compound - Control)")
    ax.set_title("Differential Attention to Preceding Word")
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(1, 5))
    plt.colorbar(sm, ax=ax, label="Compositionality")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp4_attention_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp4_attention_patterns.png'}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("WHERE IS 'WASHING MACHINE' STORED IN LLMs?")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Number of compounds: {len(COMPOUNDS)}")
    print(f"Number of templates: {len(TEMPLATES)}")

    # Log environment
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TransformerLens: {getattr(tl, '__version__', 'unknown')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Save config
    config = {
        "seed": SEED,
        "model": "gpt2",
        "n_compounds": len(COMPOUNDS),
        "n_templates": len(TEMPLATES),
        "device": DEVICE,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load model
    model = load_model("gpt2")

    # Run experiments
    print("\n\n" + "="*70)
    print("RUNNING ALL EXPERIMENTS")
    print("="*70)

    # Experiment 1: Next-token prediction
    exp1_results = experiment1_next_token_prediction(model)
    plot_experiment1(exp1_results)

    # Experiment 2: Residual stream directions
    exp2_results = experiment2_residual_directions(model)
    plot_experiment2(exp2_results)

    # Experiment 3: Layer-wise probing
    exp3_results = experiment3_layerwise_probing(model)
    plot_experiment3(exp3_results)

    # Experiment 4: Attention patterns
    exp4_results = experiment4_attention_patterns(model)
    plot_experiment4(exp4_results)

    print("\n\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")

    return exp1_results, exp2_results, exp3_results, exp4_results


if __name__ == "__main__":
    main()
