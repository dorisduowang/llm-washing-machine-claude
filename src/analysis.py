"""
Statistical analysis and additional visualizations for the compound concept experiments.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def analyze_experiment1():
    """Deep analysis of next-token prediction results."""
    print("="*70)
    print("EXPERIMENT 1: Statistical Analysis")
    print("="*70)

    with open(RESULTS_DIR / "exp1_next_token.json") as f:
        results = json.load(f)

    compounds = [r["compound"] for r in results]
    comp_ratings = np.array([r["compositionality"] for r in results])
    p_compound = np.array([r["p_word2_compound_mean"] for r in results])
    p_control = np.array([r["p_word2_control_mean"] for r in results])
    boost_ratios = np.array([r["boost_ratio"] for r in results])
    ranks_compound = np.array([r["rank_compound_mean"] for r in results])
    ranks_control = np.array([r["rank_control_mean"] for r in results])

    print(f"\n{'Compound':<22} {'P(w2|w1)':<12} {'P(w2|ctrl)':<12} {'Boost':<10} {'Rank(cmp)':<12} {'Rank(ctrl)':<12}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["boost_ratio"], reverse=True):
        print(f"{r['compound']:<22} {r['p_word2_compound_mean']:<12.4f} "
              f"{r['p_word2_control_mean']:<12.4f} {r['boost_ratio']:<10.1f} "
              f"{r['rank_compound_mean']:<12.0f} {r['rank_control_mean']:<12.0f}")

    # Statistical tests
    print("\n--- Statistical Tests ---")

    # Paired Wilcoxon: compound probs vs control probs
    stat, p_val = stats.wilcoxon(p_compound, p_control, alternative='greater')
    print(f"Wilcoxon (P(w2|w1) > P(w2|ctrl)): W={stat:.1f}, p={p_val:.2e}")

    # Spearman: compositionality vs boost ratio
    r_boost, p_boost = stats.spearmanr(comp_ratings, boost_ratios)
    print(f"Spearman (compositionality vs boost): r={r_boost:.3f}, p={p_boost:.3f}")

    # Spearman: compositionality vs rank
    r_rank, p_rank = stats.spearmanr(comp_ratings, ranks_compound)
    print(f"Spearman (compositionality vs rank): r={r_rank:.3f}, p={p_rank:.3f}")

    # Log boost for better visualization
    log_boost = np.log10(boost_ratios + 1)
    r_log, p_log = stats.spearmanr(comp_ratings, log_boost)
    print(f"Spearman (compositionality vs log10(boost)): r={r_log:.3f}, p={p_log:.3f}")

    # Group by compositionality level
    print("\n--- By Compositionality Level ---")
    for level in sorted(set(comp_ratings)):
        mask = comp_ratings == level
        n = mask.sum()
        mean_boost = boost_ratios[mask].mean()
        mean_p = p_compound[mask].mean()
        mean_rank = ranks_compound[mask].mean()
        print(f"  Level {level} (n={n}): mean boost={mean_boost:.1f}x, "
              f"mean P(w2|w1)={mean_p:.4f}, mean rank={mean_rank:.0f}")

    # Effect size: Cohen's d for compound vs control probabilities
    diff = p_compound - p_control
    cohens_d = diff.mean() / diff.std()
    print(f"\nCohen's d (compound vs control probs): d={cohens_d:.2f}")

    # Bootstrap CI for mean boost ratio
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(boost_ratios), len(boost_ratios), replace=True)
        boot_means.append(np.median(boost_ratios[idx]))
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    print(f"Median boost ratio: {np.median(boost_ratios):.1f}x (95% CI: [{ci_lower:.1f}, {ci_upper:.1f}])")

    # Identify compounds where word2 is in top-1 after word1
    top1 = [r for r in results if r["rank_compound_mean"] <= 1.5]
    print(f"\nCompounds where word2 is rank 1 after word1: {len(top1)}/{len(results)}")
    for r in top1:
        print(f"  {r['compound']}: P={r['p_word2_compound_mean']:.4f}")

    # Create enhanced figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: probability comparison
    ax = axes[0, 0]
    x = np.arange(len(compounds))
    sorted_idx = np.argsort(p_compound)[::-1]
    bar_width = 0.35
    bars1 = ax.bar(x - bar_width/2, [p_compound[i] for i in sorted_idx],
                   bar_width, label='P(w2 | word1)', color='steelblue')
    bars2 = ax.bar(x + bar_width/2, [p_control[i] for i in sorted_idx],
                   bar_width, label='P(w2 | control)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([compounds[i] for i in sorted_idx], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("P(word2)")
    ax.set_title("Next-Token Probability: Compound Word1 vs. Control Word")
    ax.legend()

    # Top-right: rank comparison
    ax = axes[0, 1]
    sorted_idx_rank = np.argsort(ranks_compound)
    ax.barh(x, [ranks_compound[i] for i in sorted_idx_rank],
            bar_width*2, label='Rank after word1', color='steelblue')
    ax.set_yticks(x)
    ax.set_yticklabels([compounds[i] for i in sorted_idx_rank], fontsize=7)
    ax.set_xlabel("Rank of word2")
    ax.set_title("Rank of Word2 After Seeing Word1")
    ax.set_xscale('log')
    ax.invert_yaxis()

    # Bottom-left: log boost ratio by compositionality
    ax = axes[1, 0]
    colors_map = {1: '#e74c3c', 2: '#e67e22', 3: '#f1c40f', 4: '#2ecc71', 5: '#27ae60'}
    for i, r in enumerate(results):
        ax.scatter(r['compositionality'], np.log10(r['boost_ratio'] + 1),
                  c=colors_map.get(r['compositionality'], 'gray'), s=100, edgecolors='k', zorder=5)
        ax.annotate(r['compound'], (r['compositionality'], np.log10(r['boost_ratio'] + 1)),
                   fontsize=6, ha='left', va='bottom', rotation=10)
    ax.set_xlabel("Compositionality Rating (1=idiomatic, 5=compositional)")
    ax.set_ylabel("log10(Boost Ratio + 1)")
    ax.set_title(f"Compositionality vs. Prediction Boost\n(Spearman r={r_log:.3f}, p={p_log:.3f})")

    # Bottom-right: Top-5 predictions for key compounds
    ax = axes[1, 1]
    key_compounds = ["washing machine", "hot dog", "coffee table", "guinea pig", "shooting star", "mountain cabin"]
    y_pos = 0
    for r in results:
        if r["compound"] in key_compounds:
            top5_str = ", ".join([f"{t[0]}({t[1]:.3f})" for t in r["top5_after_word1"][:5]])
            ax.text(0.02, y_pos, f"{r['compound']}:", fontsize=9, fontweight='bold',
                   transform=ax.transAxes, va='top')
            ax.text(0.02, y_pos - 0.04, f"  Top-5 after '{r['word1']}': {top5_str}",
                   fontsize=7, transform=ax.transAxes, va='top')
            y_pos -= 0.12
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Top-5 Predictions After Word1 (Selected Compounds)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_detailed_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'exp1_detailed_analysis.png'}")

    return {
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p_val),
        "spearman_boost_r": float(r_boost),
        "spearman_boost_p": float(p_boost),
        "spearman_logboost_r": float(r_log),
        "spearman_logboost_p": float(p_log),
        "cohens_d": float(cohens_d),
        "median_boost": float(np.median(boost_ratios)),
        "median_boost_ci": [float(ci_lower), float(ci_upper)],
        "n_top1": len(top1),
    }


def analyze_experiment2():
    """Deep analysis of residual stream direction results."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Statistical Analysis")
    print("="*70)

    with open(RESULTS_DIR / "exp2_residual_directions.json") as f:
        results = json.load(f)

    # Extract final layer metrics
    final_r2 = []
    final_sim_w1 = []
    final_sim_w2 = []
    final_residual = []
    comps = []
    names = []

    for r in results:
        last = r["layer_metrics"][-1]
        if last is None:
            continue
        final_r2.append(last["reconstruction_r2"])
        final_sim_w1.append(last["sim_compound_word1"])
        final_sim_w2.append(last["sim_compound_word2"])
        final_residual.append(last["residual_norm_ratio"])
        comps.append(r["compositionality"])
        names.append(r["compound"])

    final_r2 = np.array(final_r2)
    final_sim_w1 = np.array(final_sim_w1)
    final_sim_w2 = np.array(final_sim_w2)
    final_residual = np.array(final_residual)
    comps = np.array(comps)

    print(f"\n{'Compound':<22} {'R²':<8} {'cos(C,w1)':<12} {'cos(C,w2)':<12} {'Residual':<10}")
    print("-" * 64)
    for i, name in enumerate(names):
        print(f"{name:<22} {final_r2[i]:<8.3f} {final_sim_w1[i]:<12.3f} "
              f"{final_sim_w2[i]:<12.3f} {final_residual[i]:<10.3f}")

    print(f"\n--- Summary ---")
    print(f"Mean R²: {final_r2.mean():.3f} ± {final_r2.std():.3f}")
    print(f"Mean cos(C,w1): {final_sim_w1.mean():.3f} ± {final_sim_w1.std():.3f}")
    print(f"Mean cos(C,w2): {final_sim_w2.mean():.3f} ± {final_sim_w2.std():.3f}")
    print(f"Mean residual ratio: {final_residual.mean():.3f} ± {final_residual.std():.3f}")

    # Is R² significantly > 0? (one-sample t-test)
    t_stat, p_val = stats.ttest_1samp(final_r2, 0)
    print(f"\nt-test (R² > 0): t={t_stat:.2f}, p={p_val:.2e}")

    # Spearman: compositionality vs R²
    r_r2, p_r2 = stats.spearmanr(comps, final_r2)
    print(f"Spearman (compositionality vs R²): r={r_r2:.3f}, p={p_r2:.3f}")

    # Spearman: compositionality vs residual
    r_res, p_res = stats.spearmanr(comps, final_residual)
    print(f"Spearman (compositionality vs residual): r={r_res:.3f}, p={p_res:.3f}")

    # Layer-wise R² evolution
    print("\n--- Layer-wise R² Evolution ---")
    n_layers = 12
    mean_r2_by_layer = np.zeros(n_layers)
    std_r2_by_layer = np.zeros(n_layers)
    for layer in range(n_layers):
        r2_vals = []
        for r in results:
            m = r["layer_metrics"][layer]
            if m is not None:
                r2_vals.append(m["reconstruction_r2"])
        mean_r2_by_layer[layer] = np.mean(r2_vals)
        std_r2_by_layer[layer] = np.std(r2_vals)
        print(f"  Layer {layer:2d}: mean R² = {mean_r2_by_layer[layer]:.3f} ± {std_r2_by_layer[layer]:.3f}")

    # Create enhanced figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: R² evolution by layer with mean line
    ax = axes[0, 0]
    ax.fill_between(range(n_layers),
                    mean_r2_by_layer - std_r2_by_layer,
                    mean_r2_by_layer + std_r2_by_layer,
                    alpha=0.3, color='steelblue')
    ax.plot(range(n_layers), mean_r2_by_layer, 'o-', color='steelblue', linewidth=2,
            label=f'Mean R² (n={len(results)})')
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (compound ≈ α·word1 + β·word2)")
    ax.set_title("How Well Can Constituents Reconstruct Compound Direction?")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Top-right: Cosine similarity heatmap (compound vs w1, w2)
    ax = axes[0, 1]
    sim_data = np.zeros((len(names), 3))
    for i, name in enumerate(names):
        sim_data[i, 0] = final_sim_w1[i]
        sim_data[i, 1] = final_sim_w2[i]
        sim_data[i, 2] = final_r2[i]
    im = ax.imshow(sim_data, aspect='auto', cmap='YlOrRd', vmin=0.8, vmax=1.0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['cos(C, w1)', 'cos(C, w2)', 'R²'])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Similarity Metrics (Final Layer)")
    plt.colorbar(im, ax=ax)

    # Bottom-left: Residual vs compositionality
    ax = axes[1, 0]
    colors_map = {1: '#e74c3c', 2: '#e67e22', 3: '#f1c40f', 4: '#2ecc71', 5: '#27ae60'}
    for i, name in enumerate(names):
        ax.scatter(comps[i], final_residual[i],
                  c=colors_map.get(comps[i], 'gray'), s=100, edgecolors='k', zorder=5)
        ax.annotate(name, (comps[i], final_residual[i]),
                   fontsize=6, ha='left', va='bottom', rotation=10)
    ax.set_xlabel("Compositionality Rating")
    ax.set_ylabel("Residual Norm Ratio (Unique Component)")
    ax.set_title(f"Compositionality vs. Unique Compound Component\n(Spearman r={r_res:.3f}, p={p_res:.3f})")

    # Bottom-right: Layer-wise residual evolution for selected compounds
    ax = axes[1, 1]
    selected = ["washing machine", "hot dog", "coffee table", "mountain cabin", "shooting star"]
    for r in results:
        if r["compound"] in selected:
            layers = []
            residuals = []
            for m in r["layer_metrics"]:
                if m is not None:
                    layers.append(m["layer"])
                    residuals.append(m["residual_norm_ratio"])
            ax.plot(layers, residuals, 'o-', label=r["compound"], linewidth=1.5, markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Residual Norm Ratio")
    ax.set_title("Unique Component Evolution (Selected Compounds)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_detailed_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'exp2_detailed_analysis.png'}")

    return {
        "mean_r2": float(final_r2.mean()),
        "std_r2": float(final_r2.std()),
        "mean_residual": float(final_residual.mean()),
        "ttest_r2_stat": float(t_stat),
        "ttest_r2_p": float(p_val),
        "spearman_comp_r2_r": float(r_r2),
        "spearman_comp_r2_p": float(p_r2),
        "spearman_comp_residual_r": float(r_res),
        "spearman_comp_residual_p": float(p_res),
    }


def analyze_experiment3():
    """Deep analysis of probing results."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Statistical Analysis")
    print("="*70)

    with open(RESULTS_DIR / "exp3_probing.json") as f:
        results = json.load(f)

    p1 = results["probe1_word1_identity"]
    p2 = results["probe2_compound_vs_control"]

    print("\n--- Probe 1: Word1 Identity (Token Erasure) ---")
    print(f"Accuracy across all layers: {np.mean(p1['accuracy']):.3f}")
    print(f"  (All layers show perfect accuracy => NO token erasure in GPT-2)")
    print(f"Number of classes: {results['n_word1_classes']}")
    chance = 1.0 / results['n_word1_classes']
    print(f"Chance level: {chance:.3f}")

    print("\n--- Probe 2: Compound vs. Control ---")
    max_layer = np.argmax(p2["accuracy"])
    max_acc = p2["accuracy"][max_layer]
    print(f"Peak accuracy: {max_acc:.3f} at layer {max_layer}")
    print(f"Layer 0 accuracy: {p2['accuracy'][0]:.3f}")
    print(f"Final layer accuracy: {p2['accuracy'][-1]:.3f}")

    # Test if peak accuracy is significantly above chance (0.5)
    # Since we have CV scores, we can use a one-sample t-test on the mean
    # However, we only have the mean and std across folds
    # We'll compute a z-test instead
    n_samples = results["n_compound_samples"] + results["n_control_samples"]
    se = p2["accuracy_std"][max_layer] / np.sqrt(5)  # 5-fold CV
    z = (max_acc - 0.5) / se if se > 0 else float('inf')
    p_val = 1 - stats.norm.cdf(z)
    print(f"z-test (peak accuracy > 0.5): z={z:.2f}, p={p_val:.2e}")

    print("\n--- Probe 3: By Compositionality Level ---")
    for level, data in results["probe3_by_compositionality"].items():
        if data["accuracy"]:
            mean_acc = np.mean(data["accuracy"])
            print(f"  Compositionality {level}: mean accuracy = {mean_acc:.3f}")

    return {
        "probe1_mean_accuracy": float(np.mean(p1["accuracy"])),
        "probe2_peak_accuracy": float(max_acc),
        "probe2_peak_layer": int(max_layer),
        "probe2_z": float(z),
        "probe2_p": float(p_val),
    }


def analyze_experiment4():
    """Deep analysis of attention pattern results."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Statistical Analysis")
    print("="*70)

    with open(RESULTS_DIR / "exp4_attention.json") as f:
        results = json.load(f)

    print(f"\n{'Compound':<22} {'Max Attn Diff':<14} {'Layer':<8} {'Comp Rating':<12}")
    print("-" * 56)
    for r in results:
        diffs = r["attn_diff_mean_per_layer"]
        max_idx = np.argmax(np.abs(diffs))
        print(f"{r['compound']:<22} {diffs[max_idx]:<14.4f} {max_idx:<8} {r['compositionality']:<12}")

    # Aggregate: mean attention difference across all compounds
    all_diffs = np.array([r["attn_diff_mean_per_layer"] for r in results])
    mean_diffs = all_diffs.mean(axis=0)
    print(f"\nMean attention difference by layer:")
    for l, d in enumerate(mean_diffs):
        sig = "*" if abs(d) > 0.02 else ""
        print(f"  Layer {l:2d}: {d:+.4f} {sig}")

    # Test: does compound context produce more attention to word1 than control?
    # Use all layers, paired t-test across compounds
    attn_compound_all = np.array([np.array(r["compound_attn_to_word1"]).mean(axis=1) for r in results])
    attn_control_all = np.array([np.array(r["control_attn_to_prev"]).mean(axis=1) for r in results])

    # Layer-wise paired t-tests
    print("\nPaired t-tests by layer (compound vs control attention):")
    for layer in range(12):
        t_stat, p_val = stats.ttest_rel(attn_compound_all[:, layer], attn_control_all[:, layer])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  Layer {layer:2d}: t={t_stat:+.2f}, p={p_val:.4f} {sig}")

    return {"mean_attn_diff_by_layer": mean_diffs.tolist()}


def create_summary_figure():
    """Create a single summary figure combining key findings."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Load all results
    with open(RESULTS_DIR / "exp1_next_token.json") as f:
        exp1 = json.load(f)
    with open(RESULTS_DIR / "exp2_residual_directions.json") as f:
        exp2 = json.load(f)
    with open(RESULTS_DIR / "exp3_probing.json") as f:
        exp3 = json.load(f)
    with open(RESULTS_DIR / "exp4_attention.json") as f:
        exp4 = json.load(f)

    # 1. Boost ratio bar chart
    ax = axes[0, 0]
    boost_ratios = [(r["compound"], r["boost_ratio"], r["compositionality"]) for r in exp1]
    boost_ratios.sort(key=lambda x: x[1], reverse=True)
    colors_map = {1: '#e74c3c', 2: '#e67e22', 3: '#f1c40f', 4: '#2ecc71', 5: '#27ae60'}
    bars = ax.barh(range(len(boost_ratios)),
                   [np.log10(b[1] + 1) for b in boost_ratios],
                   color=[colors_map.get(b[2], 'gray') for b in boost_ratios])
    ax.set_yticks(range(len(boost_ratios)))
    ax.set_yticklabels([b[0] for b in boost_ratios], fontsize=7)
    ax.set_xlabel("log10(Boost Ratio + 1)")
    ax.set_title("(A) Word1 → Word2 Prediction Boost")
    ax.invert_yaxis()

    # 2. Layer-wise R² evolution
    ax = axes[0, 1]
    n_layers = 12
    mean_r2 = np.zeros(n_layers)
    std_r2 = np.zeros(n_layers)
    for layer in range(n_layers):
        vals = [r["layer_metrics"][layer]["reconstruction_r2"]
                for r in exp2 if r["layer_metrics"][layer] is not None]
        mean_r2[layer] = np.mean(vals)
        std_r2[layer] = np.std(vals)
    ax.fill_between(range(n_layers), mean_r2 - std_r2, mean_r2 + std_r2,
                    alpha=0.3, color='steelblue')
    ax.plot(range(n_layers), mean_r2, 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("(B) Compound ≈ α·word1 + β·word2")
    ax.set_ylim(0, 1.1)

    # 3. Compound vs control classification
    ax = axes[0, 2]
    p2 = exp3["probe2_compound_vs_control"]
    ax.errorbar(p2["layer"], p2["accuracy"], yerr=p2["accuracy_std"],
                marker='s', capsize=3, color='darkorange', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("(C) Compound vs. Control Classification")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # 4. Token erasure (word1 identity)
    ax = axes[1, 0]
    p1 = exp3["probe1_word1_identity"]
    ax.plot(p1["layer"], p1["accuracy"], 'o-', color='steelblue', linewidth=2)
    chance = 1.0 / exp3["n_word1_classes"]
    ax.axhline(y=chance, color='red', linestyle='--', alpha=0.5, label=f'Chance ({chance:.2f})')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("(D) Word1 Identity from Word2 Position\n(Token Erasure Test)")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # 5. Attention patterns
    ax = axes[1, 1]
    all_compound_attn = np.array([np.array(r["compound_attn_to_word1"]).mean(axis=1) for r in exp4])
    all_control_attn = np.array([np.array(r["control_attn_to_prev"]).mean(axis=1) for r in exp4])
    layers = range(12)
    ax.plot(layers, all_compound_attn.mean(axis=0), 'b-o', label='Compound', linewidth=2, markersize=4)
    ax.plot(layers, all_control_attn.mean(axis=0), 'r-s', label='Control', linewidth=2, markersize=4)
    ax.fill_between(layers,
                    all_compound_attn.mean(axis=0) - all_compound_attn.std(axis=0),
                    all_compound_attn.mean(axis=0) + all_compound_attn.std(axis=0),
                    alpha=0.15, color='blue')
    ax.fill_between(layers,
                    all_control_attn.mean(axis=0) - all_control_attn.std(axis=0),
                    all_control_attn.mean(axis=0) + all_control_attn.std(axis=0),
                    alpha=0.15, color='red')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("(E) Attention: Word2 → Word1/Prev")
    ax.legend()

    # 6. Compositionality vs key metrics
    ax = axes[1, 2]
    comps = [r["compositionality"] for r in exp1]
    boosts = [np.log10(r["boost_ratio"] + 1) for r in exp1]
    r_val, p_val = stats.spearmanr(comps, boosts)
    ax.scatter(comps, boosts, s=80, c='steelblue', edgecolors='k')
    for r in exp1:
        ax.annotate(r["compound"],
                   (r["compositionality"], np.log10(r["boost_ratio"] + 1)),
                   fontsize=5, ha='left', va='bottom', rotation=10)
    ax.set_xlabel("Compositionality Rating")
    ax.set_ylabel("log10(Boost Ratio + 1)")
    ax.set_title(f"(F) Compositionality vs. Boost\n(r={r_val:.3f}, p={p_val:.3f})")

    plt.suptitle("Where is 'Washing Machine' Stored in GPT-2?\nKey Findings Summary",
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_figure.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'summary_figure.png'}")


if __name__ == "__main__":
    stats1 = analyze_experiment1()
    stats2 = analyze_experiment2()
    stats3 = analyze_experiment3()
    stats4 = analyze_experiment4()
    create_summary_figure()

    # Save all stats
    all_stats = {
        "experiment1": stats1,
        "experiment2": stats2,
        "experiment3": stats3,
        "experiment4": stats4,
    }
    with open(RESULTS_DIR / "statistical_analysis.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
