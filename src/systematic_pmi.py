"""
Systematic PMI-Based Compound Knowledge Testing in GPT-2
Roadmap Item 1: Test whether models know the 2nd token in high-PMI 2-token phrases.

This script:
1. Extracts high-PMI bigrams from NLTK corpora
2. Filters to those that tokenize as exactly 2 GPT-2 tokens
3. Measures P(token2 | context + token1) vs P(token2 | context + control)
4. Analyzes the relationship between PMI and prediction accuracy

Author: Doris Wang
"""

import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# 1. EXTRACT HIGH-PMI BIGRAMS FROM NLTK CORPORA

def download_nltk_data():
    """Download required NLTK corpora."""
    import nltk
    for corpus in ["brown", "reuters", "gutenberg"]:
        nltk.download(corpus, quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def extract_bigrams_from_nltk(min_freq=20, top_n=500):
    """
    Extract bigrams from NLTK corpora and compute PMI scores.
    
    PMI(x,y) = log2( P(x,y) / (P(x) * P(y)) )
    
    High PMI means the two words co-occur much more than chance predicts.
    """
    import nltk
    from nltk.corpus import brown, reuters, gutenberg

    print("Extracting bigrams from NLTK corpora...")
    
    # Collect all words from multiple corpora for robustness
    all_words = []
    for corpus in [brown, reuters, gutenberg]:
        try:
            words = [w.lower() for w in corpus.words() if w.isalpha()]
            all_words.extend(words)
        except Exception as e:
            print(f"  Skipping corpus: {e}")

    total_words = len(all_words)
    print(f"  Total words collected: {total_words:,}")

    # Count unigrams and bigrams
    unigram_counts = Counter(all_words)
    bigram_list = list(zip(all_words[:-1], all_words[1:]))
    bigram_counts = Counter(bigram_list)

    # Compute PMI for each bigram
    pmi_scores = []
    for (w1, w2), count in bigram_counts.items():
        if count < min_freq:
            continue
        # Both words must appear frequently enough
        if unigram_counts[w1] < min_freq or unigram_counts[w2] < min_freq:
            continue
        # Skip very short words (likely noise)
        if len(w1) < 3 or len(w2) < 3:
            continue

        p_xy = count / len(bigram_list)
        p_x = unigram_counts[w1] / total_words
        p_y = unigram_counts[w2] / total_words

        pmi = math.log2(p_xy / (p_x * p_y))

        pmi_scores.append({
            "word1": w1,
            "word2": w2,
            "bigram": f"{w1} {w2}",
            "pmi": pmi,
            "freq": count,
            "freq_w1": unigram_counts[w1],
            "freq_w2": unigram_counts[w2],
        })

    # Sort by PMI descending
    pmi_scores.sort(key=lambda x: x["pmi"], reverse=True)
    print(f"  Bigrams with freq >= {min_freq}: {len(pmi_scores)}")

    return pmi_scores[:top_n]


# 2. FILTER TO GPT-2 COMPATIBLE BIGRAMS

def filter_for_gpt2(bigrams, tokenizer):
    """
    Keep only bigrams where both words tokenize to exactly 1 GPT-2 token.
    This is essential for clean analysis of next-token prediction.
    """
    valid = []
    for entry in bigrams:
        w1, w2 = entry["word1"], entry["word2"]
        # GPT-2 tokenizer adds a space prefix, so we test with leading space
        tok1 = tokenizer.encode(f" {w1}", add_special_tokens=False)
        tok2 = tokenizer.encode(f" {w2}", add_special_tokens=False)

        if len(tok1) == 1 and len(tok2) == 1:
            entry["token_id_w1"] = tok1[0]
            entry["token_id_w2"] = tok2[0]
            valid.append(entry)

    print(f"  Bigrams with single-token w1 and w2: {len(valid)}")
    return valid


# 3. MEASURE NEXT-TOKEN PREDICTION

TEMPLATES = [
    "The {word} was",
    "She bought a {word} for",
    "I saw the {word} in the",
    "There is a {word} near the",
    "He fixed the {word} with",
    "A new {word} arrived",
    "The old {word} needed",
    "We need a {word} to",
]

# Control words: common adjectives that should not strongly predict the target
CONTROL_WORDS = ["red", "big", "new", "old", "small", "good", "great", "real"]


def get_next_token_prob(model, tokenizer, context_text, target_token_id, device):
    """
    Given a context string, return the probability assigned to a specific
    next token, its rank among all vocabulary tokens, and the top-5 predictions.
    """
    input_ids = tokenizer.encode(context_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        # logits at the last position
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    target_prob = probs[target_token_id].item()

    # Rank (0-indexed, so rank 1 = index 0 after sorting)
    sorted_indices = torch.argsort(probs, descending=True)
    rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item() + 1

    # Top 5 predictions
    top5_ids = sorted_indices[:5].tolist()
    top5_tokens = [tokenizer.decode(t).strip() for t in top5_ids]
    top5_probs = [probs[t].item() for t in top5_ids]

    return target_prob, rank, list(zip(top5_tokens, top5_probs))


def find_valid_control(word2, tokenizer, device, model):
    """
    Find a control word that tokenizes to a single token for fair comparison.
    """
    for ctrl in CONTROL_WORDS:
        tok = tokenizer.encode(f" {ctrl}", add_special_tokens=False)
        if len(tok) == 1:
            return ctrl
    return "red"  # fallback


def run_prediction_experiment(model, tokenizer, bigrams, device, max_bigrams=200):
    """
    For each bigram (w1, w2):
    - Measure P(w2 | template + w1) across all templates
    - Measure P(w2 | template + control) across all templates
    - Compute boost ratio = mean P(w2|w1) / mean P(w2|control)
    """
    results = []
    test_bigrams = bigrams[:max_bigrams]

    print(f"\nRunning prediction experiment on {len(test_bigrams)} bigrams...")

    for entry in tqdm(test_bigrams, desc="Testing bigrams"):
        w1 = entry["word1"]
        w2 = entry["word2"]
        target_id = entry["token_id_w2"]
        ctrl = find_valid_control(w2, tokenizer, device, model)

        compound_probs = []
        control_probs = []
        compound_ranks = []
        control_ranks = []
        top5_example = None

        for template in TEMPLATES:
            # Compound context: template with w1
            ctx_compound = template.format(word=w1)
            p_compound, rank_compound, top5 = get_next_token_prob(
                model, tokenizer, ctx_compound, target_id, device
            )
            compound_probs.append(p_compound)
            compound_ranks.append(rank_compound)

            if top5_example is None:
                top5_example = top5

            # Control context: template with control word
            ctx_control = template.format(word=ctrl)
            p_control, rank_control, _ = get_next_token_prob(
                model, tokenizer, ctx_control, target_id, device
            )
            control_probs.append(p_control)
            control_ranks.append(rank_control)

        mean_p_compound = np.mean(compound_probs)
        mean_p_control = np.mean(control_probs)

        # Avoid division by zero
        boost = mean_p_compound / max(mean_p_control, 1e-10)

        result = {
            "word1": w1,
            "word2": w2,
            "bigram": entry["bigram"],
            "pmi": entry["pmi"],
            "freq": entry["freq"],
            "mean_prob_compound": float(mean_p_compound),
            "mean_prob_control": float(mean_p_control),
            "boost_ratio": float(boost),
            "mean_rank_compound": float(np.mean(compound_ranks)),
            "mean_rank_control": float(np.mean(control_ranks)),
            "best_rank_compound": int(min(compound_ranks)),
            "top5_after_w1": top5_example,
        }
        results.append(result)

    return results


# 4. ANALYSIS AND VISUALIZATION

def analyze_results(results, output_dir):
    """Generate statistical analysis and plots."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    # Basic statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total bigrams tested: {len(df)}")
    print(f"Median boost ratio: {df['boost_ratio'].median():.1f}x")
    print(f"Mean boost ratio: {df['boost_ratio'].mean():.1f}x")
    print(f"Bigrams where w2 is rank 1 after w1: "
          f"{(df['best_rank_compound'] == 1).sum()} / {len(df)}")
    print(f"Bigrams with boost > 1: {(df['boost_ratio'] > 1).sum()} / {len(df)}")
    print(f"Bigrams with boost > 10: {(df['boost_ratio'] > 10).sum()} / {len(df)}")

    # PMI vs Boost correlation
    # Only include bigrams with positive boost for log-scale analysis
    df_positive = df[df["boost_ratio"] > 0].copy()
    df_positive["log_boost"] = np.log10(df_positive["boost_ratio"])

    spearman_r, spearman_p = stats.spearmanr(df_positive["pmi"], df_positive["log_boost"])
    pearson_r, pearson_p = stats.pearsonr(df_positive["pmi"], df_positive["log_boost"])

    print(f"\nPMI vs log(Boost) correlation:")
    print(f"  Spearman r = {spearman_r:.3f}, p = {spearman_p:.2e}")
    print(f"  Pearson  r = {pearson_r:.3f}, p = {pearson_p:.2e}")

    # Wilcoxon test: compound vs control
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
        df["mean_prob_compound"], df["mean_prob_control"],
        alternative="greater"
    )
    print(f"\nWilcoxon signed-rank test (compound > control):")
    print(f"  W = {wilcoxon_stat:.0f}, p = {wilcoxon_p:.2e}")

    # Top 20 strongest and weakest
    print("\n--- Top 20 Highest Boost ---")
    top20 = df.nlargest(20, "boost_ratio")
    for _, row in top20.iterrows():
        print(f"  {row['bigram']:25s}  PMI={row['pmi']:6.2f}  "
              f"P(w2|w1)={row['mean_prob_compound']:.4f}  "
              f"Boost={row['boost_ratio']:8.1f}x  Rank={row['best_rank_compound']}")

    print("\n--- 10 Lowest Boost (potential failures) ---")
    bot10 = df.nsmallest(10, "boost_ratio")
    for _, row in bot10.iterrows():
        print(f"  {row['bigram']:25s}  PMI={row['pmi']:6.2f}  "
              f"P(w2|w1)={row['mean_prob_compound']:.4f}  "
              f"Boost={row['boost_ratio']:8.1f}x  Rank={row['best_rank_compound']}")

    # PLOTS

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Systematic PMI-Based Compound Knowledge Testing in GPT-2",
                 fontsize=14, fontweight="bold")

    # Plot 1: PMI vs Log Boost Ratio
    ax = axes[0, 0]
    scatter = ax.scatter(df_positive["pmi"], df_positive["log_boost"],
                         alpha=0.5, s=20, c=df_positive["pmi"], cmap="viridis")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Boost = 1x")
    z = np.polyfit(df_positive["pmi"], df_positive["log_boost"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_positive["pmi"].min(), df_positive["pmi"].max(), 100)
    ax.plot(x_line, p(x_line), "r-", alpha=0.7,
            label=f"Spearman r={spearman_r:.3f}")
    ax.set_xlabel("PMI Score")
    ax.set_ylabel("log₁₀(Boost Ratio)")
    ax.set_title("PMI vs Prediction Boost")
    ax.legend(fontsize=9)

    # Plot 2: Distribution of boost ratios
    ax = axes[0, 1]
    log_boosts = df_positive["log_boost"]
    ax.hist(log_boosts, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(x=np.median(log_boosts), color="red", linestyle="--",
               label=f"Median = {10**np.median(log_boosts):.1f}x")
    ax.set_xlabel("log₁₀(Boost Ratio)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Boost Ratios")
    ax.legend(fontsize=9)

    # Plot 3: Best rank distribution
    ax = axes[1, 0]
    rank_bins = [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 50000]
    rank_labels = ["1", "2-4", "5-9", "10-49", "50-99",
                   "100-499", "500-999", "1K-5K", "5K+"]
    rank_data = pd.cut(df["best_rank_compound"], bins=rank_bins + [999999],
                       labels=rank_labels + ["50K+"], right=False)
    rank_counts = rank_data.value_counts().reindex(rank_labels + ["50K+"], fill_value=0)
    bars = ax.bar(range(len(rank_counts)), rank_counts.values,
                  color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xticks(range(len(rank_counts)))
    ax.set_xticklabels(rank_counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Best Rank of w2 after w1")
    ax.set_ylabel("Count")
    ax.set_title("Where Does w2 Rank After Seeing w1?")
    # Highlight rank 1
    if rank_counts.iloc[0] > 0:
        bars[0].set_color("darkgreen")

    # Plot 4: PMI vs P(w2|w1) directly
    ax = axes[1, 1]
    ax.scatter(df["pmi"], df["mean_prob_compound"], alpha=0.5, s=20, color="steelblue")
    ax.set_xlabel("PMI Score")
    ax.set_ylabel("P(w2 | context + w1)")
    ax.set_title("PMI vs Direct Prediction Probability")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "systematic_pmi_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {fig_path}")

    # ---- Save detailed results ----
    stats_dict = {
        "total_bigrams": len(df),
        "median_boost": float(df["boost_ratio"].median()),
        "mean_boost": float(df["boost_ratio"].mean()),
        "rank1_count": int((df["best_rank_compound"] == 1).sum()),
        "boost_gt_1": int((df["boost_ratio"] > 1).sum()),
        "boost_gt_10": int((df["boost_ratio"] > 10).sum()),
        "boost_gt_100": int((df["boost_ratio"] > 100).sum()),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "wilcoxon_stat": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
    }

    results_path = os.path.join(output_dir, "systematic_pmi_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "statistics": stats_dict,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Save CSV for easy viewing
    csv_path = os.path.join(output_dir, "systematic_pmi_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    return stats_dict



# 5. MAIN

def main():
    print("=" * 60)
    print("Systematic PMI-Based Compound Knowledge Testing in GPT-2")
    print("Roadmap Item 1: Do models know the 2nd token in high-PMI phrases?")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    # Step 1: Extract high-PMI bigrams
    download_nltk_data()
    bigrams = extract_bigrams_from_nltk(min_freq=20, top_n=500)

    # Step 2: Filter for GPT-2 compatibility
    bigrams = filter_for_gpt2(bigrams, tokenizer)

    # Step 3: Run experiment
    results = run_prediction_experiment(
        model, tokenizer, bigrams, device, max_bigrams=200
    )

    # Step 4: Analyze and visualize
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results"
    )
    stats_dict = analyze_results(results, output_dir)

    print("\n" + "=" * 60)
    print("DONE. Key finding:")
    print(f"  Out of {stats_dict['total_bigrams']} high-PMI bigrams,")
    print(f"  {stats_dict['rank1_count']} had w2 as the #1 prediction after w1")
    print(f"  {stats_dict['boost_gt_10']} had boost ratio > 10x")
    print(f"  PMI-Boost correlation: Spearman r={stats_dict['spearman_r']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
