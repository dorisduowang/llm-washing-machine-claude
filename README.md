# Where Is "Washing Machine" Stored in LLMs?

This project studies how a language model represents compound concepts such as "washing machine", "hot dog", "school bus", and "red herring". The core question is simple but important: does a model store a compound concept as its own dedicated internal representation, or does it build the concept from smaller pieces as text unfolds?

The answer from these experiments is not a clean yes or no. In GPT-2, compound concepts look highly component based in the residual stream, but they still produce compound specific sparse features and very strong next token behavior. In other words, the model often does not need a separate "washing machine" direction. It can represent "machine" in a modified context, activate a different sparse feature pattern, and strongly predict "machine" after "washing".

The project also extends the original question beyond a single dominant compound head. It asks whether the model gives probability mass to plausible minority continuations such as "washing powder", "coffee beans", "school board", and "dark room". It then asks whether sentence context can rescue these minority continuations when a familiar compound head would otherwise dominate.

## Key Findings and Results

The residual stream results suggest that compound concepts are usually close to their component representations. At layer 11, the mean cosine similarity between a compound representation and the head noun alone is 0.954 across the 21 compound concepts. For "washing machine", the representation of the compound remains very close to "machine" in isolation, with cosine similarity 0.951. This does not support the idea that the model stores most compound concepts as clean, isolated residual stream directions.

The SAE results tell a more detailed story. Even though the residual stream directions are close, the sparse feature sets can differ substantially. Across the same compound set, 56.3 percent of active SAE features are unique to the compound context. "Red herring" and "dark horse" show especially high unique feature fractions, which fits the intuition that idiomatic compounds need more specialized meaning than transparent compounds such as "dish washer" or "kitchen chair".

The next token experiments show that some compound concepts are stored very strongly in sequential prediction. After the prompt "The washing", GPT-2 predicts "machine" as the top next token with probability 0.471, which is a 4,221 fold increase over the baseline probability of "machine" after "The". The strongest case in the dataset is "guinea pig", where the head probability after "guinea" rises by 28,037 fold. These results suggest that much of the model's compound knowledge is expressed through transition probabilities rather than through a single dedicated representation.

The minority continuation analysis shows that compound dominance is not universal. In GPT-2 Small, the expected target head beats all minority heads in some cases, such as "washing machine", "vending machine", and "swimming pool". However, in 6 of 14 analyzable compounds, plausible minority heads beat the target compound head under minimal context. This happens for generic modifiers such as "coffee", "school", "fire", "dark", "kitchen", and "office". The same 6 of 14 pattern replicates in GPT-2 Medium. This means the model is not simply memorizing one continuation for every modifier. It distributes probability across several plausible continuations when the modifier is broad.

The contextual override experiment shows that richer sentence context can redirect the model away from a dominant continuation. In GPT-2 Small, disambiguating contexts rescue 11 of 14 tested minority continuations. The minority head probability rises by 1,360 percent on average, while the dominant target probability falls by 60 percent. In GPT-2 Medium, the same experiment rescues all 14 of 14 tested minority continuations, with minority probability rising by 1,720 percent and target probability falling by 72 percent. This is the strongest evidence that compound behavior is not only a rigid bigram reflex. The model can use sentence meaning to select a different continuation.

Taken together, the results support a layered account. The residual stream mostly keeps compound concepts near their component meanings. Sparse features reveal finer compound specific structure. Next token prediction stores strong modifier to head expectations. Sentence context can override those expectations when the prompt gives the model enough semantic evidence.

## Why This Project Matters

Language models have many more referenceable concepts than residual stream dimensions. GPT-2 Small has 768 residual stream dimensions, but natural language contains millions of possible objects, events, idioms, phrases, and composed meanings. Compound concepts are a useful test case because they sit between single words and full sentences. They are common enough to measure, structured enough to compare, and varied enough to show the difference between transparent composition and idiomatic meaning.

The results matter for mechanistic interpretability because they show why a single measurement can be misleading. Cosine similarity makes compounds look almost identical to their heads. SAE features reveal that the model still changes the internal feature pattern. Next token probabilities reveal that the model can store very strong lexical expectations without needing a separate direction for the full phrase. A good account of compound concepts needs all three views.

## Reproducing the Experiments

The project is designed so that scripts resolve paths relative to the repository root. This means the results do not depend on a specific local folder name.

```bash
uv venv
source .venv/bin/activate
uv sync
```

If we are using pip instead of uv, create a virtual environment and install the project in editable mode.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the core residual stream, SAE, and next token priming experiments with:

```bash
python src/experiments.py
```

Run the minority continuation experiments with:

```bash
python src/minority_continuation_analysis.py --model gpt2
python src/minority_continuation_analysis.py --model gpt2-medium
```

Run the contextual override experiments with:

```bash
python src/contextual_override.py --model gpt2
python src/contextual_override.py --model gpt2-medium
```

CPU execution works, but GPT-2 Medium is noticeably slower without a GPU. A CUDA GPU with at least 8 GB of VRAM is recommended for comfortable reruns. The required disk space depends on the downloaded model weights and cached SAE files. GPT-2 Small is relatively light, while GPT-2 Medium downloads roughly 1.5 GB of model weights.

## Repository Structure

```text
README.md
REPORT.md
REPRODUCIBILITY.md
planning.md
literature_review.md
resources.md
src/
  experiments.py
  minority_continuation_analysis.py
  contextual_override.py
  manual_sae.py
datasets/
  compound_concepts/
    minority_continuations.json
results/
  summary.json
  exp1_residual_stream.json
  exp2_sae_features.json
  exp3_next_token.json
  minority_continuations_gpt2.json
  minority_continuations_gpt2_medium.json
  contextual_override_gpt2.json
  contextual_override_gpt2_medium.json
  plots/
```

`REPORT.md` contains the full research report, including the experimental design, statistical results, interpretation, and next steps. `REPRODUCIBILITY.md` gives exact commands, expected summary outputs, and branch verification steps.

Downloaded paper files, local workspace files, caches, and local logs are intentionally excluded from version control. Curated literature notes remain in `literature_review.md` and `resources.md`.
