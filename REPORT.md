# Where Is "Washing Machine" Stored in LLMs?

## 1. Executive Summary

This project asks whether a language model stores a compound concept such as "washing machine" as its own dedicated internal representation, or whether the model builds that concept from smaller components as the text unfolds. The experiments focus on GPT-2 because it is small enough for detailed mechanistic analysis and has widely used sparse autoencoder resources, while still being large enough to show recognizable language behavior.

The main result is that compound concepts are represented differently depending on the level of analysis. In the residual stream, compound representations stay very close to their component representations. Across 21 compounds, the mean cosine similarity between the compound and the head noun alone is 0.954. This suggests that GPT-2 usually does not create a clean, isolated residual stream direction for the full compound. Instead, it modifies an existing component representation.

The sparse autoencoder results reveal structure that cosine similarity hides. About 56 percent of active SAE features for compounds are unique to the compound context, meaning they are not active for either component alone. This is especially strong for idiomatic compounds such as "red herring" and "dark horse". The model can therefore keep the overall residual direction close to the head noun while still changing the sparse feature pattern in a compound specific way.

The next token results show why a dedicated direction is often unnecessary. After the prompt "The washing", GPT-2 predicts "machine" as the top next token with probability 0.471, which is a 4,221 fold increase over baseline. Much of the model's practical knowledge of the phrase "washing machine" is therefore expressed through sequential prediction. The model does not need to hold the whole concept in a single direction if the modifier reliably makes the head very likely.

The later experiments broaden this picture. The minority continuation analysis shows that dominant compound heads do not always win. In 6 of 14 analyzable compounds, plausible minority heads such as "school board", "coffee shop", or "office building" beat the original target head under minimal context. The contextual override experiment then shows that richer sentence context can redirect the model away from a default continuation. GPT-2 Small rescues 11 of 14 tested minority continuations, and GPT-2 Medium rescues all 14 of 14.

## 2. Goal

**Hypothesis**: In large language models, there are too many referenceable concepts for each one to have a unique or nearly orthogonal direction in the residual stream. This research investigates whether specific compound concepts like "washing machine" are explicitly represented, or whether the model mainly stores component level information and uses context to make the head word more likely.

**Why this matters**: With only 768 dimensions in GPT-2's residual stream but millions of referenceable concepts, understanding how compound concepts are encoded reveals fundamental principles of how LLMs organize semantic knowledge. This has implications for interpretability, feature editing, and understanding the limits of current SAE-based analysis.

**Contribution**: Prior work has studied superposition (Elhage et al. 2022), feature absorption in SAEs (Chanin et al. 2024), and compound noun semantics in BERT models (Ormerod et al. 2024). This project connects those threads by measuring how compound concepts are decomposed by SAEs and how much next token prediction contributes to compound concept storage.

## 3. Data Construction

### Dataset Description
We used a custom dataset of 21 compound concepts spanning a compositionality spectrum:
- **Source**: Hand-curated for this study, based on linguistically-motivated categories
- **Compositionality scale**: very_low (idiomatic, e.g., "red herring") to very_high (transparent, e.g., "kitchen chair")
- **Categories**: appliances, food, vehicles, idioms, furniture, clothing, etc.

### Example Samples

| Compound | Components | Category | Compositionality |
|----------|-----------|----------|-----------------|
| washing machine | washing, machine | appliance | medium |
| hot dog | hot, dog | food | low |
| red herring | red, herring | idiom | very_low |
| kitchen chair | kitchen, chair | furniture | very_high |
| swimming pool | swimming, pool | structure | high |

### Tokenization
GPT-2's BPE tokenizer handles these compounds in several different ways:

| Compound | Tokenization | Note |
| --- | --- | --- |
| washing machine | `['washing', ' machine']` | 2 tokens, clean split |
| coffee machine | `['co', 'ffee', ' machine']` | 3 tokens, modifier split |
| red herring | `['red', ' her', 'ring']` | 3 tokens, head split |
| kitchen chair | `['kit', 'chen', ' chair']` | 3 tokens, modifier split |

This tokenization means that for multi-token components, the model must already do compositional work just to represent the component, adding complexity to the analysis.

## 4. Experiment Description

### Methodology

#### High-Level Approach
We conducted three complementary experiments on GPT-2 Small (124M parameters, 768 dimensions, 12 layers) using TransformerLens for activation access and pre-trained SAEs (24,576 features per layer) from jbloom/GPT2-Small-SAEs-Reformatted.

#### Why GPT-2 Small?
- Well-studied model with extensive SAE coverage
- Pre-trained SAEs available for all layers
- Computationally tractable for comprehensive analysis
- Provides a tractable first testbed before moving to larger models

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | Tensor computation |
| TransformerLens | 2.17.0 | Model internals access |
| Manual SAE | Custom | SAE inference (weights from HuggingFace) |
| matplotlib and seaborn | 3.10.8 and 0.13.2 | Visualization |
| scipy | 1.15.3 | Statistical tests |

#### Hardware
- GPU: NVIDIA RTX A6000 (48GB)
- Execution time: ~3 minutes total

### Experimental Protocol

#### Experiment 1: Residual Stream Cosine Similarity
For each compound concept:
1. Process "The [compound]" through GPT-2, extract residual stream at all 12 layers
2. Process "The [modifier]" and "The [head]" separately
3. Compute cosine similarity between:
   - Compound's last token vs. head noun alone (same word, different context)
   - Compound's last token vs. additive composition (average of separate components)
   - Compound's last token vs. modifier alone

#### Experiment 2: SAE Feature Analysis
Using pre-trained SAEs at layers 1, 6, and 11:
1. Encode residual stream activations through SAE
2. Identify active features (activation > 0) for compound and components
3. Compute:
   - Fraction of compound features unique to compound context
   - Overlap with head-alone and modifier-alone features
   - Jaccard similarity between feature sets
   - How modifier features change when in compound context

#### Experiment 3: Next Token Prediction and Compositional Priming
For each compound "modifier head":
1. Compute P(head | "The modifier"), the probability of the head noun after seeing the modifier
2. Compute P(head | "The"), the baseline probability
3. Compute priming ratio as P(head given modifier) divided by P(head given baseline)
4. Record rank of head noun in next-token predictions

### Raw Results

#### Experiment 1: Residual Stream Similarity (Layer 11)

| Compound | Comp. Level | vs Head Alone | vs Additive | vs Modifier |
|----------|------------|---------------|-------------|-------------|
| washing machine | medium | 0.951 | 0.954 | 0.812 |
| coffee machine | high | 0.957 | 0.967 | 0.844 |
| hot dog | low | 0.936 | 0.943 | 0.761 |
| red herring | very_low | 0.895 | 0.925 | 0.748 |
| kitchen chair | very_high | 0.956 | 0.968 | 0.831 |
| dark horse | very_low | 0.936 | 0.947 | 0.777 |
| school bus | high | 0.983 | 0.966 | 0.823 |
| guinea pig | low | 0.942 | 0.938 | 0.760 |
| swimming pool | high | 0.961 | 0.965 | 0.810 |
| ice cream | medium | 0.937 | 0.943 | 0.762 |

**Mean across all 21 compounds**: vs head alone = 0.954 +/- 0.019, vs additive = 0.959 +/- 0.013

#### Experiment 2: SAE Feature Decomposition (Layer 11)

| Compound | Comp. Level | Unique to Compound | Overlap with Head | Overlap with Modifier | Jaccard with Head |
|----------|------------|-------------------|----------------|-------------------|----------------|
| washing machine | medium | 57.9% | 35.1% | 24.6% | 0.256 |
| coffee machine | high | 54.7% | 34.4% | 20.3% | 0.265 |
| hot dog | low | 66.7% | 23.5% | 15.7% | 0.152 |
| red herring | very_low | **82.1%** | 14.3% | 7.1% | 0.079 |
| dark horse | very_low | **78.3%** | 19.3% | 13.3% | 0.147 |
| kitchen chair | very_high | 46.9% | 34.7% | 32.7% | 0.205 |
| dish washer | high | **21.5%** | 69.2% | 27.7% | 0.479 |
| school bus | high | 48.9% | 46.7% | 28.9% | 0.333 |
| guinea pig | low | 61.7% | 31.7% | 18.3% | 0.209 |
| swimming pool | high | 53.7% | 33.3% | 20.4% | 0.231 |

**Mean across all 21 compounds**: Unique fraction = 0.563 +/- 0.131, Jaccard with head = 0.255 +/- 0.091

#### Experiment 3: Next-Token Priming

| Compound | P(head after modifier) | Baseline P(head) | Priming Ratio | Rank |
|----------|-------------------|-----------------|---------------|------|
| washing machine | **0.4711** | 0.000112 | **4,221x** | **#1** |
| vending machine | **0.5416** | 0.000112 | **4,853x** | **#1** |
| sewing machine | **0.3741** | 0.000112 | **3,352x** | **#1** |
| swimming pool | **0.3184** | 0.000079 | **4,022x** | **#1** |
| guinea pig | **0.6624** | 0.000024 | **28,037x** | **#1** |
| ice cream | 0.0743 | 0.000020 | 3,757x | #3 |
| slot machine | 0.0624 | 0.000112 | 559x | #2 |
| hot dog | 0.0307 | 0.000127 | 242x | #5 |
| dark horse | 0.0097 | 0.000067 | 144x | #8 |
| red herring | 0.0019 | 0.000434 | **4.3x** | #67 |
| kitchen chair | 0.0012 | 0.000064 | 19x | #83 |
| office desk | 0.0000 | 0.000017 | **1.3x** | #1429 |
| rain coat | 0.0001 | 0.000015 | 3.9x | #888 |

**Top 5 predictions after "The washing"**: `machine` (47.1%), `of` (8.4%), `-` (8.2%), `machines` (4.9%), `up` (3.8%)

## 5. Result Analysis

### Key Findings

**Finding 1: The residual stream does not show clean dedicated compound directions.** All compounds show high cosine similarity with their head nouns processed in isolation. The mean similarity of 0.954 indicates that the compound representation is better described as a modification of the head noun representation than as a clearly separated direction. The additive composition of the modifier and head is even closer on average, with mean similarity 0.959, which suggests that compound representations often sit near their components in representation space.

**Finding 2: SAE features reveal compound specific computation despite high residual similarity.** Cosine similarity makes the compound and the head noun look almost the same, but the sparse feature decomposition is much more sensitive. Across the compounds, 56.3 percent of SAE features active for the compound are not shared with either component alone. This means the model can make a subtle change in residual direction while producing a noticeably different sparse feature pattern.

**Finding 3: Compositionality predicts how unique the sparse feature pattern becomes.** Low compositionality compounds such as "red herring" and "dark horse" have substantially more unique SAE features than high compositionality compounds. The unique feature fraction is 71.2 percent for the low compositionality group and 48.2 percent for the high compositionality group. The Mann Whitney U test gives p = 0.0005 with Cohen's d = 2.54, which is a very large effect. This fits the linguistic intuition that idioms need more specialized representation because their meaning cannot be recovered from their parts alone.

**Finding 4: "Washing machine" is largely constructed through next token priming.** After seeing "The washing", GPT-2 assigns 47.1 percent probability to "machine" as the next token, ranked first. This is a 4,221 fold increase over baseline. The model therefore stores much of the practical knowledge of the phrase in its transition behavior. It does not need a dedicated "washing machine" direction if the sequential context already makes "machine" overwhelmingly likely.

**Finding 5: Priming strength varies widely across compounds.** Some modifiers behave almost like fixed cues. "Washing", "vending", "sewing", "swimming", and "guinea" produce very strong expectations for their familiar heads. Other modifiers, such as "red", "rain", and "office", have many common continuations and therefore produce much weaker priming. This variation matters because it predicts whether a compound behaves like a rigid memorized phrase or a flexible composition.

### Hypothesis Testing Results

The compositional representation hypothesis is supported. Cosine similarity with additive composition averages 0.959, which means compound representations are often well approximated by combining the component representations. The Spearman correlation between compositionality and additive similarity is r = 0.602 with p = 0.004.

The dedicated SAE feature hypothesis is partially supported. A large share of features are unique to the compound context, but these features activate alongside component related features rather than replacing them. The compound representation looks like an augmented version of its components, not a completely separate representation.

The compositional priming hypothesis is strongly supported for most compounds. The median priming ratio is 99.2 fold. For "washing machine" specifically, "machine" is the top prediction after "washing" with probability 47 percent.

The layer dynamics hypothesis is partially supported. The fraction of features unique to the compound context generally increases from early to late layers, which suggests that compound specific computation builds across the network. Residual stream cosine similarity remains high even in later layers, so the directional modification stays subtle.

### Surprises and Insights

1. **"Guinea pig" has the highest priming ratio, 28,037 fold.** The modifier "guinea" almost exclusively predicts "pig", making it the most frozen compound in the dataset, even though the phrase is rated as low compositionality.

2. **"Dish washer" is an outlier.** It has only 21.5 percent unique features and 69.2 percent overlap with "washer" alone. This suggests the model treats "dish washer" almost identically to "washer", which is sensible because a dishwasher is a kind of washer.

3. **"Red herring" has the most unique features but weak priming.** The unique feature fraction is 82.1 percent, while the priming ratio is only 4.3 fold. This makes sense because "red" predicts many possible heads, but once "herring" appears, the model activates a specialized feature pattern for the idiomatic meaning.

4. **Cosine similarity is a blunt instrument.** Even "red herring", a very low compositionality compound, has 0.895 cosine similarity with "herring" alone. The SAE decomposition reveals much more structure, since only 14.3 percent of the relevant features overlap with the head noun alone.

## 5a. Minority-Continuation Analysis

### Motivation

The original next token experiment asks a one sided question: given a modifier, how strongly does the model predict the expected compound head? That does not tell us whether the model is simply memorizing the dominant continuation or whether plausible alternative heads receive meaningful probability mass. For example, after "washing", the model strongly predicts "machine", but it may still assign nontrivial probability to legitimate continuations such as "powder", "method", or "cycle".

### Design

For each of 15 compounds in the dataset, we compare the target compound head against several hand-curated minority heads (plausible alternative continuations after the same modifier) and unrelated control heads. We use five minimal prompt templates ("The {modifier}", "A {modifier}", etc.) and measure average next-token probability and rank for each head type. This follows the methodology outlined in the project roadmap (Item 2: "Check if minority class instances are where the model fails").

### Key Results

The central result is that compound dominance is not universal. In 6 of 14 analyzable compounds, the best minority head beats the original target compound head under minimal context. This is most visible for generic modifiers. After "coffee", the model prefers continuations such as "shop" over "machine". After "school", it gives substantial probability to "year", "district", and "board". After "office", it prefers continuations such as "building" and "manager" over "desk".

| Dominance pattern | Compounds | Example |
| --- | --- | --- |
| Target strongly dominant (ratio > 30x) | washing machine (183x), vending machine (796x), swimming pool (32x) | After "The washing", P("machine") = 0.61, P("powder") = 0.003 |
| Target moderately dominant (4-20x) | ice cream (18x), time machine (6x), sewing machine (6x), parking lot (5x), hot dog (4x) | After "The hot", P("dog") = 0.06, P("water") = 0.01 |
| Minority beats target (ratio < 1x) | coffee (0.1x), school (0.3x), dark (0.4x), fire (0.6x), kitchen (0.03x), office (0.01x) | After "The coffee", P("shop") = 0.11, P("machine") = 0.007 |

This result changes the interpretation of the original priming experiment. Massive priming for "washing machine" is real, but it is not representative of every compound. For generic modifiers like "office", "kitchen", and "school", the model does not strongly commit to a single familiar compound head. Instead, it distributes probability across several plausible continuations.

The minority heads are not just random noise. The median ratio of best minority head probability to best unrelated head probability is 1,798 fold in GPT-2 Small. Even when minority heads lose to the target compound head, they remain orders of magnitude above unrelated controls. This confirms that the model recognizes them as plausible continuations.

The pattern replicates on GPT-2 Medium. Minority heads again beat the target compound head in 6 of 14 analyzable compounds. The median target to best minority ratio drops from 4.65 fold in GPT-2 Small to 1.84 fold in GPT-2 Medium, while best minority heads remain far above unrelated controls. This suggests that the dominance spectrum is not an artifact of GPT-2 Small. If anything, the larger model appears to allocate more probability mass to plausible minority continuations.

| Model | Minority beats target | Median target to minority ratio | Median minority to unrelated ratio |
| --- | ---: | ---: | ---: |
| GPT-2 Small | 6 of 14 | 4.65x | 1,798x |
| GPT-2 Medium | 6 of 14 | 1.84x | 1,579x |

As a prompt-robustness spot check, excluding the indefinite-article template ("A {modifier}") preserves the qualitative result. GPT-2 Small still shows minority heads beating targets in 6 of 14 analyzable compounds, and GPT-2 Medium increases to 8 of 14. This reduces concern that the result is driven by awkward article choices such as "A ice" or "A office."

### Connection to Compositionality

The dominance spectrum aligns with the compositionality scale from the original analysis. Compounds that the model treats as strongly dominant tend to be those with unique modifier-head pairings (e.g., "vending" almost exclusively pairs with "machine"), while compounds where minorities win tend to have generic modifiers that participate in many compounds (e.g., "office" pairs with desk, chair, building, space, manager, etc.).

## 5b. Contextual Override Experiment

### Motivation

The minority-continuation analysis (Section 5a) reveals that compound dominance varies enormously: some compounds like "washing machine" completely suppress alternatives, while others like "coffee machine" lose to minority heads. But even in the dominant cases, this was measured under minimal context ("The washing"). A deeper question remains: for those compounds where the target head dominates, can richer sentence-level context rescue the suppressed minority continuations?

### Design

For each modifier we construct three context conditions. The "target" in this experiment means the model's dominant/default head under minimal context, which is the original compound head for most items but can also be a stronger default continuation discovered by the minority-continuation analysis.

| Condition | Example | Expected effect |
| --- | --- | --- |
| Neutral | "The washing" | Model defaults to dominant head |
| Reinforcing | "He repaired the washing" | Context reinforces dominant head |
| Rescuing | "She poured the washing" | Context should steer toward minority head ("powder") |

We test 14 modifier-minority pairs across 9 modifiers (washing, coffee, swimming, parking, fire, ice, school, hot, dark), measuring P(target_head) and P(rescued_minority) under each condition. The key metric is **override strength**: how many log-units the rescuing context shifts the target/minority probability ratio relative to neutral.

### Key Results

Disambiguating context rescues minority continuations in most tested cases. Of 14 modifier and minority head pairs, 11 show a successful rescue in GPT-2 Small, meaning the minority head probability exceeds the target head probability under the rescuing context. The mean minority head probability rises by 1,360 percent from neutral to rescuing context, while the target head probability falls by 60 percent. The Wilcoxon test gives W = 105 and p = 6.1e-5.

The contextual override effect becomes stronger in GPT-2 Medium. Repeating the same evaluation rescues all 14 of 14 tested minority continuations. Mean override strength rises to 2.245 log10 units, minority head probability increases by 1,720 percent, and target head probability falls by 72 percent. This extends the GPT-2 Medium robustness result beyond minimal context and suggests that the larger model is even more responsive to sentence level disambiguation.

| Model | Rescue successes | Mean override strength | Minority prob change | Target prob change |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 Small | 11 of 14 | 1.990 | +1,360% | -60% |
| GPT-2 Medium | 14 of 14 | 2.245 | +1,720% | -72% |

The strongest rescues are the cases where the sentence context makes the minority continuation highly natural and sharply reduces the default compound continuation.

| Modifier to minority | Override strength | Neutral P(min) | Rescue P(min) |
| --- | --- | --- | --- |
| dark to room (vs horse) | +3.04 | 0.004 | 0.072 |
| coffee to beans (vs shop) | +2.95 | 0.008 | 0.096 |
| swimming to lessons (vs pool) | +2.56 | 0.002 | 0.174 |
| ice to hockey (vs cream) | +2.48 | 0.019 | 0.546 |
| parking to ticket (vs lot) | +2.38 | 0.022 | 0.594 |

The three GPT-2 Small failures are also informative because they show where default compound priming remains hard to overcome.

| Modifier to minority | Override strength | Why rescue failed |
| --- | --- | --- |
| washing to powder | +1.53 | "machine" dominance is extreme (P=0.83); even "She poured the washing" only shifts "powder" to P=0.04 |
| washing to instructions | +2.52 | Override strength is high but "instructions" is too implausible a next token in any context |
| parking to garage | +0.18 | "garage" and "lot" both denote parking structures; the rescuing prompts do not sufficiently disambiguate |

### Interpretation

These results suggest that GPT-2's compound knowledge is not merely a rigid bigram reflex in the tested contexts. The model can integrate sentence level semantic cues to override its default compound priming in most cases. The failures are informative: extremely high PMI compounds like "washing machine" create a priming effect so strong that even semantically appropriate context cannot fully overcome it. This parallels the finding in the original report that compounds with higher co-occurrence statistics show stronger priming, and it extends that finding by showing that priming strength also predicts resistance to contextual override.

### Connecting Priming Strength to Override Resistance

Cross-referencing the original next-token priming ratios (Experiment 3) with contextual override outcomes reveals a clear pattern: compounds with stronger original priming are harder to override.

| Modifier/default head | Original/default tendency | Override outcomes |
| --- | --- | --- |
| washing machine | 4,963x | Both minorities failed to rescue |
| swimming pool | 549x | Both minorities successfully rescued |
| hot dog | 45x | Rescued |
| parking lot | 28x | 1 of 2 rescued |
| coffee to shop | default head discovered in Section 5a | Both minorities successfully rescued |

The only compound where both rescue attempts failed, "washing machine", also has by far the highest original priming ratio. This suggests a unified account: the same co-occurrence statistics that produce extreme next token priming also create representations that are resistant to contextual modulation. In other words, the model's confidence in a compound is not a binary property. It is a continuous variable that helps determine both how strongly the compound is primed and how flexibly it can be overridden by context.

### Scope and Remaining Work

The original next token experiment mainly asked whether the model predicts the familiar compound head after a modifier. Section 5a addresses that concern by adding plausible minority continuations and unrelated controls. This extension shows that compound dominance is far from universal: in 6 of 14 tested compounds, a minority head beats the target compound head even under minimal context.

The original residual stream and SAE experiments use short prompts such as "The washing machine". Section 5b addresses context sensitivity by adding sentence level cues. The contextual override experiment shows that richer context can rescue minority continuations in 79 percent of tested GPT-2 Small cases and in all tested GPT-2 Medium cases. The remaining failed rescues are still useful because they identify compounds whose priming is unusually resistant to contextual evidence.

Tokenization remains an important detail. Some compounds split unevenly, such as "coffee", which tokenizes into `co` and `ffee`. The minority continuation extension records skipped multi token heads so the next token comparisons remain tokenization aware, but future work could use models with different tokenizers to test how much this affects the pattern.

The minority heads and contextual override prompts are manually selected. The unrelated controls make the comparison more informative than a one sided priming test, but a stronger version would derive alternatives from corpus PMI, noun compound sense annotations, or human plausibility judgments.

The SAE analysis should also be read with the usual caution. Pretrained SAEs can show absorption and polysemantic features, so exact unique feature percentages should not be treated as literal counts of concepts. The qualitative pattern is still informative because it consistently separates residual stream similarity from sparse feature differences.

The new next token extensions have been replicated on GPT-2 Medium, which reduces the single model concern for the minority continuation and contextual override results. The residual stream and SAE analyses are still primarily GPT-2 Small results. A natural next step is to repeat those analyses on larger open models and additional model families.

## 6. Conclusions

### Summary
The experiments do not find evidence that "washing machine" is stored as a clean dedicated direction in GPT-2's residual stream. Instead, the evidence supports three complementary mechanisms. First, the residual stream representation is a subtle modification of the "machine" representation when preceded by "washing", with cosine similarity 0.951. Second, that subtle modification activates a substantially different set of SAE features, with 57.9 percent unique to the compound context. Third, the model's next token predictions massively favor "machine" after "washing", with 47.1 percent probability and rank 1.

This account becomes stronger once minority continuations and contextual override are included. Dominant priming is not universal: for compounds with generic modifiers such as "office" and "kitchen", minority continuations can outcompete the target compound head. For compounds where a default head is strong, the default can still often be redirected by sentence context. GPT-2 Small rescues 11 of 14 tested minority continuations, and GPT-2 Medium rescues all 14. The model's compound knowledge therefore combines co-occurrence statistics with contextual semantics.

### Implications
For interpretability, the main lesson is that cosine similarity alone is insufficient. It makes compound representations look almost identical to their heads, while SAE decomposition reveals richer structure. This matters for any study that tries to locate concepts by looking for a single direction.

For the superposition hypothesis, compound concepts provide a concrete example of how language models manage the problem of too many concepts and too few dimensions. The model can construct multi token concepts sequentially, using a mixture of component representations, sparse feature changes, and next token expectations.

For practical probing, the results suggest that multi token concepts need special care. A probe that only asks whether a concept has a direction may miss the fact that much of the concept is expressed through the model's prediction dynamics. A probe that only measures next token probability may miss the sparse feature structure that appears once the full compound is present.

For compound priming, the results support a continuum. Extremely high PMI compounds such as "washing machine" resist contextual override, while less rigid compounds such as "dark horse" can be redirected more easily. The model's confidence in a compound is therefore gradual rather than binary.

### Confidence in Findings
High confidence in the core next-token priming findings (large effect sizes, consistent patterns across 21 compounds). The minority-continuation finding is strengthened by GPT-2 Medium replication: both models show minority heads beating targets in 6 of 14 analyzable cases, with minority heads far above unrelated controls. The contextual-override result is also strengthened by GPT-2 Medium replication, where the same setup rescues 14 of 14 minority continuations. The SAE analysis is moderately confident because SAE artifacts could affect exact percentages, though the overall pattern is unlikely to be explained by artifacts alone.

## 7. Next Steps

### Immediate Follow-ups
1. **Broader cross-model validation**: Extend the GPT-2 Medium replication to Llama-family or other open models, especially for residual-stream and SAE analyses.
2. **Corpus-derived minority heads**: Replace hand-curated minority continuations with heads derived from corpus PMI or noun-compound sense annotations to reduce experimenter bias.
3. **Causal interventions**: Use activation patching to ablate compound-specific SAE features and measure whether this disproportionately harms dominant vs. minority continuation predictions.
4. **Override threshold analysis**: For the three failed rescues, systematically increase contextual strength to identify the threshold at which the model can override extreme priming.

### Alternative Approaches
- Use Gemma Scope SAEs (up to 1M features) to see if higher-resolution decomposition reveals more dedicated compound features
- Apply feature channel coding (Adler et al. 2025) as an alternative to SAE-based analysis
- Probe attention heads to understand how the model integrates modifier information into the head noun position

### Open Questions
- At what model scale do compound concepts begin to earn dedicated directions?
- How does the model handle novel compounds it has never seen (e.g., "quantum washing")?
- Can the compound-specific SAE features be used to edit compound concept knowledge?

## References

- Elhage et al. (2022). "Toy Models of Superposition." arXiv:2209.10652
- Chanin et al. (2024). "A is for Absorption." arXiv:2409.14507
- Minegishi et al. (2025). "Rethinking SAE Evaluation via Polysemous Words." arXiv:2501.06254
- Ormerod et al. (2024). "How Is a Kitchen Chair like a Farm Horse?" Computational Linguistics.
- Miletic & Schulte im Walde (2023). "A Systematic Search for Compound Semantics in Pretrained BERT." EACL 2023.
- Merullo et al. (2023). "Language Models Implement Simple Word2Vec-style Vector Arithmetic." arXiv:2305.16130
- Adler et al. (2025). "Towards Combinatorial Interpretability." arXiv:2504.08842
- Giglemiani et al. (2024). "Evaluating Synthetic Activations composed of SAE Latents." arXiv:2409.15019
- Park et al. (2023). "The Linear Representation Hypothesis." arXiv:2311.03658
