# Dataset Download Summary

**Date:** 2026-02-07
**Dataset Directory:** `datasets/`

## Successfully Downloaded Datasets

### Dataset 1: Noun Compound Senses (NCS) - SUCCESS
- **Status:** Fully downloaded and verified
- **Source:** https://github.com/marcospln/noun_compound_senses
- **Location:** `datasets/noun_compound_senses/`
- **Size:** 567 KB
- **Contents:**
  - 280 English noun compounds with compositionality ratings
  - 180 Portuguese noun compounds
  - P1, P2, P3 sentence variants in neutral contexts
  - Input files with sentence IDs and compositionality scores

**Data Format:**
The dataset provides CSV files with the following structure:
- English data: `dataset/en/neutral/P1_sents.csv`, `P2_sents.csv`, `P3_sents.csv`
- Input with compositionality: `input/sentids_en.csv` (282 rows including header)
- Each compound includes: compound name, compositionality score, example sentences

**Sample Compounds with Compositionality Scores:**
```
academy award             3.52  (moderately compositional)
acid test                 1.22  (mostly idiomatic)
washing machine           (included in dataset)
application form          4.8   (highly compositional)
```

**Notes:**
- The dataset is based on Reddy et al. (2011) compositionality dataset
- Naturalistic sentences require downloading ukWaC corpus separately
- Currently contains neutral context sentences ready to use

---

### Dataset 2: Reddy et al. Compositionality Ratings - SUCCESS (via NCS)
- **Status:** Available within NCS dataset
- **Source:** Included in `noun_compound_senses/input/sentids_en.csv`
- **Location:** `datasets/noun_compound_senses/input/sentids_en.csv`
- **Size:** 282 entries (281 compounds + header)
- **Contents:**
  - 280+ noun-noun compounds
  - Human compositionality judgments
  - Example sentences from ukWaC corpus

**Data Format:**
CSV with columns: compound, compositionality, sentence1, sentence2, sentence3

**Note:**
The standalone HuggingFace dataset `LanguageToolsLab/compound_compositionality` does not exist. However, the original Reddy et al. (2011) data is incorporated into the NCS dataset and fully accessible.

---

### Dataset 3: MAGPIE Idiom Dataset - SUCCESS
- **Status:** Fully downloaded and verified
- **Source:** https://github.com/hslh/magpie-corpus
- **Location:** `datasets/magpie/`
- **Size:** 92 MB
- **Contents:**
  - 56,622 total instances of potentially idiomatic expressions
  - 1,756 different idiom types
  - Crowdsourced meaning labels from British National Corpus

**Files:**
1. `MAGPIE_filtered_split_random.jsonl` - 48,395 instances
   - 75% annotation confidence threshold
   - Binary labels (idiomatic/literal)
   - Random train/dev/test split

2. `MAGPIE_filtered_split_typebased.jsonl` - 48,395 instances
   - Same filtering as above
   - Type-based split (no idiom overlap between splits)

3. `MAGPIE_unfiltered.jsonl` - 56,622 instances
   - All annotations regardless of confidence
   - 5-way labels: i (idiomatic), l (literal), f (figurative), o (other), ? (unknown)

**Data Format:**
JSON Lines with fields:
- `idiom`: The potentially idiomatic expression (e.g., "off the beaten track")
- `label`: Sense label (i/l/f/o/?)
- `confidence`: Annotation confidence (0.0-1.0)
- `context`: List of surrounding sentences
- `label_distribution`: Distribution of annotator judgments
- `split`: train/dev/test designation
- `genre`: BNC genre classification
- `offsets`: Character positions in sentence

**Sample Entry:**
```json
{
  "idiom": "off the beaten track",
  "label": "i",
  "confidence": 1.0,
  "judgment_count": 3,
  "context": ["...", "how about some ideas for safe running off the beaten track?", "..."]
}
```

**Note:**
The HuggingFace version `gsarti/magpie` exists but uses deprecated dataset scripts. The GitHub repository provides the complete, usable dataset.

---

### Dataset 4: Custom Compound Nouns Test Set - SUCCESS
- **Status:** Created successfully
- **Location:** `datasets/compound_nouns_test.jsonl`
- **Size:** 4.5 KB (35 compounds)
- **Contents:**
  - 35 carefully selected compound nouns
  - Spans full compositionality spectrum (1-5 rating)
  - Hand-curated with explanatory notes

**Compositionality Distribution:**
- **Rating 5 (Fully compositional):** 8 compounds
  - Examples: "coffee table", "brick house", "mountain cabin", "garden hose"

- **Rating 4 (Partially compositional):** 9 compounds
  - Examples: "washing machine", "swimming pool", "parking lot", "driving license"

- **Rating 3 (Moderately compositional):** 6 compounds
  - Examples: "living room", "blackboard", "greenhouse", "redhead"

- **Rating 2 (Weakly compositional):** 4 compounds
  - Examples: "shooting star", "guinea pig", "strawberry", "eggplant"

- **Rating 1 (Non-compositional/Idiomatic):** 8 compounds
  - Examples: "hot dog", "butterfly", "deadline", "bookworm", "ladybug"

**Data Format:**
```json
{
  "compound": "washing machine",
  "word1": "washing",
  "word2": "machine",
  "compositionality_rating": 4,
  "notes": "Partially compositional - a machine for washing, but 'washing' is typically clothing-specific"
}
```

**Purpose:**
This dataset is specifically designed to:
- Test LLM representations across compositionality levels
- Include the target compound "washing machine" (rating: 4)
- Provide clear reference examples for each compositionality level
- Facilitate controlled experiments on compound noun understanding

---

## Supporting Files Created

### 1. datasets/README.md
Comprehensive documentation including:
- Description of all four datasets
- Data formats and structure
- Usage instructions
- Citation information
- Dataset statistics table
- License information

### 2. datasets/.gitignore
Git ignore rules for:
- Large corpus files (*.xml, *.conll)
- Cache and temporary files
- Generated data files
- System files

---

## Dataset Statistics Summary

| Dataset | Entries | Size | Language | Annotation Type | Format |
|---------|---------|------|----------|----------------|--------|
| NCS Dataset | 280 (EN), 180 (PT) | 567 KB | EN, PT | Compositionality + Variants | CSV |
| Reddy et al. | 281 | (in NCS) | EN | Compositionality ratings | CSV |
| MAGPIE (filtered) | 48,395 | 46 MB × 2 | EN | Idiomaticity (binary) | JSONL |
| MAGPIE (unfiltered) | 56,622 | 53 MB | EN | Idiomaticity (5-way) | JSONL |
| Custom Test | 35 | 4.5 KB | EN | Compositionality (1-5) | JSONL |
| **TOTAL** | **~105K instances** | **~92 MB** | **EN, PT** | **Multiple** | **CSV, JSONL** |

---

## Verification Results

All datasets have been successfully:
1. Downloaded and extracted
2. Verified for correct format
3. Sampled to confirm data integrity
4. Documented with README files
5. Protected with .gitignore rules

## Next Steps

Suggested research workflows:

1. **Baseline Analysis:**
   - Load custom test set (35 compounds)
   - Extract LLM representations
   - Correlate with compositionality ratings

2. **Large-Scale Validation:**
   - Use Reddy et al. data (281 compounds) for validation
   - Compare against human compositionality judgments
   - Use NCS dataset for cross-lingual analysis

3. **Idiomaticity Studies:**
   - Use MAGPIE filtered data for binary classification
   - Train/test on type-based split for generalization
   - Use unfiltered data for fine-grained analysis

4. **Comprehensive Evaluation:**
   - Combine all datasets for multi-faceted analysis
   - Focus on "washing machine" and similar partial compositional cases
   - Compare compositional vs. non-compositional representations

## Citations

1. **NCS Dataset:**
   Garcia, M., Vieira, T. K., Scarton, C., Idiart, M., & Villavicencio, A. (2021). Probing for idiomaticity in vector space models. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021).

2. **Reddy et al.:**
   Reddy, S., McCarthy, D., & Manandhar, S. (2011). An empirical study on compositionality in compound nouns. In Fifth International Joint Conference on Natural Language Processing, IJCNLP 2011, pages 210-218.

3. **MAGPIE:**
   Haagsma, H., Bos, J., & Nissim, M. (2020). MAGPIE: A Large Corpus of Potentially Idiomatic Expressions. In Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020), pages 279-287.

---

## File Locations

All dataset paths are relative to the repository root.

**Key files:**
- Reddy compositionality data: `noun_compound_senses/input/sentids_en.csv`
- MAGPIE filtered (random split): `magpie/MAGPIE_filtered_split_random.jsonl`
- MAGPIE filtered (type-based): `magpie/MAGPIE_filtered_split_typebased.jsonl`
- Custom test set: `compound_nouns_test.jsonl`
- Documentation: `README.md`
- Git ignore rules: `.gitignore`

**Status:** All datasets ready for use in LLM compound concept representation research.
