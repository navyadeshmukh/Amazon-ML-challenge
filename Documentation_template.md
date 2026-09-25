# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Slytherine  
**Team Members:** TODO  
**Submission Date:** TODO

---

## 1. Executive Summary
We solve entity resolution with a cascaded pipeline: text normalisation, high-recall blocking (TF-IDF top-K per country/source + optional multilingual embedding retrieval), and a LightGBM pair classifier over ~40+ features including string-similarity, address, embedding cosine, and "competition" features. The decision rule (probability threshold + exclusive-assignment) is tuned directly on the F_0.5 metric using out-of-fold predictions, with singletons scored as in the competition. The pipeline follows a precision-first design: broad recall during candidate generation, conservative precision during final matching.

---

## 2. Methodology

### 2.1 Problem Analysis
*Fill after EDA (numbers from the training data):*
- Train sizes: S1 = TODO, S2 = TODO, S3 = TODO; ground-truth links = TODO; singleton S1 share = TODO %.
- Observed name noise: TODO (e.g., legal-suffix differences, `&`/`and`, typos, word-order swaps, DBA names).
- Observed address noise: TODO (e.g., Rd/Road, missing PIN, landmark clauses such as "Near ...", component reordering).
- Country differences: TODO (India: PIN + landmarks; US: ZIP + suite/unit; France, unseen in training: accents, 5-digit code postal, `rue/av/bd`).
- Do S2/S3 records ever link to more than one S1? TODO (printed by the pipeline; drives the exclusive-assignment rule).

### 2.2 Solution Strategy
```
Normalise → Blocking (TF-IDF + Embeddings) → Pair features → LightGBM P(same) → Threshold + exclusive assignment → matches
```
**Approach Type:** Cascaded funnel — Blocking + Classifier (gradient-boosted trees) with optional pretrained-embedding features  
**Core Innovation:** (1) "competition" features, which let the model see whether a pair is clearly the best for both its S1 entity and its S2/S3 record, a strong lever for precision under F_0.5; (2) decoding parameters tuned on the true macro F_0.5 metric, including singletons; (3) optional multilingual embedding features from BGE-M3/E5 for cross-lingual robustness.

**Design Principle:** Eliminate cheaply and safely first; perform expensive comparisons only on the small set of plausible candidates; make the final match decision conservatively.

**Preprocessing (`src/normalize.py`)**
- Accent stripping, lowercasing, `&` to `and`, apostrophe removal, `M.G.` to `mg`.
- Names: legal suffixes removed (pvt, ltd, private, limited, inc, llc, corp, sarl, sas, sa, gmbh, ...), stop words removed, common abbreviations expanded.
- Addresses: Rd/St/Ave/Blvd/Opp/Nr etc. expanded; landmark clauses (`near ...`) removed from the "core" address but retained in the full address; PIN/ZIP extracted by a generic 5-6 digit regex.
- Country: lowercased with a small alias map, treated as an open set of labels (never one-hot, never hard-coded to US/India).
- Original columns are preserved; normalised columns are added.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** TF-IDF cosine nearest neighbours on three sparse views: name char 2-4-grams, name word 1-2-grams, name+address char 3-4-grams. Top-K (K = 25) per view, computed separately for Source 2 and Source 3, restricted to the same country label (falls back to all records if a country has no records in the other source). Optional dense embedding views (BGE-M3/E5) with top-15 per view. The candidate set is the union over all views.
- **Candidate pairs generated:** train = TODO; test = TODO (avg TODO per S1 entity). Reduction ratio = TODO.
- **How you ensured true matches were not lost:** blocking recall ceiling measured on training ground truth = TODO (printed as "blocking recall ceiling" and saved in `reports/metrics.json`). K and the set of views were chosen to push this above TODO. Union of complementary views (character-level for typos, word-level for reordering, name+address for weak names, dense embedding for semantic/transliteration) protects against any single view failing.
- `candidate_pairs.tsv` is exactly the set scored by the model at inference (no further filtering before scoring).

---

## 4. Matching Model

**Features used:**
- Name features: Levenshtein, Jaro-Winkler, space-free (compact) Levenshtein, partial ratio, token-sort, token-set, Jaccard, containment, first-token equality, acronym match, length difference; TF-IDF cosine (char and word); optional multilingual embedding cosine (BGE-M3/E5, MIT/Apache-2.0, used as-is).
- Address features: Levenshtein, token-set, partial ratio, Jaccard, containment, house-number agreement, city/state tail-token Jaccard, TF-IDF cosine (char and word), address-missing flag.
- Other: PIN/ZIP match (1 match / 0 conflict / -1 missing), PIN prefix match, country match, source indicator, and competition features (rank and gap of the pair versus other candidates of the same S1 entity, and versus other S1 entities competing for the same S2/S3 record; candidate counts on both sides).

**Model type:** LightGBM (MIT licence) binary classifier, lr 0.05, 63 leaves, feature/bagging fraction 0.8, early stopping on logloss; 5-fold GroupKFold grouped by S1 entity (no leakage), fold models averaged at test time. Far below the 8B-parameter limit; no fine-tuned models.  
**Threshold selection method:** grid search on out-of-fold probabilities over threshold 0.30-0.98 and over exclusive assignment on/off (each S2/S3 record kept only for its highest-probability S1, valid because S1 is deduplicated), maximising macro F_0.5 including singletons. Chosen: threshold = TODO, exclusive = TODO.

**Generalisation to France:** country enters only as a match flag and blocking partition; accent stripping, generic postal-code regex, French legal suffixes (SARL/SAS/SA/EURL) and street abbreviations are handled; an optional `--min-thr-safety` margin raises the threshold for extra precision on the unseen country. Leave-one-country-out (LOCO) validation simulates unseen France performance.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** OOF (train, 5-fold grouped) = TODO; per country: US = TODO, India = TODO; public leaderboard = TODO.
- **LOCO (leave-one-country-out):** US held-out = TODO; India held-out = TODO.
- **Common false positives (wrong merges):** TODO. Inspect `reports/oof_errors.tsv` (type = FP), typical candidates: chains/branches sharing a name in the same city, same name with different address numbers.
- **Common false negatives (missed matches):** TODO. Inspect type = FN (model rejected) and FN_not_in_candidates (blocking miss), typical candidates: DBA/trade names, heavily abbreviated addresses, missing PIN.
- **Ablations:** TODO (with/without embeddings; without competition features; K = 15/25/40; exclusive on/off).

---

## 6. Conclusion
TODO: 2-3 sentences after final run. Suggested content: a lightweight, fully reproducible tree-based pipeline with strong blocking and metric-aware decoding reached OOF F_0.5 of TODO; the largest gains came from TODO; future work is a cross-encoder re-ranker (MIT/Apache, <= 8B) for borderline pairs and hard-negative mining.

---

## Appendix

### A. Code Artefacts
Located in `code/business_entity_resolution/`:
| File | Purpose |
|---|---|
| `src/normalize.py` | Name / address / country / PIN normalisation |
| `src/blocking.py` | TF-IDF views + optional dense embedding top-K candidate generation |
| `src/features.py` | Pair features incl. embedding and competition features (parallel) |
| `src/embeddings.py` | Optional pretrained multilingual encoder wrapper |
| `run_pipeline.py` | **Entry point**: train, OOF tuning, LOCO, inference, writes both TSVs, `reports/` |
| `src/make_submission_zip.py` | Builds `<team>_submission.zip` |
| `README.md`, `requirements.txt` | Run instructions, pinned versions |

Reproduce: `pip install -r code/requirements.txt && cd code/business_entity_resolution && python run_pipeline.py` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Seed 42. Runtime: TODO. Hardware: TODO.

### B. Additional Results
TODO: blocking recall vs K, threshold vs F_0.5 curve (see `reports/threshold_sweep.tsv`), feature importance (top features are printed by the pipeline), per-country results, LOCO results.

### C. Compliance
No external lookups, geocoding, or ER APIs. Libraries: LightGBM (MIT), RapidFuzz (MIT), scikit-learn/pandas/numpy/scipy (BSD), sentence-transformers (Apache-2.0, optional), torch (BSD-3, optional). All models < 8B parameters.
