# Business Entity Resolution — Team Slytherine

Pipeline: normalise → blocking (TF-IDF top-K + optional dense embeddings, per country, per source) → pair features
→ LightGBM (5-fold grouped OOF) → F0.5 threshold + exclusive-assignment tuning → outputs.

## Setup
```bash
pip install -r code/requirements.txt
# Optional (for --embedder flag):
pip install -r code/requirements-embeddings.txt
```

## Quick Start
```bash
cd code/business_entity_resolution

# Fast dev loop (k=15, 3 folds):
python run_pipeline.py --quick

# Full baseline (TF-IDF + string features):
python run_pipeline.py --loco

# With multilingual embeddings (recommended, needs GPU):
python run_pipeline.py --embedder e5small --device cuda --loco

# BGE-M3 embeddings (best for multilingual):
python run_pipeline.py --embedder bgem3 --device cuda --loco
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, plus
`reports/metrics.json`, `reports/oof_errors.tsv`, and `reports/threshold_sweep.tsv`.

## Options
| Flag | Meaning |
|---|---|
| `--quick` | k=15, 3 folds (dev loop) |
| `--embedder e5small` | Add multilingual sentence-embedding features + dense blocking view (`e5base`, `bgem3`, `minilm` also available; needs `requirements-embeddings.txt`) |
| `--device cuda` | Use GPU for embeddings (default: auto-detect) |
| `--loco` | Leave-one-country-out check (proxy for unseen France) |
| `--min-thr-safety 0.03` | Extra precision margin at test time |
| `--k 25` | Top-K per sparse blocking view (default: 25) |
| `--kd 15` | Top-K per dense blocking view (default: 15) |
| `--folds 5` | Number of CV folds (default: 5) |
| `--n-jobs -1` | Parallel feature workers (default: all cores) |

`./run_all.ps1 -Team Slytherine` runs the whole recommended workflow on Windows.

## Validate Before Uploading
```bash
cd student_resource
python utils/validate_submission.py \
  --matching ../Amazon-ML-challenge/output/matching_results.tsv \
  --candidate ../Amazon-ML-challenge/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Files
| File | Purpose |
|---|---|
| `code/business_entity_resolution/src/normalize.py` | Name / address / country / PIN normalisation (originals kept) |
| `code/business_entity_resolution/src/blocking.py` | TF-IDF views + optional dense embedding top-K candidate generation |
| `code/business_entity_resolution/src/features.py` | String-similarity (parallel), address, PIN, embedding and competition features |
| `code/business_entity_resolution/src/embeddings.py` | Optional pretrained multilingual encoder (used as-is, no fine-tuning) |
| `code/business_entity_resolution/run_pipeline.py` | **Entry point**: train, OOF tuning, LOCO, inference, file writing |
| `code/business_entity_resolution/src/make_submission_zip.py` | Builds `<team>_submission.zip` |
| `code/requirements.txt` | Core dependencies (pinned versions) |
| `code/requirements-embeddings.txt` | Optional embedding dependencies |

Seeds are fixed (`SEED=42`). No external data / APIs are used.

## Build the Submission Zip
```bash
cd code/business_entity_resolution
python src/make_submission_zip.py --team Slytherine
```

## Reports Generated
| Report | Contents |
|---|---|
| `reports/metrics.json` | Blocking recall, reduction ratio, OOF F0.5, per-country, LOCO, threshold, args |
| `reports/oof_errors.tsv` | False positives / false negatives (OOF) with names, addresses, probabilities |
| `reports/threshold_sweep.tsv` | Full threshold × exclusive sweep: F0.5, P, R, singleton acc at each point |
