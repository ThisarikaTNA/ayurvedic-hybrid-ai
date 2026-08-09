# Explainable Hybrid AI for Ayurvedic Knowledge Profiles

This repository contains an MSc coursework research prototype combining:

- multi-label prediction of dataset-assigned Vata, Pitta and Kapha tags;
- a provenance-aware SQLite knowledge base and deterministic rule engine; and
- condition-scoped retrieval of similar knowledge profiles and checked claims.

It is not a diagnostic, prescribing or clinically validated system. Dataset rows are knowledge profiles, not historical patients. Model outputs are agreement estimates for dataset-assigned tags, not a person's medically true Dosha.

## Frozen coursework result

The final selected model is a symptoms-only one-vs-rest Logistic Regression pipeline with `C=0.5`, balanced class weights and thresholds of `0.45` for Vata, Pitta and Kapha. Phase 10 evaluated this frozen pipeline once on 67 held-out profiles:

| Metric | Result |
|---|---:|
| Macro-F1 | 0.626241 |
| Micro-F1 | 0.697479 |
| Weighted-F1 | 0.706196 |
| Exact-match accuracy | 0.238806 |
| Hamming loss | 0.358209 |

The final evaluation includes a disclosed pre-execution metadata exposure and two pre-execution aborts. Perfect pre-unsealing blindness is not claimed. See `docs/phase10_final_test_evaluation_report.md`.

## Functional application

The functional application is the deterministic Phase 9 script-based hybrid pipeline. It requires a caller-selected condition and structured safety facts, applies the safety gate before ML personalization, invokes the frozen model only when permitted, performs condition-scoped retrieval and returns provenance-separated explanation traces.

The coursework brief does not require a graphical interface. The current command-line demonstration therefore provides functional application evidence, although a demonstration video is still a required student deliverable.

Run demonstrations only in a disposable copy because the script refreshes Phase 9 output files. It never writes user data or predictions to the production database.

```powershell
python scripts/run_phase9_hybrid_integration.py
```

Expected result: all 23 predetermined synthetic scenarios pass, with production `model_predictions`, `retrieval_results` and `final_recommendations` remaining empty.

## Quick setup

Python 3.12 is required. The frozen evaluation used Python 3.12.7 and scikit-learn 1.5.1.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
```

Do not run `scripts/run_phase10_final_test_evaluation.py`. The Phase 10 result is final and its completion marker intentionally prevents reruns.

Detailed setup and PyCharm instructions are in `docs/PYCHARM_RUN_GUIDE.md`.

## Project structure

```text
config/                     Frozen component and evaluation configuration
data/raw/                   Immutable supplied dataset copies
data/processed/             Reproducible cleaned data and preprocessing evidence
data/knowledge_base/        Production SQLite prototype database
docs/                       Phase reports and submission documentation
models/phase4_validation/   Frozen selected model bundle and candidate artifacts
outputs/                    Machine-readable phase evidence and manifests
reports/figures/            Validation figures
schemas/                    Rule, integration and evaluation JSON Schemas
scripts/                    Reproducible phase and demonstration entry points
sql/                        SQLite schema, indexes, triggers and migrations
src/data/                   Preprocessing
src/models/                 Group splitting, training and validation comparison
src/knowledge_base/         Database, seeding and deterministic rules
src/retrieval/              Condition resolution and profile retrieval
src/integration/            Phase 9 safety-first hybrid orchestration
src/evaluation/             Guarded Phase 10 evaluator; historical, do not rerun
tests/                      Synthetic and integrity regression tests
```

See `docs/PROJECT_ARCHITECTURE.md` for the component and data-flow explanation.

## Reproducible commands

All commands assume the project root is the working directory and the editable installation above is active.

```powershell
# Complete synthetic/integrity regression suite
python -m pytest -q

# Read-only production-database integrity tests
python -m pytest -q tests/test_knowledge_base.py tests/test_rule_engine.py

# Phase 9 synthetic hybrid demonstration in a disposable project copy
python scripts/run_phase9_hybrid_integration.py
```

Earlier data-processing and model-development scripts remain for auditability, but they should not be rerun as part of the final demonstration. Rerunning model-development phases is unnecessary and would not change the accepted frozen result.

## Implemented phases

1. Dataset structure and provenance audit.
2. Leakage-aware preprocessing with preserved raw values.
3. Disease-group-disjoint train, validation and final-test split.
4. Validation comparison of dummy, Logistic Regression and Random Forest pipelines across two feature experiments.
5. Reproducible selection of five knowledge-base conditions.
6. Normalized, provenance-aware SQLite knowledge base.
7. Structured deterministic IF-THEN rule engine.
8. Condition-scoped FTS5/BM25 and Dosha-overlap retrieval.
9. Safety-first deterministic hybrid integration on synthetic scenarios.
10. One-time final evaluation of the frozen ML component.
11. Submission audit and non-destructive packaging.

## Important limitations

- The supplied data contains disease knowledge profiles rather than independent clinical encounters.
- Dataset-assigned Dosha relationships may be templated, conflicting or medically unverified.
- The final test contains only 67 grouped profiles; Kapha performance is weak.
- Raw model probabilities are not calibrated clinical confidence.
- Reference checking is not expert review or clinical validation.
- The knowledge base contains five referral-information and five general-information rules, but no contraindication or exclusion rules and no expert-reviewed rules.
- Passing the limited safety gate does not establish comprehensive medical safety.
- The metadata exposure recorded in Phase 10 means perfect pre-unsealing blindness cannot be claimed.

## Submission status

`SUBMISSION_MANIFEST.md` identifies included files, exclusions and outstanding student actions. Major outstanding rubric deliverables are a current two-page project description, a 4,000-5,000-word PDF technical report, an implementation notebook if the lecturer enforces the notebook format literally, and a 5-10-minute demonstration video.
