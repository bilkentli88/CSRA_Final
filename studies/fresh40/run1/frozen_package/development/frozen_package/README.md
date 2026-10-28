# CSRA development execution and analysis

Use this package for the existing 100 items only. It prepares the stronger main study by collecting unquantized development scores, fitting independent input/observer baselines and simulating fresh-sample precision. It does not execute fresh items, establish journal readiness, or supply new human factual reviews.

## What to run

1. Open CSRA_Development.ipynb in Colab. Select the A100 GPU used for qualification.
2. Run section 1, mount Drive and upload CSRA_Development.zip when requested. The existing 100-item input and verified hardware profiles are already included. Do not upload the earlier pilot ZIP or rerun hardware qualification.
3. Run section 2 to install the pinned stack and initialize the development folder.
4. Run section 3. It audits all possible prompt/label continuations before scoring, then runs all three known-target answer stages, derives new FP32 targets, and runs all four reporters. The reference estimate is about 2.9 A100 GPU-hours near 512 tokens, plus setup/auditing/loading/retries. This is not a promise of elapsed time. Progress stays in MyDrive/CSRA_Development_runs/run1. Do not run that same folder from two runtimes simultaneously.
5. Run section 4 for CPU embeddings, grouped predictive analysis and sample-size scenarios. It uses 1,000 simulated datasets and 1,999 bootstrap draws for each base metric/N combination; variance and location scenarios reuse the same draws mathematically. This CPU time is extra. You may switch to a CPU runtime after section 3: rerun sections 1–2, skip section 3 and run section 4. Drive will restore the frozen package without another upload.
6. Run section 5 and return CSRA_Development_results.zip.

On interruption, rerun sections 1–2 in the same Drive folder and resume the unfinished section. Committed model scores, finished prediction folds and completed calibration combinations are reused. An unfinished final checkpoint line is preserved separately and only its uncommitted bytes are removed. Malformed committed records, changed package hashes, duplicates and nonfinite scores stop the run. A failed command displays its actual traceback. Keyboard interruption terminates the notebook's active subprocess.

Section 3 scores 14,400 answer jobs and 28,800 report jobs. It uses no old NF4 answer/report scores. Mistral reports on the three known targets but never supplies a development answer-substitution target. The runner accepts no new dataset path and rejects Mistral as an answer target.

## Factual review remains real work

The inherited items.csv is preserved byte for byte. Its APPROVE entries are historical fields with unverified independent-review provenance, not completed new reviews. The provisional development diagnostics and draft training artifact are labeled accordingly. No confirmatory release or final fresh-data freeze is claimed.

The review folder provides a deterministically selected 20-item timing sample (five per relation), an empty two-reviewer form and coordinator key. Two identified humans should review factual correctness, ambiguity, alternative entities and semantic types independently, blind to the new model results. Record actual minutes and disagreement. Keep the coordinator key separate from reviewer copies. This exercise estimates workload; it is not a factual audit of all 100 or all fresh items. See review/INSTRUCTIONS.md.

## Scientific implementation

See docs/IMPLEMENTATION_DECISIONS.md for exact estimands, nested splits, equal-cost query accounting, calibration assumptions and implementation choices made before development execution. The parent design is preserved in docs/PARENT_DESIGN.md. The development protocol is internally locked, with no mandatory external registration or fabricated registration receipt.

The primary analysis uses two text forecasters (TF-IDF and pinned frozen MPNet) and a common Ridge decoder with two report vectors. The optional fixed GradientBoosting decoder can be run later with --decoder gradient_boosting, without selecting the primary learner by results. Further main-study sensitivities and the fresh execution/final analysis runner remain to be implemented and frozen before fresh outcomes.

Local verification covers artificial-outcome tests and package integrity. The new GPU scorer, actual-token audit and encoder download have not executed locally. Passing the earlier synthetic hardware profiles does not substitute for executing this new runner. The notebook stops rather than changing precision, tokenizing differently or silently truncating inputs.
