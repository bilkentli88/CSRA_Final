# Reproduction guide

Run commands from the repository root. The default supported command is `python reproduce.py`. It writes generated files under ignored `reproduced_results/`, leaving the supplied experimental inputs, locks and expected outputs unchanged.

## CPU environment

Use Python 3.12 in a separate virtual environment and install `requirements.txt`. The tested pins are NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3 and Matplotlib 3.10.8. The statistical-only dependency list is `requirements-analysis.txt`; `requirements.txt` also installs figure dependencies. No network is needed after dependency installation for the CPU routes.

```bash
python -m pip install -r requirements.txt
python reproduce.py paper
```

The default route performs these steps:

1. Checks the root inventory, original 46-file source snapshot, 237 historical source identities, package locks and manuscript/scoring bindings.
2. Verifies prompt identities and reconstructs factorial, development and Fresh40 aggregates from all 89,280 released jobs.
3. Checks saved prediction losses, initial Fresh40 intervals and R², primary factorial intervals and six orthonormal full-role point estimates (98 numerical checks).
4. Repeats the October 27 analysis: 48 pooled sensitivity contrasts, 144 target-specific intervals, 144 omitted-target intervals, error strata and factorial role/answer-association readouts. This regenerates the full group-bootstrap calculations from the released data with their original seeds.
5. Rebuilds all 22 current main and supplementary tables. Reference captions/layout come from `paper/reference/tables/`; numerical bodies come from identified sources and must match all displayed V10 numeric entries. No estimate is substituted from the reference LaTeX. Some original-study bootstrap/refit results and the factorial primary bootstrap draws are supplied numerical inputs rather than recomputed fits in this fast route.
6. Regenerates all six figures. Later figures read full-precision estimates rather than their earlier rounded plotting constants. Output bytes can differ through plotting precision, PDF metadata or fonts; the estimands and underlying records are unchanged.

The final report is `reproduced_results/paper/validation.json`. `table_coverage.json` identifies source paths and the use of archived refits for each table. `table_numeric_comparison.csv` lists every checked displayed numeric entry. An assertion or nonzero exit is a failed reproduction; a partially populated output directory is not a successful run.

## Frozen predictor replay

```bash
python reproduce.py fresh40
```

Makes an isolated copy and loads the original fitted artifacts, using saved embeddings and reconstructed score aggregates. Compares 5,040 policy predictions (720 algorithm/case rows × seven policies) and their errors/intervals with the saved CSV. No training, encoder download or model inference. Joblib artifacts depend on the pinned package versions.

## Outcome/agreement robustness refits

```bash
python reproduce.py posthoc
```

Rebuilds log-support and bounded outcomes from raw answer scores, reuses saved encoder representations, and refits text forecasts and all policy decoders on the original 100 development items. Fresh40 outcomes are excluded from fitting/tuning/observer selection. It compares 2,880 prediction rows and 48 contrast/interval rows against the archived results with absolute numerical tolerance 1e-8. Outputs are retained in a new `reproduced_results/posthoc_refit_*` directory. The script needs a temporary directory outside the input tree, which the wrapper supplies.

This retrospective refit does not create another untouched test set or prospective freeze.

## Full factorial refit

```bash
python reproduce.py factorial
```

Rebuilds the report aggregates and repeats the primary 10,000-draw grouped bootstrap, including Ridge refitting, with frozen `analyze_pilot.py`. The four primary contrasts use 98.75% marginal intervals. Original system-specific category intervals are descriptive 95% intervals. The unfinished nonlinear bootstrap is not invoked or claimed as a completed result.

## Development refits

```bash
python reproduce.py development
```

Copies the run to a new output directory, removes only copied analysis caches and refits Ridge and gradient-boosting nested group-held-out pipelines from saved embeddings/scores. Returned Ridge evidence and the original secondary gradient-boosting refit are not overwritten. This route repeats supervised fitting; it does not rerun MPNet embedding inference.

## Historical environment and full original-study analyses

Use a **second environment** with `original_release/requirements.txt`. Do not replace the later-study packages in the same environment. Historical pins are NumPy 2.0.2, pandas 2.2.2, SciPy 1.16.3, scikit-learn 1.6.1, statsmodels 0.14.6 and the stated patsy range. The original runtime lock is retained under `original_release/`.

```bash
python -m pip install -r original_release/requirements.txt
python reproduce.py original
python reproduce.py original-full
python reproduce.py original-reanalysis
```

`original` performs the submitted point checks. `original-full` also repeats 2,000-draw predictive and mixed-model alignment bootstraps in an isolated copy and checks the sealed expected results. `original-reanalysis` repeats the audit and the 2,000-draw revised alignment, scalar/full-role, alternative-target and descriptive analyses. These are slower fitting routes, distinct from regenerating tables from their saved outputs. Full statistical refits can show platform-dependent numerical differences; retain the software environment and inspect any reported tolerance failure.

No route can recreate missing original candidate/screening snapshots or individual NF4 inference checkpoints.

## New language-model scoring

See [docs/GPU_RERUN.md](docs/GPU_RERUN.md). These are separate experimental reruns requiring model downloads and compatible CUDA hardware. Preserve supplied scores and local freeze records, use new output directories and retain the new environment/score manifests. New scores must not replace the published evidence to conceal discrepancies.

## Integrity and publication

`python run_reproduction.py integrity` uses only the standard library. The root SHA256SUMS.txt covers the distributed files (excluding itself). `.gitattributes` disables newline conversion to preserve frozen bytes across checkouts. Generated results, virtual environments and bytecode are excluded from the release inventory and Git.

The existing `run_reproduction.py` is the low-level compatibility entry point. Its historical routes remain available; use `reproduce.py paper` for the current manuscript. No command rewrites the article, creates a remote release, establishes external registration or uploads GitHub content.
