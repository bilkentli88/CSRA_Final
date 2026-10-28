# CSRA post hoc robustness checks

These checks supplement the user's CSRA Revised Repository R4 and the 33-page manuscript *Source Reports Under Factual Conflict: Elicitation Effects and Predictive Value*. Read `REPORT.md` first.

The script reads an extracted R4 repository and writes all new analysis files to this package's `results/` directory. It preserves the original study code, scores, locks and reported primary results. No language-model inference, model downloads or GPU are required. Python 3.12 is recommended; exact analysis dependencies are listed in `requirements.txt`.

## Reproduce

Extract the original R4 repository and this package, then run:

```sh
python -m pip install -r requirements.txt
python -B run_checks.py --repo /absolute/path/to/CSRA_Revised_Repository
```

For a separate results directory, add `--out /absolute/path/outside/the/R4/repository`.

The original R4 package is a required input; this add-on does not duplicate its data or model files. The R4 source archive is `CSRA_Revised_Repository_R4(1).zip`. A run verifies its 399 inventoried files and study freeze, reconstructs the new bounded target from raw answer scores, recomputes nested text forecasts and decoders from the old100 data, and checks original predictions against saved results.

## What is evaluated

- Both the original log-support outcome and a bounded answer-probability contrast.
- Both the original feature policies and policies with an explicit, visible answer-document agreement indicator appended to every policy.
- The original gradient-boosting decoder and Ridge sensitivity, with unchanged hyperparameters.
- Six policies: visible, source report, selected external report, direct probe, probe plus source, and probe plus observer.
- Six contrasts under each outcome/feature/decoder combination, including the matched-cost joint-policy comparison.

Every supervised fit uses the original 100 development items only. Nested grouped text forecasts are recomputed separately for each outcome. Existing MPNet encodings are reused. Original observer choices are retained for all variants rather than reselected. The new agreement feature requires no new model calls and does not identify which non-document alternative is factually correct.

These are **post hoc sensitivity analyses after the Fresh40 outcomes were inspected**. They are not new untouched validation data and do not replace the original frozen analyses. The descriptive 95% intervals resample forty whole Fresh40 groups with the original seed and condition on the fitted development pipelines; they exclude training-sample and new-target uncertainty and have no multiplicity adjustment.

## Files

| File | Contents |
|---|---|
| `analysis_plan.json` | Full explicit analysis specification written before running these checks; its post hoc status is stated. |
| `run_checks.py` | Reproducible CPU analysis. |
| `REPORT.md` | Results, interpretation and suggested manuscript paragraph. |
| `results/contrasts.csv` | All 48 contrasts, intervals, per-target gains and leave-one-group-out point ranges. |
| `results/absolute_performance.csv` | MSE and fresh-sample R2 for every policy and target. |
| `results/per_case_predictions.csv` | All 2,880 prediction rows across outcomes, variants and decoders. |
| `results/outcome_diagnostics.csv` | Outcome variance/range and agreement prevalence by split and target. |
| `results/*_text_forecasts.npz` | Nested development and fresh text forecasts. |
| `results/verification.json` | Checks against 4,320 original policy predictions and the original source-added-to-probe intervals. |
| `results/run_manifest.json` | Environment, input identities and analysis specification. |

Normalized gains use the development variance of the relevant outcome for each target. Values from the log-support and bounded outcomes therefore have different denominators. They are not percentage improvements and should not be treated as the same estimand.
