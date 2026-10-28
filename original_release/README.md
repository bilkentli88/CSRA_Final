# Causal Source--Report Alignment (CSRA): anonymous review materials

This repository contains the anonymized data, frozen statistical specification, analysis code, diagnostic summaries, and expected outputs supporting the manuscript **“Do Verbal Source Attributions Track Causal Source Influence? A Prospectively Locked Audit Across Language Model Families.”**

The repository corresponds to the **prospectively locked CSRA v0.8 R4/A2 common-item study** reported in the manuscript. It supersedes earlier development and reserve-validation materials; those earlier experiments are not used for the confirmatory results reported here.

## Main result

The final common set contains 100 factual items evaluated on all three frozen model families. The prospectively primary incremental predictive estimate is

- `theta_SELF = -0.0006452668`
- 95% paired common-item bootstrap CI `[-0.0057623679, 0.0032968101]`
- frozen practical threshold `delta = 0.01`
- classification: **no practically meaningful positive SELF-specific increment**

The separately locked alignment analysis yields a positive common SELF-over-PEER interaction (`0.07644365`, bootstrap 95% CI `[0.05435158, 0.10058918]`) and significant model heterogeneity. Model-specific SELF-minus-PEER slope differences are positive in point estimate for Qwen, Granite, and Phi, with the Phi interval including zero.

These findings support the manuscript's central distinction: **measurable source--report alignment does not imply practically meaningful SELF-specific incremental predictive information beyond a matched PEER report.**

## Quick reproduction

Create a Python environment and run:

```bash
python -m pip install -r requirements.txt
python run_reproduction.py
```

The quick route reconstructs the frozen primary fold assignments and analysis-relation mapping from the released final common set, recomputes:

- the primary `theta_SELF` and companion `theta_PEER` point estimates;
- model-specific primary predictive metrics;
- all prespecified Ridge/fold sensitivity point estimates;
- model-specific SELF--PEER alignment estimates, the common interaction, `H`, and the heterogeneity likelihood-ratio test;
- alternate-wording robustness summaries;
- fictional-control summaries;

and verifies them against the sealed expected outputs.

A successful run ends with:

```text
PASS: all quick-reproduction checks match the sealed v0.8 expected results.
```

### Optional full bootstrap rerun

The manuscript confidence intervals use 2,000 relation-stratified paired common-item bootstrap replicates with full refitting. These can be rerun with:

```bash
python run_reproduction.py --bootstrap-primary
python run_reproduction.py --bootstrap-alignment
```

The alignment bootstrap repeatedly fits mixed-effects models and is substantially slower than the default quick reproduction. The sealed bootstrap summaries are included in `expected_results/`.

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── LICENSE
├── SHA256SUMS.txt
├── requirements.txt
├── run_reproduction.py
├── data/
│   ├── final_common_items.csv
│   ├── alignment_analysis_dataset.csv
│   ├── primary_item_effects.csv
│   └── supporting/
├── expected_results/
├── protocol/
├── code/
├── audit/
├── figures/
└── reproduced_results/
```

## Data included

`data/supporting/` contains the frozen answer-condition summaries and principal/alternate/fictional report summaries used to reconstruct the reported analyses. The much larger token-level/model-job checkpoints are not required for statistical reproduction and are not included in this anonymous review repository.

`data/final_common_items.csv` is the deterministic 100-item common set used for all three model families.

## Frozen analysis materials

- `protocol/csra_v08_analysis_spec.json`: prospectively frozen statistical specification.
- `protocol/csra_v08_environment_lock.json`: frozen software/runtime requirements.
- `protocol/csra_v08_model_tokenizer_lock.json`: exact model and tokenizer repository revisions.
- `protocol/postreview_analysis_lock.json`: post-review lock recorded before answer interventions.
- `code/csra_v08_analysis.py`: frozen analysis implementation.
- `audit/CSRA_v08_ANALYSIS_TECHNICAL_REPAIR_TR4_SPEC.json`: documented outcome-independent TR4 compatibility repair.

The identity-bearing protocol approval document and approval record are intentionally omitted during double-blind peer review. Their cryptographic hashes remain recorded in the frozen lineage materials and the complete public archive will be released after peer review.

## Technical-repair disclosure

Two outcome-independent technical repairs occurred during execution and are described in the manuscript. TR3 made the permitted OOM fallback sticky after repeated batch-size probing during Qwen report scoring; completed scores were not changed. TR4 corrected contrast-construction design rows to use an observed nuisance relation level rather than an absent hard-coded level; the frozen statistical formula was unchanged and numerical invariance of the interaction contrast across valid observed relation levels was verified before analysis.

## Figure

`figures/csra_framework.pdf` is the framework figure used in the current manuscript.

## License

Code is provided under the MIT License. The included data and analysis artifacts are supplied for scholarly verification of the submitted manuscript.
