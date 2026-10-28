# CSRA: source reports under factual conflict

Reproducibility repository for **Source Reports Under Factual Conflict: Do Answering Models Have a Predictive Advantage?**, by **Aykut T. Altay**. This package is matched to the **V10 Neurocomputing working manuscript**. The manuscript is not represented as accepted or published.

## Reproduce the article

Use **Python 3.12**. From the extracted repository directory:

```bash
python -m venv .venv
# Activate .venv using the instructions below.
python -m pip install -r requirements.txt
python reproduce.py
```

Activate with `source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate.bat` in Windows Command Prompt, or `.venv\Scripts\Activate.ps1` in PowerShell. The default command is `python reproduce.py paper`.

This CPU route needs **no GPU or model download**. It verifies 89,280 released later-study scoring jobs, recomputes score aggregates, runs 98 earlier numerical checks, recomputes the added grouped analyses and rebuilds **all 22 main and supplementary tables and six figures**. Every table's displayed numerical entries must match V10 or the command fails. It writes to:

```text
reproduced_results/paper/
  validation.json                 final success/failure check
  table_coverage.json             one source/status record per article table
  table_numeric_comparison.csv    article and regenerated numeric entries
  tables/                        regenerated LaTeX and formatted CSV rows
  figures/                       regenerated vector figures
  revision/                      full-precision CSV results and diagnostics
  absolute_performance.csv       recomputed target/policy MSE and fresh R²
```

The default route uses saved original-study statistical refits and the saved factorial primary bootstrap draws where listed in the coverage report. It regenerates presentation and checks the underlying records; it does **not** rerun every historical fit or collect new language-model scores. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the separate refitting routes and their environments. PDF timestamps, font embedding and minor plotting precision can differ; article numerical table entries are checked at their displayed precision.

## Additional reproduction routes

| Command | What it repeats |
|---|---|
| `python reproduce.py verify` | Integrity, raw-score aggregation, earlier numerical checks and added grouped analyses |
| `python reproduce.py fresh40` | Replays the frozen fitted predictors and compares 5,040 policy predictions |
| `python reproduce.py posthoc` | Refits the October 26 outcome/baseline sensitivities on old100 only; compares all 2,880 saved prediction rows and 48 contrasts |
| `python reproduce.py factorial` | Repeats the factorial 10,000-draw primary grouped bootstrap, including Ridge fits |
| `python reproduce.py development` | Repeats the development nested group pipelines for both decoders |
| `python reproduce.py original` | Original submitted statistical point checks; **separate historical environment** |
| `python reproduce.py original-full` | Original primary and mixed-model alignment bootstraps; **separate historical environment** |
| `python reproduce.py original-reanalysis` | Later original-study scalar/full-role and alignment reanalyses; **separate historical environment** |

The fitted-predictor and post hoc commands use the pinned later-study environment installed by requirements.txt. The historical environment is described in the reproduction guide. Full refits can take substantially longer than the default route.

## Files

| Path | Purpose |
|---|---|
| `reproduce.py` | Supported user entry point |
| `scripts/` | Integrity, raw-score replay, predictor replay and current paper rendering |
| `paper/manuscript_binding.json` | Exact V10 main/supplementary source/PDF identities and table order |
| `paper/reference/` | V10 reference tables/figures, used to check outputs |
| `paper/prompts/` | Exact prompt text displayed in the V10 supplement |
| `original_release/` | Original submitted 46-file statistical repository, unchanged |
| `original_reanalysis/` | Original-study audit, recovered provenance, refit code and results |
| `studies/factorial/` | 28,800 report jobs, frozen implementation and analysis |
| `studies/development/` | 43,200 answer/report jobs, embeddings and group-held-out fits |
| `studies/fresh40/` | 17,280 jobs, fitted predictors, embeddings and fresh predictions |
| `new_analysis/posthoc_20261026/` | Outcome/agreement sensitivities and development-only refits |
| `new_analysis/revision_20261027/` | Target intervals, omitted-target diagnostics and expanded factorial readouts |
| `manuscript_support/checks/` | Earlier numerical checks and their evidence inputs |
| `verification/` | Checks actually completed while preparing this release |
| `docs/` | Output map, data dictionary, provenance, GPU rerun and GitHub upload guidance |

Frozen execution packages retain their historical READMEs and plans. The root documentation describes this complete package. Some archived plans describe unexecuted larger studies; only the completed runs listed above support V10.

## Scope

The repository supports reproduction from the available records. The original 600-candidate snapshot, complete screening outputs and individual original NF4 scoring checkpoints are unavailable. Original-study analyses therefore start from the released score summaries. Later-study raw scores are included. [GPU rerun guidance](docs/GPU_RERUN.md) explains how to collect a separate new run using the frozen code; no claim of identical floating-point model scores on different hardware is made.

Fresh40 contains forty purposively selected, author-verified facts. The archived alias **Alex Miller** denotes the sole author, not an independent reviewer. Local locks are content/integrity records, not externally timestamped preregistration. The October 26/27 additions are post hoc. No alternative-query control or expanded prospective evaluation is included. See [provenance and scope](docs/PROVENANCE_AND_SCOPE.md).

Code is distributed under the accompanying MIT license, subject to nested notices and [licensing scope](LICENSE_NOTES.md). Model weights are not distributed. Citation metadata is in CITATION.cff. This prepared package does not itself update GitHub or create a DOI; [upload instructions](docs/GITHUB_UPLOAD.md) describe the next step.

## Article presentation

V10 separates the Neurocomputing main article (`review,12pt`) from a standalone supplement (`preprint,12pt`). The main article contains Tables 1–10, Figures 1–5 and Eqs. (1)–(13). The supplement contains Sections S1–S10, Tables S1–S12, Figure S1 and Eq. (S1). All numerical table bodies, displayed equation contents, scientific claims and figure assets are preserved.

`paper/manuscript_binding.json` records exact source/PDF identities for both documents and each table's displayed number and location. The integer `number` field preserves the 1–22 rendering order; `display_label` and `document_number` identify the publication numbering. Cross-document references are validated against the compiled labels in the Overleaf project.

The scientific records are byte-identical to R9. The current CPU paper route is rerun for this release. Prior detailed predictor replay and refit checks remain in provenance/prior_releases. The KAIS V8 manuscript is retained separately as a possible later submission.
