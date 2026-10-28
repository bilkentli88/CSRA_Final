# Provenance and interpretation

## Preserved evidence

`provenance/SOURCES.json` identifies the uploaded source archives by filename, byte count and SHA256. `PRESERVED_SOURCE_FILES.json` records every included file from the original submitted repository and the three returned study archives. The original 46 files are additionally listed in `ORIGINAL_RELEASE_FILES.json`. Python caches are excluded; frozen source files, inputs, locks, raw score ledgers and returned scientific outputs are unchanged.

The combined repository date/version identifies packaging, not the date on which the old experiments were prospectively registered. The root `SHA256SUMS.txt` inventories this release and is not itself a historical protocol lock.

## Original study

The preserved original repository contains statistical summaries for 100 selected items on three NF4-scored targets. Exact recovered prompt templates, ten fictional-control definitions, the four review CSV versions, recovered selection records and deterministic fold assignments are in `original_reanalysis/audit/` and `manuscript_support/prompts/` and `paper/prompts/`.

Two original analysis-lock documents exist: the anonymous metadata record retained under `original_release/protocol/` and the canonical execution record recovered under `original_reanalysis/audit/recovered_provenance/`. They have different lengths/hashes and serve different roles. `results/lock_lineage_verification.json` in that audit directory explains their linkage. Neither file is overwritten or presented as identical to the other.

The 240-row review and supplied selection algorithm reproduce the released 100-item set. This verifies the supplied records' consistency; it does not independently authenticate reviewer identity, human/AI task allocation, factual correctness of every decision, or the original external timing. The directory name `independent_audit` in historical records refers to numerical checking, not an independent human factual review.

Material original collection records not contained in this combined release include the official final 600-candidate source JSONL/source snapshot, complete screening outputs/activation manifest, and individual original NF4 scoring checkpoints. The recovered preactivation and execution documents are excerpts supplied through V1 rather than newly retrieved complete original collection archives. Archive-level verification scripts may therefore require the separate original ZIPs named in their historical READMEs. Their saved reports are included as historical audit outputs, not as checks rerun by the root quick command.

## Later experiments

The factorial experiment uses 100 previously studied items, with 57 ordered answer-triple clusters and 28,800 report jobs. Its internal exploratory freeze explicitly disclaims external registration. Draft flags inside frozen prompt/protocol files are historical preparation metadata. Completed job ledgers and `primary_analysis/analysis_status.json` establish what actually executed. Secondary nonlinear bootstrap intervals were not completed.

FP32 development uses the old100/900-case data and 57 groups. The returned run contains Ridge analysis. The secondary gradient-boosting output included here was generated in an earlier CPU refit from the same frozen code/inputs; it is not a second GPU collection or a fresh test. Additional diagnostic results discussed in an earlier planning memo but unused by R3 are not promoted into primary results here.

Fresh40 contains 40 purposively selected, relation-balanced items and 360 cases on the same three known FP32 targets. The author checked factual answers using online sources and returned the completed workbook. This is author-conducted factual verification, not an independent reviewer agreement assessment or a test of unaided recall. Source links and the exact workbook are preserved. On 21 October 2026 the sole author clarified that Alex Miller, the alias entered in the Fresh40 review records, denotes the author. The recorded approvals therefore do not represent an additional independent reviewer. The original identity-verification flags describe the archived checks; this later author statement is documented separately in [FRESH40_REVIEW_CLARIFICATION.md](FRESH40_REVIEW_CLARIFICATION.md) and `provenance/FRESH40_REVIEW_CLARIFICATION.json`. This clarification applies to Fresh40 and does not resolve the separate historical study’s human/AI review allocation.

All Fresh40 predictors and observer selections were fitted on old100 data before its locally recorded fresh scoring. The frozen run includes the predictors, embeddings, training manifest and local pre-outcome freeze. Hash and local commit checks establish internal consistency; no external registration service or trusted timestamp is thereby established. The fresh outcomes have now been inspected and are development evidence. Mistral supplied reports only, with no Mistral target-transfer result.

## Noninterchangeable estimands

- Original scalar VSA: document log support minus prior log support.
- V1 complete-role predictor: three orthonormal log-score contrasts; Ridge regularization is defined in that coordinate system.
- Later complete-role predictor: three log-score ratios against prior (ALR), not a substitution for V1's estimates.
- Original with/without-answer comparison: omits the answer from predictors; original reporting prompts still show it.
- Factorial visibility: changes whether the reporting prompt shows the recorded answer.
- Development common decoder: compares a source-corresponding pair with two external reports under the common replacement decoder.
- Fresh40 policies: separately calibrated predictors under explicitly defined access to visible inputs, reports and probes.

Fresh40 normalized MSE contrasts retain old100 target variances as fixed denominators. Its 95% intervals are descriptive group-bootstrap intervals conditional on the fitted old100 pipelines. They do not include training-set uncertainty. Query counts exclude common outcome collection and do not equate model-specific GPU cost.

## Current V10 / R10 binding

R10 matches the V10 Neurocomputing working manuscript identified in `paper/manuscript_binding.json`. Reference tables and figures are in `paper/reference/`. The current executable paper renderer regenerates every table body from numerical/design records and compares displayed entries with V10. It retains the exact historical 46-file repository and the three returned experimental archives; root documentation and new rendering code are packaging additions.

The October 26 agreement/bounded-outcome fits and October 27 heterogeneity/factorial readouts remain explicitly post hoc. They do not create fresh confirmation. The alternative-query control and expanded prospective evaluation have not executed. The default paper command uses some archived statistical refits/primary bootstrap draws as numerical inputs; separate commands repeat those fits. Exact scope and tests completed for this package are recorded in `verification/`.

Redundant old Overleaf-side helper copies and obsolete V3 presentation snapshots were removed from the active package. Dated analysis packages, raw scores, fitted predictors and original experimental source identities are preserved. The local package does not update the public repository or assert external registration.
