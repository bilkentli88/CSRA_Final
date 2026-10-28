# CSRA Fresh40 qualification

This package takes the 40 approved facts into a new development qualification. It asks whether source reports add useful information beyond visible inputs, a training-selected external observer and a direct no-document probe. It does not establish publication readiness.

## Run in Colab

1. Extract this ZIP on your computer and open **CSRA_Fresh40_Colab.ipynb** in Colab.
2. Select an **A100 GPU with 40 GB** (the qualified scoring mode is unquantized FP32). Run the five code cells in order.
3. In Cell 1 upload the **same complete ZIP**, not the reviewed workbook or an earlier package. The notebook saves to `MyDrive/CSRA_Fresh40_runs/run1`.
4. Cell 3 audits tokens, encodes all possible visible texts on CPU, fits only old100 data, selects observers and freezes everything. CPU fitting can take time. No fresh model outcomes are collected here.
5. Cell 4 runs all targets and reporters automatically. Cell 5 analyzes and downloads **CSRA_Fresh40_results.zip**. Return that ZIP for assessment.

For a disconnect, reconnect the same notebook and run the cells again. The existing frozen package and committed batches are verified and reused. Do not run two Colab runtimes against the same Drive directory. The local writer lock does not coordinate separate runtimes. For an intentionally new run, change `run1` in Cell 1 before beginning. Never reuse or edit a directory to bypass a failed integrity check.

The seven scoring stages total 17,280 prompts (5,760 answers; 11,520 reports). The prior qualified forward-time reference is about 1.16 A100 hours, excluding model download/loading, CPU preparation, retries, interruptions and review. It is a planning reference, not a runtime or cost guarantee. Public Hugging Face downloads work without a token subject to service rate limits; optional Colab `HF_TOKEN` secret is read without printing its value.

## What is preserved

`inputs/reviewed_facts.xlsx` is the exact returned workbook. `review_audit.json` preserves all40 submitted approvals, dates, reviewer name and comments. The workbook's formula caches were empty; decisions were recomputed from its source and review cells. No question or answer edits were found. The entered reviewer identity and independent review were not separately verified. Internal subject URNs are not asserted Wikidata identifiers. Prior AI evidence statuses remain distinct from the submitted approvals.

The immutable old100 package and aggregated900 cases are included for training. Its own hashes and historical100-only checks remain intact; this is a separate fresh-data runner. Old100 factual re-review remains unverified. No old pickled predictors or old text embeddings are reused.

## Reading results

`analysis/READ_RESULTS.md` summarizes the GB primary contrasts. JSON contains each target and the Ridge sensitivity; CSV contains every prediction and squared error. Positive normalized MSE reduction favors self in the three primary comparisons. The proposed0.01 screen and two-positive-target check are engineering criteria, not a validated scientific minimum or an automatic publication judgment. Bootstrap intervals are descriptive development summaries.

Both candidate-aware text forecasters are recomputed before fresh outcomes. All decoder and strongest-observer selection decisions use only old100 data. Visible inputs do not expose factual reference identity; public probe features are invariant to swapping the P and I designations. Reference-aware features appear only in a separately labeled ceiling. Self reports mean same model weights/configuration, not retained answer-generation activations.

After inspection, Fresh40 must remain development data. A later confirmatory sample and sample size will need a separate frozen protocol. Mistral remains a reporter, never a target in this run.

## Technical validation

Run `python tests/test_workflow.py` in the pinned analysis environment. Tests use explicitly synthetic embeddings and scoring fixtures, exercise actual training, scoring aggregation, frozen prediction, bootstrap and export, and do not claim live GPU validation. Real scoring refuses synthetic freezes. See `LOCAL_VALIDATION.json` for the checks actually run for this delivery. Live model/token downloads and GPU scoring must be verified by the notebook's mandatory pre-outcome and startup audits.

The source, protocol and inputs are SHA256 locked. These are local reproducibility checks, not an externally registered protocol. No scientific fresh outcomes are distributed with the package.
