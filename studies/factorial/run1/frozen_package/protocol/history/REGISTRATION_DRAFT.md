# Registration draft: source-report wording and answer visibility

**Status:** Draft for author review, not submitted, registered or timestamped. Complete registry metadata and externally deposit the exact source package and package-lock SHA-256 before any new pilot report scoring. Technical synthetic preflight is allowed first. This registration covers only the pilot.

**Title:** Ownership wording, hedging, and answer visibility in forced-choice language-model source reports: an existing-item factorial pilot.

## Prior information and population

Study 1 used the same selected 100 items and three quantized systems; its answer/report outcomes and corrected analyses are known. This pilot reuses the fixed answer summaries in inputs/answer_targets.csv, but generates all eight matched report cells anew. It cannot be described as an independent replication on fresh items. No new pilot model reports have been collected during preparation of this draft. The submitting author must verify that statement at registration and disclose any intervening collection; do not attest if it is false.

The sample is all 100 supplied items, without outcome-driven replacement. There are 57 normalized ordered P/C/I answer triples. Fixed systems are Qwen3-4B-Instruct-2507, Granite-3.3-2B-Instruct, and Phi-4-mini-instruct, with exact model/tokenizer revisions and precision in model_lock.json. These are three development systems, not a random sample of families. The target is the frozen strength-matched C-document versus P-document difference in C-versus-P answer log support (document-substitution effect). It is not a hidden-state measure.

## Design and measurement

Fully crossed ownership OWN/OTHER x certain/hedged wording x recorded answer shown/hidden, for each item, system and three strengths (C_SKEPTICAL, C_PROVISIONAL, C_BARE). All report requests are independent fresh conversations. OWN says “You previously answered”; OTHER says “Another instance of the same model answered.” Only the shown cells contain the Recorded answer block. Hiding a recorded choice does not remove answer clues inherent in the question/document. Exact text, role definitions and four cyclic A-D mappings are in prompts.json.

100 x 3 systems x 3 strengths x 8 cells x 4 mappings = 28,800 logical report-scoring jobs. Average role-mapped log supports across mappings before four-role softmax. q_both is the resulting both-source probability; three additive log ratios relative to prior define the primary predictive representation. Frozen A-D token IDs, NF4 double quantization/FP32 scoring, software and recovery are specified in config.json and PILOT_IMPLEMENTATION_ADDENDUM.md.

## Four primary estimands

P1: q_both(OWN) - q_both(OTHER).

P2: q_both(hedged) - q_both(certain).

P3: q_both(shown) - q_both(hidden).

Each marginalizes equally over the other factors, strengths, the complete-case items and three fixed systems. No directional prediction is required.

P4: mean over systems and hedging levels of theta_OWN(shown) - theta_OWN(hidden). theta_OWN = held-out R2(X3) - R2(X2), where X1 is baseline no-document margin + strength + relation; X2 adds three OTHER log ratios; X3 adds three OWN log ratios. The primary baseline omits the frozen answer role. Ridge alpha=1 with free intercept; numeric preprocessing fits only on training folds. Target normalization is the original complete-case target population variance per system, held fixed within bootstrap. The five-fold map holds whole normalized ordered answer triples together and is frozen in folds.json using seed 20261016, before report collection.

## Inference and decision rule

10,000 relation-stratified paired bootstrap draws of whole answer triples; all strengths, systems and factor cells travel together. Seed 20261016. Prediction and training-fold preprocessing are refitted for every draw. Fixed test-fold zero-weight draws are deterministically redrawn and counted, as specified in the addendum. Four two-sided percentile intervals use tails 0.00625/0.99375 (98.75% each). Declare a primary contrast distinguishable from zero only when its adjusted interval excludes zero. Nominal simultaneous coverage depends on adequate bootstrap behavior. Ordinary 95% and all secondary intervals are descriptive. Nonsignificance is not equivalence; no post-outcome utility margin or new primary subgroup is introduced.

## Secondary analyses

Frozen-answer-informed Ridge; document-minus-prior VSA instead of three log ratios; model-specific category effects and predictive fits; full four-category probability and modal distributions; answer/modal association within model, strength, relation and cell; two-way/three-way q_both interactions; signed and absolute differences between hedging and ownership effects. Uncorrected Cramer's V and undefined-table rules are in the addendum. GradientBoostingRegressor uses 100 trees, learning rate .05, depth 2, squared-error loss, subsample 1, random_state 20261016, same folds, no tuning; 2,000 paired draws and duplicated rows for resampled refits. Ridge uses 10,000. Secondary completion status must be disclosed. Historical R2 thresholds .01 with .005/.02 sensitivity are reference values, not validated utility or equivalence thresholds.

## Failure handling and stopping

Registration and all-three-model synthetic preflight precede new pilot report outcomes. Batch 8 falls back to 4, 2 and singleton isolation on OOM. Commit terminal singleton OOM and never overwrite completed/terminal rows. Other errors stop execution. Do not inspect interim hypothesis analyses, stop on significance, select the best report wording or change the primary learner. Analyze only once all logical jobs have committed statuses. An item with any terminal job is excluded jointly across every cell and system, without replacement; document all exclusions. More than 5% attrition flags incomplete evaluation. Fewer than 20 complete items stops this analysis pipeline. Checkpoint/lock guards and their limits are in the addendum.

## Purpose and subsequent study

This pilot establishes technical feasibility and precision on already used items. It does not justify broad introspection, full-precision, or family-general claims. Any fresh study needs a separate factual-review record, eligible candidate pool, resource decision, final sample size, fold map and independent registration before fresh outcomes. Existing/fresh items are never pooled as an untouched confirmatory sample. Proceed or narrow scope on feasibility, annotation quality and attainable precision, regardless of pilot effect direction or significance. The earlier 600-item idealized planning candidate is not an empirically established power guarantee.

## Registry metadata the author must supply

Author names, affiliations, contributions and AI assistance must be truthful; no human review, provenance or historical registration is inferred from these files. Add registry-required metadata, the exact protocol/source attachments, and the SHA-256 of protocol/package_lock.json. After registry submission, enter its real HTTPS URL, timezone-aware UTC timestamp and exact hash in a copy of registration_template.json. The author attestation must be true. Keep the registry export with results. These fields are factual author actions, not approval of a hypothetical future study.
