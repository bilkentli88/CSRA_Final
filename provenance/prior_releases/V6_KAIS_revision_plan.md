# CSRA revision plan

27 October 2026. Working manuscript: **V6**, **Source Reports Under Factual Conflict: Do Answering Models Have a Predictive Advantage?** Current target: **Knowledge and Information Systems (KAIS)**, using its subscription publication route. The focused V6 revision restores the original observer question while retaining V5's evidence and qualifications. Computational Linguistics remains a possible later target after scientific strengthening; no confidence-bound threshold establishes journal suitability.

The current KAIS editorial revision is complete. The experimental extensions below remain a research plan, not completed findings or a claim that every extension is formally required by KAIS. Submission timing should be decided explicitly after reviewing the remaining control gap.

## Order of work

| Step | Work | Completion criterion | Current status |
|---|---|---|---|
| 1 | Establish current manuscript and repository inputs; record limitations and hashes | Current repository resolved, scientific inputs preserved, analysis additions separated from historical freezes | Complete |
| 2 | Quantify Fresh40 heterogeneity and Phi's contribution | Reproduce V4 estimates; add target-specific intervals, leave-one-target-out summaries, absolute errors, group influence and descriptive strata | Complete; exploratory saved-prediction analysis |
| 3 | Expand factorial readouts beyond the both-source category | Per-system uncertainty, document-minus-prior support, full role distributions and report–answer association by visibility; reproduce original primary results | Complete; exploratory saved-score analysis |
| 4 | Integrate completed analyses into the manuscript and reproduction commands | Consistent abstract/results/discussion; readable tables; complete compilation; commands regenerate the new results | Complete in V5/R5; focused KAIS narrative and format completed in V6 |
| 5 | Specify and qualify an alternative-query control on existing data | Exact same-reporter/input/budget comparison; development-only training; frozen question and features; tokenizer and software checks; pilot outputs | Next; requires new GPU scores before empirical completion |
| 6 | Design and externally register one new evaluation | Verified sampling frame; new item review; chosen targets/relations; simulation-based sample size; complete frozen predictors; actual external registration before evaluation scoring | Pending Step 5 and item/model feasibility work |
| 7 | Run the registered evaluation and assess claims | Complete fixed-size collection; registered primary analysis plus labelled sensitivities; all outcomes retained regardless of direction | Pending new data |
| 8 | Final manuscript, research release and submission decision | Claim/evidence audit, reference verification, final journal and fee checks, matching research package or authentic versioned archive identifier | V6 editorial revision complete; control/timing and research release remain pending |

The work proceeds through these steps in order. Steps 2–4 are now supported by saved-data analyses; later experimental steps must not be represented as completed. The V6 manuscript includes only completed research. No new model inference was performed in this revision batch.

## Step 1 — Evidence and provenance

Use the current `CSRA_Revised_Repository.zip`, not an assumed public GitHub state. Its SHA-256 is `106ebd86e1c2094b8042c7a20bf0a1bb924303f6863cd83e7739e80ac4531f9c`. It contains the October 26 robustness package, although the older root reproduction guide did not yet cover it. The saved-score analysis baseline was the complete V4 Overleaf project. V5 integrated those analyses; V6 revises its narrative and KAIS format without changing the scientific results.

Preserve all historical experimental inputs, raw scoring records, freezes and original analyses. The original 600-candidate snapshot, complete screening outputs and individual original NF4 checkpoints remain unavailable. Existing disclosure resolves the reporting issue; it cannot restore those records. Fresh40's reviewer alias denotes the sole author. Do not describe this as independent verification. Internal hashes identify contents; they do not supply a trusted external timestamp.

The country-of-origin records with Africa as a country alternative were already excluded. The V4 abstract was already 207 words. The two Sun 2026 citations are different papers. These are checks, not reasons to repeat experiments.

## Step 2 — Fresh40 heterogeneity

Use the October 26 saved predictions for all four outcome/feature specifications and both decoders. For all six existing policy contrasts:

- Reproduce the equal-target estimates and 10,000-draw whole-fact bootstrap intervals.
- Calculate target-specific intervals with the same paired group resamples.
- Omit each target in turn and report the equal mean for the two remaining fixed targets. This does not provide inference to a population of models.
- Retain absolute MSE, MAE and fresh R-squared alongside normalized gains.
- Inspect gains by relation, strength and observed answer–document agreement, and inspect single-group influence. These are descriptive diagnostics, not causal mechanisms or separately confirmed subgroup effects.

All new intervals are descriptive 95% intervals, unadjusted for the enlarged collection of comparisons and conditional on fitted development predictors. Keep the existing development-only training and original outcome-specific normalizations. Do not compare gains across outcomes as percentage reductions in one common quantity.

Completed result: under gradient boosting with agreement and log support, Phi's gain is 0.1203 [0.0260, 0.2351], while Granite and Qwen have intervals spanning zero. Omitting Phi leaves 0.0100 [-0.0206, 0.0393]. With bounded probability and agreement, every individual-target interval spans zero. These findings qualify the average gain without establishing absence of an effect.

## Step 3 — Factorial readouts

Retain the four original primary contrasts and their original 98.75% marginal intervals. Add the following exploratory readouts, using the original 57-group, relation-stratified bootstrap and paired prompt cells:

1. Per-system both-source support contrasts and descriptive intervals.
2. The support balance `q_document - q_prior`, using the existing normalization over all four roles, and the document/prior log-support ratio as a second representation.
3. All four category supports by answer visibility, so a shift away from document support is not automatically attributed to prior support.
4. The association between support balance and the recorded answer being C versus P, separately for shown and hidden answers. Its shown-minus-hidden contrast summarizes how the association changes with visibility. I answers remain in the complete role summaries and are explicitly excluded from this particular C-versus-P association.

The answer role is observed rather than randomized. Association is not accuracy, causal attribution or proof of reconstruction. New contrasts use descriptive unadjusted intervals. The manuscript reports only the completed Ridge predictive analysis; the unfinished nonlinear bootstrap remains documented as history, without a nonlinear inferential claim.

Completed result: Qwen's both-source visibility shift is near its category floor, but its document-minus-prior support balance changes by -0.2848 [-0.3960, -0.1848]. Its recorded-answer association is greater when the answer is shown. Thus the both-source outcome misses a substantial reporting pattern.

## Step 4 — Manuscript and reproducibility

Place target-specific uncertainty beside the main prediction results and the expanded factorial readouts beside the factorial results. Keep the initial frozen specification separate from both rounds of post hoc analyses. Explain Phi's heterogeneous contributions without claiming an internal mechanism. Clarify the alternative-question comparison as a remaining gap. Consolidate repeated qualifications, retaining the qualifications needed to interpret each result.

Keep operational definitions precise. Prefer “document-substitution effect” and “document/prior report contrast” in explanatory prose while preserving historical names and equation labels needed to reproduce earlier results. Do not rename all archived code, quantities and records just to simplify terminology.

Root reproduction commands must cover both the October 26 robustness refits and October 27 saved-score analyses. Update the manuscript/output map, software guidance and release checksums. Supply a complete Overleaf project, not a patch to apply manually. Do not claim that the public GitHub repository has been updated by creating a local ZIP.

## Step 5 — Alternative-query control

Use the same reporting model, document, question, recorded answer, ownership cues, four option rotations and call count as the source-report policy. Ask a fixed alternative question about document plausibility with four fixed response options. Call it an alternative-query control: its predictive value need not be zero.

Collect its scores on the 100 development items and the already inspected Fresh40 items. Treat this as development/pilot work, not confirmation. Construct three relative log-score features with a fixed reference category. Fit the control-augmented policy using development labels only and the same text forecasts, explicit agreement, probe features, decoder and target-specific calibration as the source-augmented policy.

Compare probe-plus-source directly with probe-plus-control, rather than comparing whether each separately improves on probe-only. Match query counts and disclose differences in prompt length and runtime. Fix the question/options before seeing these new control scores; record the design and every subsequent deviation. Use fictional examples for token and batching checks, and retain all raw scores and prompt identities. Also preserve the existing selected-observer comparisons: reporter identity and question content address different questions.

## Step 6 — New evaluation protocol

Do not yet fix 110, 200 or 300 as a guaranteed adequate number. The control pilot is needed to characterize uncertainty for the source-versus-control comparison. Simulate precision and decision probabilities across plausible effects, including zero, with paired fact groups and a fixed sample size. Include Monte Carlo error. Do not choose the sample size or stop collection to obtain a positive result. Keep 0.01 as a historical engineering benchmark unless an independent substantive justification supports a future decision threshold.

Define the sampling frame before drawing items. Random sampling within a documented eligible pool supports inference to that pool, not all factual questions. Separate item eligibility and factual verification from model performance. Use type-valid alternatives, identity/alias checks, complete unordered-triple checks and explicit rules for ambiguous or changing facts. Have an additional human verifier check new references if one is available; record their actual contribution. Do not substitute an author alias or an AI check for an independent human review.

Assess the feasible expansion in answering-model families/sizes and factual relations. Additional targets require development calibration and outcome collection for those targets. New relations require either development examples or an explicitly specified unseen-relation transfer evaluation. Hardware feasibility tests should use unrelated or fictional inputs. Final model identities, revisions, precision, relations and sampling rules must be fixed before the new evaluation.

Recommended primary comparison: probe-plus-source versus probe-plus-alternative-query, with the explicit agreement feature common to both. Recommended primary measurement: the bounded probability document-substitution outcome, with gradient boosting fixed in advance. Retain log support, Ridge, probe-only and selected-observer comparisons as specified secondary analyses. Finalize this choice after the control pilot and before registration; do not reclassify the old Fresh40 analyses as prospective.

Choose explicitly between inference for frozen fitted predictors and inference for a training procedure. Fixed-predictor intervals are valid for the former. If a training-procedure claim is intended, design a development-group resampling/refit analysis that repeats learned text forecasts, supervised tuning and observer selection where applicable. Refitting only the final decoder omits parts of training uncertainty. Pooled nested cross-validation is a separate estimand and must not replace the new held-out evaluation under the same label.

The external registration must bind the final protocol, data manifest, fitted assets, primary comparison, multiplicity policy, sample-size justification, exclusion rules and stopping rule. A local hash or commit without a trusted external timestamp is not an external preregistration. Register only once the package is actually ready and before new evaluation scoring.

## Steps 7–8 — Execution and submission

Collect the complete fixed-size evaluation with resumable raw-score checkpoints. Run the registered analysis even if estimates are zero, negative or imprecise. Explain departures transparently. The purpose is to resolve uncertainty, not to manufacture a positive increment.

Update the article around the resulting evidence and its remaining limits. A well-supported measurement finding can be valuable without demonstrating a positive source advantage; novelty, breadth and clarity still matter. No confidence-bound threshold determines journal tier. Verify references against primary records, final journal scope, current JCR year/category and the no-APC publication route. JAIR should not be treated as an easy fallback simply because a ranking service lists Q2.

Publish an immutable research release or deposit and insert its actual identifier into the manuscript. Preserve historical omissions and distinguish scientific archives from compiled manuscript files. A new rerun is a new study; it does not retroactively recreate unavailable historical evidence. External publication/deposition is a later action, not something already achieved by these local packages.

## Delivered record for completed steps

- `new_analysis/revision_20261027/analysis_spec.json`: explicit post hoc analysis scope recorded before executing the added script.
- `new_analysis/revision_20261027/analyze_saved_scores.py`: CPU implementation.
- `new_analysis/revision_20261027/results/`: complete tables, diagnostics and verification.
- `new_analysis/revision_20261027/REPORT.md`: interpretation and limitations.
- V6 KAIS Overleaf project: completed results only; restored observer-centered narrative, unchanged estimates, no new-study results or registration claims.

Later steps remain pending until their completion criteria are met. Update this plan's status table after each step and retain a dated change history.

## V6 editorial record

Restored the central answering-model versus observer question; retained the measurement corrections, stronger visible baselines, bounded-outcome sensitivities, target-specific intervals and expanded factorial readouts. Reordered the source-versus-observer result ahead of conditional utility, rewrote the framing and discussion, and converted the project to the Springer Nature template with numbered references. No new model inference, predictor fitting or statistical estimates were produced. See REVISION_NOTES_V6.md and MANUSCRIPT_VALIDATION.json.
