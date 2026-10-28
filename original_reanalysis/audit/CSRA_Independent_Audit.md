# Independent audit of the CSRA v0.8 submission

15 October 2026 | Manuscript IPM-D-26-05842

Updated after recovery of the execution packages, preactivation assets and factual-review CSVs: the two lock hashes refer to distinct, internally linked documents; the exact prompts, ten fictional controls and sealed 240-row review are verified. Deterministic selection reproduces the released final 100-item CSV byte for byte. The numerical audit results are unchanged.

**Assessment:** the published numerical results reproduce, and the two most consequential reviewer reanalyses are supported. Substantial revision is justified. The existing study supports an answer-conditioned predictive result and ownership-framing effects. Its present design does not establish a difference in access to the computation that produced an answer. Publication readiness and originality remain unresolved by numerical reproduction alone.

## Scope and verification

The principal source is the replacement upload **CSRA-8000.zip**, the v0.8 R4/A2 common-item release. It contains 100 selected factual items, 1,800 primary alignment rows, 900 predictive rows, aggregate answer/report scores, frozen specifications, and expected results. The first upload, **csra-source-attribution.zip**, contains the earlier v0.6 reserve experiment and was excluded from the v0.8 numerical analyses. It was used only to inspect model-history claims.

The repository's official `run_reproduction.py` passed in its pinned statistical environment: NumPy 2.0.2, pandas 2.2.2, SciPy 1.16.3, scikit-learn 1.6.1, and statsmodels 0.14.6. Its default check recomputes the principal point estimates and specified sensitivity/diagnostic summaries.

An independent implementation reconstructed CIE and VSA from the released score summaries. It used the original five item-grouped folds, Ridge penalty 1, model-specific population-variance denominators, and 2,000 relation-stratified paired common-item bootstrap draws. Numeric preprocessing and Ridge fits were refitted in each draw. Weighted sufficient statistics make these refits fast without changing the estimator; comparisons with the released scikit-learn implementation checked both the original and answer-omitted branches.

The original alignment interactions and their bootstrap intervals were reproduced through algebraically equivalent paired-difference regressions. This avoids thousands of redundant mixed-model optimizations. All five original alignment intervals, and the primary predictive interval, match the sealed values within 1e-8. Exact implementation differences are recorded in `results/implementation_validation.json`.

This audit verifies released statistical summaries and recovered review/selection records. Language-model scoring, factual correctness of review decisions, model training history, and the prospective timing of locks were not independently established. Additional analyses were chosen after peer review and are exploratory.

## 1. The primary predictive null depends on the baseline information

The table shows the additional held-out predictive contribution of SELF after PEER. Units are changes in R-squared, not fractions of the preceding model's residual error.

| Model | With recorded answer: SELF increment | 95% bootstrap CI | Without recorded answer: SELF increment | 95% bootstrap CI |
|---|---:|---|---:|---|
| Qwen | 0.001043 | [-0.002255, 0.004962] | 0.000068 | [-0.005413, 0.004852] |
| Granite | -0.002070 | [-0.013389, 0.008354] | 0.037405 | [0.011858, 0.074293] |
| Phi | -0.000909 | [-0.007879, 0.005393] | 0.007599 | [-0.003839, 0.022109] |
| Equal-model mean | **-0.000645** | **[-0.005762, 0.003297]** | **0.015024** | **[0.005249, 0.026449]** |

At the existing delta=0.01 rule, the original average is below the practical threshold. The answer-omitted average has a positive interval that crosses 0.01, so its practical magnitude is indeterminate. Granite's answer-omitted interval is entirely above 0.01. Qwen remains below the practical threshold under both baselines.

The mean held-out R-squared values explain this change:

| Predictors | With recorded answer | Without recorded answer |
|---|---:|---:|
| Non-report features | 0.877261 | 0.742978 |
| Add PEER VSA | 0.879409 | 0.818511 |
| Then add SELF VSA | 0.878763 | 0.833534 |

Including the recorded answer is defensible for assessing value beyond an already visible answer. Item-grouped prediction avoids putting the same item's strength rows into both training and test folds. The problem is the scope of the conclusion: conditioning on an answer derived from the same answering scores as CIE changes what remains to be predicted. The result cannot establish that SELF reports are generally uninformative or lack self-specific access.

**Retain:** under the specified linear Ridge probe and scalar VSA measure, SELF adds little beyond PEER and the baseline that already contains the recorded answer. Present the answer-omitted analysis alongside it as an exploratory sensitivity analysis.

## 2. Alignment combines association and score scale

| Model | Correlation: CIE with PEER | Correlation: CIE with SELF | SELF/PEER score SD | Original slope difference | Within-frame-standardized difference |
|---|---:|---:|---:|---:|---:|
| Qwen | 0.892295 | 0.880868 | 1.178600 | 0.133315 | **-0.011426** |
| Granite | 0.684204 | 0.746526 | 1.024351 | 0.074191 | 0.062322 |
| Phi | 0.810858 | 0.832449 | 1.002058 | 0.021825 | 0.021591 |

For Qwen, the larger SELF slope reflects greater score dispersion, while its Pearson association with CIE is slightly weaker. The within-frame Qwen interval is [-0.025149, 0.002318]. Granite retains a positive unadjusted association difference, with interval [0.004345, 0.119586]; Phi remains uncertain.

The common difference decreases from 0.076444 to 0.024162, with the latter interval [0.000485, 0.050450]. The common association therefore does not disappear entirely. The Qwen-versus-other-model contrast H changes from 0.085307 to -0.053383, with the latter interval [-0.094410, -0.011653]. The claim that Qwen has the strongest alignment advantage is sensitive to the measurement scale.

### Why the original contrast is not covariate-adjusted as described

Every item-condition pair has the same CIE and nuisance covariates in SELF and PEER. When those covariates enter only additively, they cancel in the SELF-minus-PEER difference. The original interaction is consequently the slope of that paired difference on CIE. Adding nuisance-by-frame interactions changes the conditional estimand.

Separate conditional paired-difference regressions, allowing nuisance effects to differ by model and frame, reproduce the reviewer's alternative point estimates:

| Model | Conditional, within-frame-standardized difference | Item-cluster standard error |
|---|---:|---:|
| Qwen | 0.049919 | 0.027227 |
| Granite | 0.173093 | 0.083591 |
| Phi | 0.001327 | 0.051278 |

These are exploratory estimates. Their raw p-values in the supporting table are not multiplicity-adjusted. The revised analysis should choose one stated conditional estimand and one cluster-aware inferential framework.

The original mixed model also gives negative adjusted CIE slopes for Granite in both frames (PEER -0.225810; SELF -0.151619). A positive difference between two slopes does not by itself establish positive adjusted source-report alignment. Report both frame-specific slopes.

## 3. The predictive metric needs a manuscript correction

The code computes `(MSE_X2 - MSE_X3) / Var(CIE)`, with population variance fixed within each model. On the original evaluation sample this equals `R2_X3 - R2_X2`. The frozen specification agrees with this implementation.

The manuscript's Equation 19 instead uses `(SSE_X2 - SSE_X3) / SSE_X2`, a relative residual-error reduction. These are different metrics. For example, Qwen's original SELF increment is 0.001043 in R-squared units but about 0.010574 as a fraction of X2 residual error.

Correct Equations 18-19 to match the implemented estimand, and retain the practical threshold in those units. A threshold of 0.01 denotes one percentage point of total target variance, not 1% of remaining prediction error. Its practical relevance still needs justification; having fixed a threshold prospectively does not itself establish its substantive importance.

## 4. Full-category diagnostics materially change the picture

- In primary Qwen reports, a recorded C answer produces a document report in 176/180 observations (97.8%, pooling frames). A recorded P answer produces a prior report in 350/420 (83.3%). This supports strong dependence on the visible answer.
- Granite reports **both** in 280/300 primary PEER observations (93.3%) and 175/300 SELF observations (58.3%). A document-versus-prior log-odds contrast omits its dominant categorical behavior.
- Phi never selects **prior** as its modal category in the 1,000 principal report observations across all five conditions.
- In the P-asserting provisional condition, Qwen reports document in 89/100 observations in each frame, and Phi in 100/100. These counts alone do not prove misattribution: a document may reinforce an already preferred answer. They motivate comparing reports against measured intervention effects rather than modal-answer agreement alone.
- Granite's fictional-control document-report share is **0/10 PEER and 9/10 SELF**. Pooling them into 0.45 conceals a substantial framing effect.

The complete four-category distributions, all P/I conditions, conditional distributions, and fictional controls by frame are included in the CSV outputs. A revised measurement analysis should use the complete four-role score vector or another explicitly justified measure, alongside the original VSA contrast. Qwen's CIE standard deviation is 23.44 nats, reinforcing the need for probability/rank robustness checks; per-permutation saturation claims cannot be fully verified from aggregate scores alone.

## 5. SELF and PEER do not establish different computational access

The scoring code constructs independent report prompts from item information, document text, and the frozen answer. Answer-phase hidden states are not passed to the reporting computation. The frozen answer is chosen from six permutation-averaged scores, rather than being one sampled conversational answer.

The current comparison measures the effect of describing the answer as one's own or another instance's, with additional wording differences. The recovered principal SELF stem asks which source "most influenced" the answer, while PEER asks which source "most likely influenced" it. The alternate wording preserves this epistemic asymmetry. Thus ownership and uncertainty wording are not independently varied. This is a possible prompt confound; the existing data do not isolate its contribution to the observed effects.

Neither a positive SELF increment nor its absence establishes privileged internal access. A same-conversation reporting variant could improve the task design, but would not by itself establish such access either. Exact templates and one reconstructed report-prompt example are now supplied with this audit.

## 6. Provenance and release issues remain

| Check | Finding |
|---|---|
| Available checksum-listed release files | All 45 present files match the release checksum list. |
| Post-factual-review lock | **Document identity resolved by the recovered answer-execution TR2 package.** The longer design-metadata document has hash beginning `5212466e`; a separate compact canonical execution lock has hash beginning `92e7c26f`, matching the runner. A technical-completion record explicitly binds both. |
| Model-family roles | The v0.8 lock labels all three as continuity families. The supplied v0.6 archive already contains results for the same model revisions, including Granite and Phi. Describe development/holdout status relative to a specific stage and document what informed prompt design. |
| Answer-pattern diversity | The 100 items contain 57 distinct ordered P/C/I answer triples. This warrants grouping/sensitivity checks and restricted generalization claims; it does not automatically make 57 the effective sample size. |
| Timing of locks | Self-declared dates and hashes provide no independent external timestamp in the supplied archive. Existing public history may supply evidence, but a registration created now cannot establish the original timing. |
| Item review | **Recovered.** The sealed 240-row CSV matches the recorded seal, gate and lock hashes. All 69 exclusions have reasons, including 57 country-of-origin items with the irrelevant option "Africa". The specified selection rule reproduces the final 100-item CSV byte for byte. |

### Recovered explanation of the lock hashes

`CSRA_v08_ANSWER_EXECUTION_R4_A2_TR2_PACKAGE.zip` supplies both documents:

- `CSRA_v08_POSTREVIEW_ANALYSIS_LOCK_R4_A2.json`: the 4,335-byte design-metadata document, exactly matching the file released in the anonymous repository; SHA-256 `5212466e65902c8b4ca28d21c31af17e8cf93f8846f843b886d62a32e1fd4703`.
- `postreview/postreview_analysis_lock.json`: the 899-byte compact execution lock, SHA-256 `92e7c26fd9f7a0782b309c419d9ad2b69c8fa5e84d1677a32cee77b54edb9bd2`, exactly matching the value required by the execution runners.
- `CSRA_v08_POSTREVIEW_TECHNICAL_COMPLETION_R4_A2.json`: explicitly records the first as its upstream metadata lock and the second as its generated canonical output. It explains that the metadata document did not supply the companion machine-readable fold, relation-map, alternate-subset, and result-shell files required downstream.

All 16 package checksum entries pass. The canonical lock's available file bindings match, and all six supplied fold maps agree with the frozen deterministic rules. The selected 100-item CSV also matches the reviewed release byte for byte. This supports a documented distinction between design metadata and an execution lock. The public archive should include both under distinct names with the technical-completion record. External evidence of pre-outcome timing remains a separate question.

### Recovered prompts and controls

`CSRA_v08_PREACTIVATION_COMPLETE_PACKAGE.zip` supplies the exact prompt JSON, original frozen prompt-template ZIP, analysis ZIP, original R2 protocol and approval record, and all ten fictional-control item specifications. Their relevant SHA-256 values match the execution commitments, and all 14 Step-2 component-manifest checks pass. This resolves the missing-template and missing-control-definition parts of the release concern. Original individual scoring checkpoints are still needed to check executed prompt hashes against these reconstructed templates.

The recovered R2 protocol explicitly calls all three models continuity families and states that v0.6 results informed the minimum-item gate and candidate allocation. This strengthens the recommendation to replace the manuscript's model-family holdout labels with an accurate account of the development history.

The R3/A1 approved package documents an earlier candidate-builder amendment that retained exact answer-triple uniqueness. It is historical context, not the final R4/A2 amendment used for the submitted results. The TR3 report-execution package provides the report-phase repair wrapper; its repair specification matches the reviewed release.

### Recovered factual review and deterministic selection

The uploaded `csra_v08_blinded_common_review_SEALED.csv` and `csra_v08_blinded_common_review_AUDITED.csv` are byte-identical, with SHA-256 `1e4378dbb59e9cd8034cf3f1a1244ea5d5f210497ac802fe119377ed4b3ff6d1`. This matches the original review seal, post-review gate and metadata analysis lock. Both have 240 unique item IDs, 171 APPROVE decisions and 69 EXCLUDE decisions. Every exclusion has a nonblank reason.

The unnumbered plain CSV is a blank review template. The numbered `(1)` copy marks all 240 items APPROVE without reasons. All four copies preserve exactly the same non-review fields and item order. Comparing the all-APPROVE copy with the sealed version reveals 69 changed decisions and their reasons. The filenames do not establish when or by whom those changes were made; AUDITED and SEALED are two names for the same supplied bytes, not evidence of two independent reviews.

| Relation | Reviewed | Approved | Excluded | Selected |
|---|---:|---:|---:|---:|
| Country of origin | 57 | 0 | 57 | 0 |
| Developer | 65 | 61 | 4 | 25 |
| Formation location | 54 | 50 | 4 | 25 |
| Original language | 35 | 35 | 0 | 25 |
| Original network | 29 | 25 | 4 | 25 |
| **Total** | **240** | **171** | **69** | **100** |

All 57 country-of-origin rows use **Africa** as their irrelevant answer and record the same reason: a geographic region/continent is not a country. The records therefore explain the loss of that whole relation through a shared type mismatch. The remaining 12 exclusions cite source mismatch, ambiguity or non-single-valued attribution, and one malformed formation-location question about a person. These are supplied review reasons; this audit has not independently checked their factual claims against original sources.

Reapplying the frozen SHA-256 ranking, taking the first 25 approved items per relation without reallocation, reproduces all released final rows, item order, selection hashes and ranks. With the released CSV's CRLF record endings, the reconstructed file is byte-identical, with SHA-256 `19c57a4a96db4d8bbe2cb529ed94a74e156eb9f1f2c407660cf2530132c90b98`. The count of 100 also passes the specified minimum of 80. All 19 review/selection checks pass.

This resolves the missing review-file, exclusion-reason and final-selection reconstruction gaps. The revised methods should describe the attrition from 240 reviewed items to 171 approved and 100 selected, and that the analyzed set covers four relations. The CSVs do not independently establish reviewer identity, independence or timing, and do not reproduce the model-screening step that produced the 240 review items.

Materials still absent include the official final R4/A2 600-candidate pool and raw source snapshot, screening outputs, actual activation manifest, final R4/A2 amendment documentation, and individual permutation scores. Some hashes are retained, but the missing content cannot be recovered from hashes. The procedural note about inspection of review-status counts before sealing also needs an accurate account of human and AI roles. It does not establish who performed which decisions.

## Recommended revision sequence

1. **Complete the archive.** Include the recovered prompts, control items, protocol, sealed review, both lock documents and technical-completion record. Locate the final R4/A2 amendment/activation records, candidate snapshot, screening outputs and checkpoints. State model-development history, review provenance and AI contributions accurately. Preserve original records and explain packaging changes rather than rewriting historical evidence.
2. **Revise the analysis and claims using existing scores.** Retain the original specified results; identify all new analyses as exploratory. Add answer-present/answer-omitted results, full-role diagnostics, a justified scale-independent alignment estimand, frame-specific slopes, and consistent cluster-aware inference. Investigate probability/rank sensitivity and grouping by repeated answer patterns.
3. **Establish the substantive contribution before larger runs.** The strongest candidate question is how much source reporting follows visible answer-document agreement, after accounting for intervention-measured source influence. This audit supports investigating that question; it does not establish that the question is a novel literature gap.
4. **If new evidence is needed, use a focused diagnostic.** Hold answer computation and its CIE fixed while varying the displayed answer in the report prompt, including an answer-withheld condition. Compare with a simple agreement-based predictor. This tests reporting sensitivity to visible information, not privileged introspection. Independently timestamp any new confirmatory protocol before collecting its outcomes.

The original reproducibility work is worth retaining. The article should be rewritten around a demonstrable result and its limits before selecting another journal. Increasing the number of models or repeating the present statistics would not, by itself, resolve Reviewer 2's concern about substantive novelty.

## Reproduction files

Run the audit against the extracted replacement repository:

```bash
python -m pip install -r requirements.txt
python audit_csra.py --repo /path/to/extracted/CSRA-8000 --out results --reps 2000
```

The repository root is the directory containing its `run_reproduction.py` and `data/` folder. No GPU is required. Input ZIP and review-CSV hashes are recorded in `input_archives.json`. The source repositories and their historical locks are not modified by the audit. The `model_history_comparison.csv` table additionally uses the first uploaded archive and is not regenerated by the single-repository audit command.

To reproduce the recovered lock-lineage check after the statistical audit:

```bash
python audit_lock_lineage.py --package /path/to/CSRA_v08_ANSWER_EXECUTION_R4_A2_TR2_PACKAGE.zip --repo /path/to/extracted/CSRA-8000 --out results/lock_lineage_verification.json
```

Exact recovered lock, completion, seal, and companion JSON files are included under `recovered_provenance/` for inspection. Their presence explains the hash discrepancy; it does not change any statistical estimate.

To verify the recovered preactivation assets:

```bash
python audit_preactivation_assets.py --package /path/to/CSRA_v08_PREACTIVATION_COMPLETE_PACKAGE.zip --out results/preactivation_asset_verification.json
```

The exact prompt JSON, controls, original protocol and freeze records are preserved under `recovered_provenance/preactivation/`. `recovered_prompts_example.md` shows one report example reconstructed from those templates and the released item/answer summaries, without any new model scoring.

To verify the recovered review versions and final selection from within this audit directory:

```bash
python audit_factual_review.py --reviews recovered_provenance/factual_review --repo /path/to/extracted/CSRA-8000 --records recovered_provenance --out results/factual_review_verification.json
```

All four original uploaded review CSVs are preserved unchanged under `recovered_provenance/factual_review/`. The script verifies their bindings, recorded counts, exclusion-reason completeness and exact final-item reconstruction. It does not perform a new factual review or establish the chronology of the supplied records.

Tables and draw-level outputs are under `results/`. The confidence intervals assume the original relation-stratified common-item sampling model and fixed normalization constants; they should not be interpreted as uncertainty over all model families or arbitrary factual domains.
