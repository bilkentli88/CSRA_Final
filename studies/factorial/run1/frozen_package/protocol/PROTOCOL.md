# Source report wording and answer visibility

## Purpose and status

**Exploratory amendment, 19 October 2026:** This existing-item pilot now uses an internal before-run source snapshot, without requiring external registration. EXPLORATORY_AMENDMENT.md and the revised implementation addendum govern the current workflow. The original registration-focused draft is preserved in protocol/history/. No registration, independent timestamp, untouched confirmation, or new pilot results are asserted by this package. All estimands, prompts, systems, folds and numerical analysis settings below are retained. Any future fresh study requires its own finalized plan; its registration is a separate methodological choice.

### Original design context

This protocol tests how ownership wording, epistemic hedging, and access to the recorded answer affect forced-choice source reports. It specifies a complete eight-cell experiment. All cells use fresh reporting prompts with identical access to the question and document. Ownership is a textual framing manipulation, not a manipulation of private access to a previous computation.

The original review draft was dated 16 October 2026; this implementation amendment is dated 19 October 2026. It is not registered or externally timestamped. No new pilot report outcomes were generated in preparing this package. Pilot items already have Study 1 outcomes. Any registration must distinguish prospective new report contrasts on those items from a confirmatory evaluation on fresh items. The working manuscript contains no Study 2 results.

The pilot uses the existing 100 items and three frozen systems. The provisional fresh-study candidate is 600 items, subject to the precision and feasibility criteria below; the higher-resolution 300-item planning check failed the conservative uncertainty criterion. Existing and fresh items will never be pooled as an untouched confirmatory sample. The design permits positive effects, negative effects, precise small effects, and inconclusive outcomes without changing the primary question or choosing a journal from pilot significance.

## Questions and scope

The questions are whether ownership and hedging shift the reported probability of both sources, whether displaying the answer shifts that probability, and whether displaying the answer changes the predictive increment of OWN reports after OTHER reports. Report dependence on the answer is evaluated as an effect of availability and as predictive redundancy. It is not interpreted as causal mediation, a fraction of a mechanism, or a direct measure of introspection.

Observer-model and answer-continuation conditions are excluded from this core study. A constructed assistant answer does not recreate an earlier answering computation, and a stronger predictor from the same model does not identify privileged access. Those questions need separate controls and a separate protocol if pursued.

## Complete factorial design

The factors are ownership (OWN, OTHER), question wording (certain, hedged), and recorded-answer visibility (shown, hidden). Every item, system, and primary document strength is evaluated in all eight cells and four cyclic role-label mappings. Report requests are independent fresh conversations. No answer-stage cache or hidden state is carried forward.

| Cell | Ownership | Wording | Answer |
|---|---|---|---|
| C1 | OWN | certain | shown |
| C2 | OWN | certain | hidden |
| C3 | OWN | hedged | shown |
| C4 | OWN | hedged | hidden |
| C5 | OTHER | certain | shown |
| C6 | OTHER | certain | hidden |
| C7 | OTHER | hedged | shown |
| C8 | OTHER | hedged | hidden |

OWN states 'You previously answered'; OTHER states 'Another instance of the same model answered.' The question adds only 'likely' in the hedged condition. Complete stems and punctuation are supplied in prompts.json.

Only the shown cells contain the Recorded answer block. The answer string is never repeated inside a stem. Hidden cells omit that block entirely. Source-role definitions, system instruction, document, question, and scoring suffix remain identical. Candidate answer names may naturally occur in the document or question; hiding the recorded choice does not hide all evidence about which choice is likely.

All eight cells are new matched prompts, including the two cells resembling the old SELF and PEER conditions. The original cells are not reused as factorial observations because the new stems remove an inline repetition of the answer. Fixtures and checks in code/pilot_common.py and tests/selftest.py verify all eight combinations, visibility control, and consistent role definitions.

## Systems and measurement

The pilot uses the three original repository and tokenizer revisions with their original NF4 weights, double quantization, and FP32 computation. The exact lock is copied to protocol/model_lock.json, without any old holdout-family labels. These systems are development systems. No general model-family claim is planned.

Primary report conditions are C_SKEPTICAL, C_PROVISIONAL, and C_BARE. There are four report roles: document, prior, both, and uncertain. Four cyclic mappings are scored. Log supports are averaged over mappings, then normalized with softmax to obtain a four-role distribution. The resulting q_both is the probability assigned to both after log-score aggregation, not a per-permutation probability average. Modal counts are secondary descriptions.

The answer stage retains the eight original conditions and six label permutations. The recorded answer is the role with the largest permutation-averaged log support, not a sampled response. The target is the strength-matched C-document versus P-document difference in C-versus-P log support. It is called the document-substitution effect. Answer stage, target construction, label qualification, and model precision stay fixed across all report cells.

The fresh study initially retains the same three systems and configuration to isolate the reporting manipulation. A full-precision or additional-model extension requires a complete crossed rerun of its answer stage and eight report cells under a new locked addendum. It cannot be mixed with old cells from another precision setting.

## Four primary contrasts

For each item and model, average over the three document strengths before averaging items and the three fixed systems. Models are weighted equally; they are not treated as a random sample of architectures. The first three contrasts use q_both and marginalize equally over the other two factors.

- P1 ownership: mean q_both in OWN minus OTHER.
- P2 hedging: mean q_both under hedged minus certain wording.
- P3 answer visibility: mean q_both with the answer shown minus hidden.
- P4 predictive visibility: theta_OWN shown minus theta_OWN hidden, averaged equally over the two hedging levels, with theta_OWN defined as the held-out gain from adding OWN after OTHER to the answer-omitted baseline.

All four contrasts are two-sided. A positive or negative estimate is reported in its observed direction. No claim that hedging dominates ownership is made from significance in one contrast and nonsignificance in another. The signed and absolute differences between those effects are secondary, with no default label of mixed mechanism for an inconclusive interval.

Primary inference uses 10,000 relation-stratified paired-item bootstrap draws with seed 20261016. All cells, strengths, and systems of an item are resampled together. In the pilot, shared answer triples are the resampling clusters within relation. Fresh items require unique ordered answer triples, so the prescribed triple resampling reduces to item resampling. Shared entities and relation effects can still create dependence; unique triples are not a proof of independence.

Four Bonferroni-adjusted two-sided percentile intervals use tail probabilities 0.00625 and 0.99375 (98.75% marginal intervals). Their simultaneous coverage is nominal and relies on adequate marginal bootstrap behavior. A contrast is declared distinguishable from zero only if this adjusted interval excludes zero. Ordinary 95% intervals and per-system estimates are reported as secondary. This specifies the family explicitly instead of combining a Holm label with unadjusted intervals. No confirmatory per-system declarations or exploratory interactions are added to that family after inspection.

## Prediction and secondary analyses

The primary predictor is Ridge with alpha 1 and a free intercept. X1 contains baseline no-document margin, strength, and relation. X2 adds the three OTHER log ratios relative to prior; X3 adds the corresponding three OWN ratios. Each hedging and visibility cell uses the same item folds. Numeric scaling and any imputation are fitted only on training folds. The target denominator is the population variance of the original target values for each system, held fixed within its bootstrap. This target normalization is distinct from recomputing report scales for correlation inference.

Five folds hold complete answer triples together and balance relation counts. Within relation, groups are ordered by descending size and a SHA-256 tie breaker using seed 20261016; assign to the least-loaded relation fold, then least-loaded total fold, then a hash tie breaker. The fold map is frozen before report scoring. Every bootstrap draw refits preprocessing and prediction while retaining this map. P4 uses the equal-system mean of R2(X3)-R2(X2), shown minus hidden, averaged across hedging.

Secondary analyses add the frozen answer role to X1; replace the three log ratios by document-minus-prior VSA; report full category distributions and answer association within document strength and relation; and estimate factorial interactions. A secondary nonlinear probe is GradientBoostingRegressor with 100 estimators, learning rate 0.05, maximum depth 2, squared-error loss, subsample 1, and random_state 20261016. It uses the same folds and predictor sets without hyperparameter search. These results receive descriptive intervals and do not enlarge the primary family.

Correlation differences and adjusted frame slopes, if reported, recompute means, SDs, correlations, and nuisance fits in every bootstrap draw. No fixed-reference slope interval is labeled as a recomputed correlation-difference interval.

The historical 0.01 R2 threshold is a secondary reference, with 0.005 and 0.02 sensitivity. It is not a validated utility threshold. Absence of a significant effect is not equivalence. Any later equivalence claim requires justified two-sided margins and adequate interval precision specified before confirmatory scoring.

## Item construction and exclusions

The pilot uses the existing selected 100 items without replacements, while retaining their known review limitations. It measures feasibility and precision on previously used material.

For the fresh study, build an independently sourced, versioned candidate pool in the four existing relation types. Provisionally select 150 items per relation, with unique subjects and unique ordered P/C/I triples. Correct, conflicting, and irrelevant alternatives must have semantic types compatible with the question. Multiple valid answers, ambiguous subject identity, unsupported factual claims, or incompatible alternatives are rejected before document/report outcomes exist. Two identified human reviewers work independently while blind to model outcomes; agreement and every adjudication are recorded. Human staffing and AI assistance must be truthfully documented before this stage starts.

Predeclared no-document screening restricts the population to items for which all three models prefer P under the qualification rule. This is an explicit outcome-based eligibility screen, not a technical exclusion. Record the full screened denominator and all failures. Select eligible reviewed items by the seed-hash rule before generating any C/P document scores or new report outcomes. If a relation cannot fill its allocation, amend the design before those outcomes rather than quietly reallocating or replacing items.

After selection, failures of label-token qualification, nonfinite or missing scores, and documented execution corruption are technical failures. Retry only with equivalent scoring semantics and predeclared batch-size reductions. Never exclude an item because its effect, report, or hypothesis result is inconvenient. After retries at batch sizes 4 and 2, unresolved missing paired cells exclude the entire item jointly across all cells and systems; every exclusion is reported and there are no replacements. The target variance is then fixed from this common complete-case set before bootstrapping. If attrition exceeds 5%, label the confirmatory evaluation incomplete and limit conclusions to the achieved complete-case population. Technical attrition may be nonrandom. There is no interim significance-based stopping.

## Sample size and feasibility gates

The pilot has 100 previously used items, three systems, three report conditions, eight cells, and four mappings: 28,800 logical report-scoring jobs. It reuses answer summaries but generates every new report cell. A fresh 600-item study adds 86,400 answer jobs and 172,800 report jobs, totaling 259,200. The 300-item alternative would total 129,600 jobs. Screening, qualification, retries, and any precision extension are additional. Logical jobs do not predict wall time, GPU memory, or monetary cost.

Planning compares 100, 300, and 600 independent synthetic items with ideal linear report coordinates. It uses the three-ALR representation and a shown-minus-hidden oracle gain of 0 or 0.02. It includes the finite-probe cost and the four-test interval level. This is favorable analysis calibration, not empirical power for natural reports, nonlinear information, endogenous answer roles, or shared semantic entities.

A candidate target is acceptable for further planning only if the lower Wilson bound for the chance that the adjusted lower endpoint exceeds 0.01, at oracle P4 0.02, is at least 0.80. The initial 399-draw sensitivity has very sparse adjusted tails. In the 1,999-draw 300-item check, 85 of 100 datasets had an adjusted lower endpoint above 0.01, with a Wilson interval of 76.7% to 90.7%; its lower bound therefore failed the 80% requirement. The prespecified 600-item alternative is also assessed at 1,999 draws, with all runs retained in the package. Before final registration, check the category endpoints using pilot estimates of paired-item contrast variability. If these depart materially from the planning assumptions, repeat the calibration with those variance scenarios and at least 1,999 draws. Report Monte Carlo intervals and keep the final outcome-analysis count at the prespecified 10,000 draws. The current simulation is not a guarantee of achieved empirical power.

The higher-resolution 600-item check yielded 99 of 100 datasets with an adjusted lower endpoint above 0.01 (Wilson interval 94.6% to 99.8%). It passes the conservative planning criterion under the idealized generating process; empirical report precision and resource feasibility remain to be established.

For category contrasts, the provisional precision goal is an adjusted interval half-width no larger than 0.03. Under a normal approximation with three fixed systems, cross-system correlation 0.5, and paired-item SD 0.2 per system, 300 independent items give an adjusted half-width about 0.024. These assumptions must be checked against the pilot. They do not establish power under arbitrary report distributions.

Proceed to fresh evaluation based on technical feasibility, annotation quality, and attainable prespecified precision, regardless of the direction or significance of pilot effects. Select the smallest candidate meeting both criteria; the current higher-resolution result rules out 300 under this conservative planning rule. If neither fits the resource budget, narrow the claims and report precision limits rather than asserting equivalence or choosing a favorable outcome. The final item count is frozen before fresh outcome scoring.

## Registration and interpretation

Before pilot report execution: review the prompt fixtures; verify model/tokenizer availability and label scoring; run synthetic technical preflight for all three models; and save the frozen package snapshot and internal timestamp using the freeze command. No external registration is required in exploratory mode. Never describe the old items as fresh, the internal timestamp as independent, or this pilot as preregistered.

Before any fresh study: identify the factual reviewers and their roles, finalize the candidate pool and sampling rule, resolve the sample-size gate and budget, and freeze hashes and the fold map. External preregistration can be chosen to support stronger prospective-evidence claims; it is not imposed by this exploratory notebook. A registration now cannot supply missing historical provenance for Study 1. Keep the old record's verified facts and unresolved gaps distinguishable.

Effects of hedging or ownership establish sensitivity to those textual manipulations. A visibility effect establishes sensitivity to availability of the recorded answer under matched prompts. Predictive redundancy remains specific to baseline, representation, and learner. A narrow null interval constrains the specified effect; a wide interval is inconclusive. None of these outcomes alone demonstrates or rules out privileged internal access.
