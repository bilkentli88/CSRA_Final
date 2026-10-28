# CSRA saved-score revision results

27 October 2026. These are **post hoc exploratory analyses** of already inspected data. They add no language-model calls and perform no predictor refitting. Their specification was written before executing the new script, after the existing outcomes and earlier analyses were known. This is not prospective registration.

## Scope and verification

The script reads 2,880 rows of saved Fresh40 predictions (eight outcome/feature/decoder specifications, each with 360 cases) and 7,200 factorial report aggregates. It reproduces all 48 earlier pooled prediction contrasts and their intervals, all three original adjusted factorial category intervals, and nine original per-target category intervals. Maximum verification discrepancies are below 1e-12. All scientific source inputs remained unchanged. Each added bootstrap uses 10,000 paired whole-group draws. No new test items or model families have been evaluated.

## Target-specific prediction uncertainty

The following rows compare probe-plus-source against probe-only under gradient boosting. Values are absolute normalized error reductions, using each outcome's own development variance. They are not percentage gains and cannot be compared across outcomes as one common estimand. Intervals are descriptive unadjusted 95% whole-fact bootstrap intervals, conditional on the fixed fitted predictors.

| outcome | variant | target | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| bounded_probability | agreement | granite | -0.0052 | -0.0300 | 0.0187 |
| bounded_probability | agreement | phi | 0.0298 | -0.0337 | 0.0905 |
| bounded_probability | agreement | qwen | 0.0099 | -0.0071 | 0.0295 |
| bounded_probability | original | granite | -0.0014 | -0.0265 | 0.0235 |
| bounded_probability | original | phi | 0.0280 | -0.0172 | 0.0775 |
| bounded_probability | original | qwen | 0.0759 | 0.0071 | 0.1665 |
| log_support | agreement | granite | 0.0112 | -0.0033 | 0.0264 |
| log_support | agreement | phi | 0.1203 | 0.0260 | 0.2351 |
| log_support | agreement | qwen | 0.0089 | -0.0483 | 0.0592 |
| log_support | original | granite | 0.0122 | -0.0024 | 0.0275 |
| log_support | original | phi | 0.1213 | 0.0141 | 0.2410 |
| log_support | original | qwen | 0.0822 | -0.0207 | 0.2242 |

After explicit agreement is added for the log outcome, the individual Phi interval excludes zero; the Granite and Qwen intervals span zero. With bounded probability and agreement, all three target intervals span zero. This does not establish zero effects or equivalence. The selection of these readouts after inspecting Fresh40 makes them exploratory even though no predictors were refitted here.

## Omitting one target

These rows give the equal mean for the other two fixed targets. Their intervals resample facts; they do not estimate uncertainty over a population of models.

| outcome | omitted_target | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| bounded_probability | granite | 0.0199 | -0.0147 | 0.0553 |
| bounded_probability | phi | 0.0024 | -0.0096 | 0.0156 |
| bounded_probability | qwen | 0.0123 | -0.0122 | 0.0374 |
| log_support | granite | 0.0646 | -0.0003 | 0.1421 |
| log_support | phi | 0.0100 | -0.0206 | 0.0393 |
| log_support | qwen | 0.0658 | 0.0172 | 0.1262 |

Omitting Phi reduces the agreement-augmented log estimate to 0.0100 [-0.0206, 0.0393], and the bounded estimate to 0.0024 [-0.0096, 0.0156]. Thus the larger average gain is not supported uniformly across targets. Additional model families and sizes would address a different generality question from adding more facts for these three targets.

## Phi diagnostics

The relation-specific point gains below are descriptive. Each relation contains ten Fresh40 groups. They are not separately confirmed effects and have no assigned subgroup confidence interval in this diagnostic table.

| outcome | value | n_cases | n_groups | estimate |
| --- | --- | --- | --- | --- |
| bounded_probability | developer | 30 | 10 | -0.1049 |
| bounded_probability | formation_location | 30 | 10 | 0.1586 |
| bounded_probability | original_language | 30 | 10 | 0.0031 |
| bounded_probability | original_network | 30 | 10 | 0.0626 |
| log_support | developer | 30 | 10 | -0.0418 |
| log_support | formation_location | 30 | 10 | 0.4567 |
| log_support | original_language | 30 | 10 | 0.0613 |
| log_support | original_network | 30 | 10 | 0.0051 |

Phi's log-support point gain is concentrated in formation-location questions (0.4567), while developer questions have a negative point gain (-0.0418). Under the bounded outcome the corresponding values are 0.1586 and -0.1049. The pattern is heterogeneous even within Phi. It suggests examining task composition and errors in a new study; it does not identify an internal causal mechanism. `fresh_stratum_gains.csv` also gives strength and observed-agreement strata; `fresh_diagnostics.csv` retains absolute MSE, MAE, fresh R-squared, development and evaluation variances and agreement frequencies. `fresh_group_influence.csv` retains all group contributions.

## Factorial visibility contrasts

Every estimate below is shown minus hidden, averaged over ownership, hedging, strengths and items for the named system. `q_both` is the normalized four-category both-source support. `document_minus_prior` is q(document)-q(prior), with the same four-role normalization. `document_prior_logratio` is logp(document)-logp(prior). New intervals are descriptive unadjusted 95% intervals. The original four primary adjusted intervals remain unchanged.

| target | outcome | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| granite | q_both | 0.2726 | 0.2447 | 0.3026 |
| granite | document_minus_prior | -0.3047 | -0.3380 | -0.2749 |
| granite | document_prior_logratio | -0.6933 | -0.9718 | -0.4009 |
| phi | q_both | 0.0791 | 0.0628 | 0.0971 |
| phi | document_minus_prior | -0.0930 | -0.1210 | -0.0677 |
| phi | document_prior_logratio | 0.4049 | 0.2329 | 0.6089 |
| qwen | q_both | 0.0000 | -0.0000 | 0.0000 |
| qwen | document_minus_prior | -0.2848 | -0.3960 | -0.1848 |
| qwen | document_prior_logratio | -5.2145 | -6.5982 | -3.8902 |

Qwen's near-floor both-category result is not invariance of reporting: visibility changes its document-minus-prior balance by -0.2848 [-0.3960, -0.1848] and its document/prior log ratio by -5.2145 [-6.5982, -3.8902]. All four visibility-specific role means are supplied in `factorial_visibility_profiles.csv`, so the balance change need not be interpreted as a direct transfer from document to prior alone.

## Recorded-answer association

For each visibility condition, the statistic is mean(q_document-q_prior | recorded answer C) minus mean(q_document-q_prior | recorded answer P). The final row for each system is the shown-minus-hidden difference in that association. C is the conflicting document's entity; P is the reference answer. Answer role is not randomized. This is an association, not an attribution-accuracy score, calibration measure or direct estimate of reconstruction. Complete role summaries retain I answers; only this explicit C-versus-P comparison excludes I answers. The sole I case is a Granite item/strength case, with four ownership/hedging report rows in each visibility condition.

| target | visibility | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| granite | shown | 0.4302 | 0.3615 | 0.4989 |
| granite | hidden | -0.0100 | -0.0556 | 0.0307 |
| granite | shown_minus_hidden | 0.4402 | 0.3856 | 0.5005 |
| phi | shown | 0.5055 | 0.4012 | 0.6077 |
| phi | hidden | 0.1900 | 0.1552 | 0.2254 |
| phi | shown_minus_hidden | 0.3155 | 0.2340 | 0.3994 |
| qwen | shown | 1.8481 | 1.7813 | 1.9165 |
| qwen | hidden | 1.2298 | 1.0969 | 1.3589 |
| qwen | shown_minus_hidden | 0.6184 | 0.5053 | 0.7474 |

The recorded-answer association increases with visibility in all three systems. The paired group intervals for the differences exclude zero descriptively. This complements the both-category effects and does not prove that answer reconstruction explains the entire report distribution or its predictive utility. Association may also reflect relation and strength composition. Further causal interpretation would require additional controls.

## What remains unresolved

- No alternative-question control has yet been scored. A selected external source reporter does not isolate the effect of asking a source question.
- Fresh40 remains a purposive sample of forty groups across three known small targets and four relations.
- Added intervals condition on existing fitted predictors and do not include training-procedure or new-model uncertainty.
- The bounded-plus-agreement primary-decoder average still has an interval spanning zero. No new untouched confirmation or meaningful-effect threshold has been established.
- The original missing candidate/screening records and individual original NF4 checkpoints remain missing. These new calculations do not reconstruct them.

## Reproduce

From the repository root with the pinned Python 3.12 analysis environment:

```sh
python run_reproduction.py revision
```

This verifies release integrity and writes the new results into a fresh directory under `reproduced_results/`. The `quick` route also includes these checks. The separate `posthoc` route refits the October 26 text forecasters and decoders on the original development data only; it does not collect model scores. Historical experimental files and frozen packages are not rewritten.
