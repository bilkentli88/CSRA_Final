# CSRA R4 audit and post hoc robustness results

Date: 26 October 2026

## Assessment

The uploaded R4 repository contains the measurement-study reanalysis, the factorial experiment, the FP32 development study and the Fresh40 evaluation. The original Fresh40 results reproduce. The new checks retain a positive source-report increment for the original log-support outcome after adding explicit answer-document agreement, but qualify its magnitude and its stability across outcome definitions and targets.

The manuscript should retain its original results with their original specifications and add these analyses as post hoc sensitivities. It should not present the new checks as prospectively frozen or as a new untouched evaluation. The scientific question is now more precise: how much predictive value do source reports retain after inexpensive visible cues, and how does that value depend on the behavioral outcome and decoder?

## Repository verification

The uploaded source archive is `CSRA_Revised_Repository_R4(1).zip`, SHA256 `977087221bf0570991abf5f98635e028ee79cbb7a1f07d6cd3cd8173697400f0`.

The existing quick reproduction route passed:

- 399 inventoried file checks and the preserved study/package bindings.
- Prompt identity and saved-score aggregation checks for 89,280 later-study jobs: 28,800 factorial, 43,200 FP32 development and 17,280 Fresh40.
- All 98 manuscript numerical checks, maximum absolute difference 6.82e-13.
- The six original orthonormal full-role point refits included by that route.

The new script separately refitted the original text forecasts and six policy decoders on the development data for both algorithms. All 4,320 corresponding original Fresh40 predictions matched the stored predictions within 2.14e-14. The original source-added-to-probe point estimates and descriptive bootstrap intervals were reproduced. These checks do not repeat language-model inference, refit the full factorial bootstrap, or establish an external timestamp.

The old original collection omissions remain: R4 does not supply the complete initial 600-candidate/screening archive or every original NF4 scoring checkpoint. Its documentation explicitly records those limits. The package's content completeness for the later scoring studies should not be described as complete recovery of all original collection records.

## Analysis specification

All new supervised fits used only the original 100 FP32 development items (900 target/item/strength cases, 57 groups). Evaluation used the existing forty Fresh40 groups (360 cases). Original gradient-boosting and Ridge settings were retained. Nested grouped TF-IDF and MPNet text forecasts were recomputed separately for each outcome, using development labels only. No Fresh40 labels were used to choose hyperparameters or observers. The original observer choices remained Granite->Phi, Phi->Qwen and Qwen->Phi.

The agreement feature is one Boolean: the recorded answer equals the entity asserted in the displayed conflicting document. It is derived from already visible inputs and added to every compared policy. It requires no additional model forwards and supplies no factual-reference designation for the other alternatives.

The bounded outcome is

`T = (mean_prob_C - mean_prob_P)_C-document - (mean_prob_C - mean_prob_P)_P-document`.

Each mean probability is computed by exponentiating each saved full-vocabulary answer-label log probability and then averaging the six label permutations. The calculation does not exponentiate the mean log score or renormalize the three candidate labels. It is bounded between -2 and 2. All text forecasters and decoders were refitted to this target on development data, rather than evaluating log-scale predictions on incompatible outcomes.

Gains use each target's development variance for the relevant outcome. The log-support and probability rows therefore describe different estimands with different denominators. Values are not percentage improvements. All intervals below are marginal descriptive 95% percentile intervals from 10,000 paired whole-group resamples, conditional on the fitted development pipelines. They exclude training-sample and new-model uncertainty and are not adjusted for the exploratory comparisons.

## Does a source report add value to the direct probe?

| Outcome and common baseline | Gradient boosting: gain [95% interval] | Ridge: gain [95% interval] |
|---|---:|---:|
| Original log support, original features | 0.071883 [0.011510, 0.143928] | 0.168733 [0.049266, 0.340004] |
| Original log support, explicit agreement added | 0.046801 [0.002383, 0.100378] | 0.120977 [0.035996, 0.237689] |
| Bounded probability, original features | 0.034142 [0.008873, 0.064282] | 0.078146 [0.021305, 0.157384] |
| Bounded probability, explicit agreement added | 0.011517 [-0.006713, 0.031189] | 0.029158 [0.003430, 0.061134] |

For the original log-support target, adding agreement reduces the primary-decoder point gain but leaves its descriptive interval above zero. The lower bound is below the manuscript's 0.01 engineering reference. Under the combined bounded-outcome and agreement check, the primary gradient-boosting interval spans zero. Ridge remains positive on average, with strong target heterogeneity.

The per-target source-added-to-probe gains under the stronger agreement baseline are:

| Outcome | Decoder | Granite | Phi | Qwen |
|---|---|---:|---:|---:|
| Log support | Gradient boosting | 0.011212 | 0.120310 | 0.008882 |
| Log support | Ridge | 0.118319 | 0.264938 | -0.020326 |
| Bounded probability | Gradient boosting | -0.005221 | 0.029837 | 0.009936 |
| Bounded probability | Ridge | -0.001802 | 0.089642 | -0.000367 |

For the bounded outcome with agreement, the positive Ridge mean is concentrated in Phi; the other two target point estimates are slightly negative. A general claim of positive gains across all targets and reasonable specifications is therefore unsupported. The original statement that all three primary target gains are positive remains true for its original log-support specification, and for gradient boosting after agreement alone.

Leaving out any single Fresh40 group keeps the gradient-boosting mean source-added-to-probe point gain positive in all four scenarios. For the original specification its range is 0.052566 to 0.081406; for bounded probability plus agreement it is 0.006073 to 0.014904. This is a point-estimate influence diagnostic, not a replacement for the bootstrap interval.

## The visible baseline is materially stronger with agreement

Under the original log-support target and gradient boosting, the source-over-visible gain changes from 0.489933 [0.358859, 0.636402] to 0.055922 [0.018197, 0.096569] when agreement is supplied to all policies. The Qwen visible-only fresh R2 improves from -0.75255 to 0.61073. Granite improves from 0.22132 to 0.71066, and Phi from 0.01457 to 0.53774.

Thus, the original very large advantage over the text-based visible baseline is sensitive to the representation of an inexpensive visible cue. This does not establish a causal mechanism for the report scores: the added feature changes both the baseline and report-augmented fitted pipelines, and it does not manipulate the language model's answering process.

## Reporter identity and query costs

The average source-versus-selected-observer comparison remains unresolved in all four primary-decoder variants. With both bounded probability and agreement, the gain is 0.011421 [-0.011566, 0.029526]. Comparing probe-plus-source with probe-plus-observer gives 0.013973 [-0.002947, 0.030506]. These intervals establish neither equivalence nor a source-specific average advantage.

The source-alone versus probe-alone primary comparison remains unresolved as well: with bounded probability and agreement it is 0.011047 [-0.041803, 0.055536]. The added agreement feature changes no model-query counts: six probe calls, 24 report calls or thirty joint calls across three strengths. Model sizes, prompt lengths and CPU costs still differ.

## Recommended manuscript treatment

1. Keep the original frozen analyses intact and label these additions as post hoc sensitivity analyses.
2. Put the explicit agreement baseline in the main comparison rather than relying only on text forecasts. Present both original and stronger baselines for continuity.
3. Add the bounded-outcome result and target-specific heterogeneity. Avoid a general assertion that source reports robustly improve prediction across outcome choices.
4. Retain the conclusion that predictive usefulness and reporter identity are distinct. The tests continue to leave an average source-model advantage unresolved.
5. Publish a release containing R4 and this add-on before claiming public availability. The manuscript's GitHub URL must identify the actual published repository/release; the local package does not itself make those files public.
6. Restore a short provenance note explaining the missing original screening records and the limits of local freeze documentation.

Suggested results paragraph:

> In a post hoc sensitivity analysis, we added an explicit indicator of agreement between the recorded answer and the document's asserted entity to every prediction policy. This indicator uses already visible information and requires no additional model calls. Under the original log-support outcome, source reports retained a gradient-boosting increment beyond the direct probe of 0.0468, with a descriptive 95% whole-group interval [0.0024, 0.1004]. Replacing the outcome with a bounded answer-probability contrast while retaining the agreement feature reduced the primary-decoder increment to 0.0115 [-0.0067, 0.0312]. Ridge yielded a positive mean of 0.0292 [0.0034, 0.0611], concentrated in Phi. These checks qualify the magnitude and generality of report complementarity, while the average comparison with a selected external reporter remains unresolved.

This package supplies supplementary analyses. The Overleaf manuscript has not been edited.
