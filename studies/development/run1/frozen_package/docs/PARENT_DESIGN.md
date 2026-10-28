# A stronger main study for the rejected CSRA article

Planning version, 20 October 2026. No new scientific outcomes have been collected. This document proposes the next study; it is not a completed experiment, a power guarantee, or a promise of journal acceptance. The earlier pilot and its frozen record remain unchanged.

## The contribution to pursue

**Research question:** Do source reports carry model-specific information about a measured document-substitution effect, beyond what independent observer reports and the visible input/answer provide, and does that information transfer to fresh facts and an unseen target model?

The proposed advance is a crossed test separating **actual answer-source/reporter identity** from **displayed ownership wording**, with an equal-query-cost observer comparison, an independently reviewed evaluation set, unquantized inference, and a held-out target model. The primary comparison replaces a second independent observer with the source-corresponding reporter, using the same frozen decoder. It therefore asks whether source correspondence is useful beyond simply buying another report.

A positive result would establish source-specific predictive usefulness under the tested conditions. A precisely bounded negative result would establish limits of that usefulness under the defined prediction task. A wide interval remains inconclusive. Broad ambition comes from stronger controls and out-of-sample evidence; the outcome is not stipulated in advance.

Suggested working title: **Model-Specific Source Attribution Under Knowledge Conflict: Ownership Cues, Observer Controls, and Transfer**.

## Why this is more than the pilot

The 100-item pilot established that report categories respond to answer visibility and wording. It did not include independent cross-model observers, fresh evaluation items, a new target model, or an unquantized reference. Those limitations are addressed directly here. The previous main predictive visibility contrast remains inconclusive and must not be rewritten as a demonstrated absence of information.

The closest work already evaluates explanations through counterfactual prediction and compares self- with cross-model explanations. Merely adding an observer or more models would not establish novelty. The specific proposed distinction is the orthogonal identity/ownership comparison on a continuous document-substitution target, without requiring observers to independently choose the target's answer, together with a frozen unseen-target evaluation and an equal-cost second-observer control. See LITERATURE_AND_GAP.md. This is a reasoned candidate contribution from a targeted search, not a claim of firstness.

## 1. Systems, precision and target holdout

| System | Role | Main inference | Development target outcomes? |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | Known target and independent reporter | Unquantized FP32 | Yes |
| Granite-3.3-2B-Instruct | Known target and independent reporter | Unquantized FP32 | Yes |
| Phi-4-mini-instruct | Known target and independent reporter | Unquantized FP32 | Yes |
| Mistral-7B-Instruct-v0.3 | Larger, fourth-family target and reporter | Unquantized FP32 | **No** |

Exact candidate model/tokenizer revisions are in study_config.json. The original three revisions are retained; Mistral's public model revision was checked from its primary Hugging Face record. Mistral may report as an observer on development targets, but no Mistral development answer-substitution targets are generated or used for fitting, tuning, or sample-size calibration. Its factual-target evaluation remains reserved for the fresh set. This is a prospective held-out **target** test, not a claim that its parameters, public information, or observer outputs are unseen.

FP32 means FP32 inference from the released checkpoint without additional four-bit quantization; casting released BF16 weights cannot recover information already absent from that checkpoint. Use FP32 computation, disabled TF32, evaluation/inference mode, no CPU/disk offload, and a fixed SDPA implementation. A preselected 100-item fresh subset also receives NF4 source-corresponding reports for the original three architectures, with FP32 compute/double quantization. The target effects and recorded answers stay those of the FP32 main run. This isolates **reporter-precision sensitivity**; it is not a factorial experiment on source-answer quantization.

Four fixed systems support comparisons across those systems and a prospective target holdout. They are not a random sample of all language-model families. No synthetic continuation, replayed answer, or different reporter by itself identifies privileged access to the earlier computation. That would require a separately identified access/internal-intervention experiment; it is not smuggled into this behavioral design.

## 2. Items and sampling

Use the old 100 items for development only, after factual re-review. Budget the fresh evaluation at **600 new items**, 150 per original relation: developer, formation location, original language, and original network. The final N is selected before fresh model outcomes, using the precision and resource gate below; 400/600/800 are candidate counts, not interchangeable post-outcome options.

Start from a planning pool of 900 fresh candidates. This assumes roughly two-thirds can survive review and fill relation quotas; actual yield is unknown. Additional candidates can be built before model outcomes if a quota is short. Require two independent factual reviews blind to model outcomes, with all disagreements and exclusions recorded. Existing APPROVE fields do not substitute for this new review. Identify actual reviewers; never label AI-only checks as independent human review.

Reject ambiguous subjects, multiple defensible answers, wrong semantic types, unsupported sources, aliases that collapse alternatives, and unresolvable factual errors before inference. Require unique subject identities and unique normalized ordered P/C/I triples, disjoint from development subjects/triples. Shared answer entities can still induce dependence; unique triples do not prove independence. Select eligible items by a frozen seed-hash rule within relation. The 100-item reporter-precision subset uses the first 25 selected fresh items per relation under a separately fixed hash key, before outcomes.

**Important change:** P is the verified factual reference answer, not a claim that every model prefers it. Do not screen fresh items for all-model preference for P. Keep correct/incorrect/unfamiliar prior states in the main population and record no-document margins. A predefined subset where a target prefers P is a secondary knowledge-conflict analysis, without replacing the main population. No additional GPU eligibility-screen jobs are assumed in the budget.

The annotation packet contains blank schemas, not invented or completed reviews. Sources, retrieval dates, snapshot hashes, human roles and AI assistance must be truthfully documented.

## 3. Influence target and answer stage

For each target model and item, score eight conditions: NO_DOC, P_SKEPTICAL, P_PROVISIONAL, P_BARE, C_SKEPTICAL, C_PROVISIONAL, C_BARE, I_PROVISIONAL. Use all six A/B/C assignments for P/C/I. Fix wording before fresh execution. Within each strength, C and P documents differ only in the substituted factual entity.

Let L be permutation-averaged log support for C minus P. The behavioral target is Y = L(C-document) − L(P-document), separately for each of three strengths. The recorded answer in a C condition is the role with the largest permutation-averaged support. Preserve all scores, not only the chosen label. This measures a document-substitution effect on answer support, not an internal causal-state ground truth.

Recompute the answer stage for unquantized development/evaluation systems. Old NF4 answer targets cannot be combined with new FP32 source reports as if they were one precision-matched experiment.

## 4. Actual identity crossed with displayed ownership

For every target's recorded C-condition answer, ask every reporter for a source report:

| Actual target \ reporter | Qwen | Granite | Phi | Mistral |
|---|---|---|---|---|
| Qwen | Source-corresponding | Independent | Independent | Independent |
| Granite | Independent | Source-corresponding | Independent | Independent |
| Phi | Independent | Independent | Source-corresponding | Independent |
| Mistral — fresh only | Independent | Independent | Independent | Source-corresponding |

Every cell has **both** ownership cues: SELF_PRESENTED and OTHER_PRESENTED. The cues are experimental labels, independent of the true source/reporter match. They must be described as presented ownership, including deliberately mismatched labels, not as verified conversational history. Actual source names/identities are not supplied in the report prompt. The recorded answer, document, question, role definitions, certainty wording and visibility are identical for paired reporter comparisons. The answer is always shown; no hedge is added. This prevents actual correspondence from being confounded with a SELF versus OTHER stem.

Report roles remain document/prior/both/uncertain, scored under four cyclic A-D mappings. Average role log supports first, then four-role softmax. Use the three additive log ratios relative to prior as the primary predictive representation. Exact draft prompt fixtures are in code/design_tools.py. All reports are stateless fresh conversations. Independent observers receive the **actual target's recorded answer**, even if they would choose another answer themselves; there is no answer-agreement eligibility filter.

For fixed target payload and cue, reporters receive the same substantive user text. Native chat templates differ by model and must be qualified and recorded. Use render-then-tokenize(add_special_tokens=False) in the new study to avoid duplicate template special tokens. Qualify complete prompt+label continuations, pin the common valid label prefix/IDs before data, and stop if qualification fails. Bound actual prompts to 512 tokens under every relevant tokenizer before outcome collection; do not silently truncate them.

## 5. Frozen input and observer baselines

B contains no-document margin, relation, document strength, recorded answer role, and input-text prediction scores. Fit two text-only forecasters on development outcomes: TF-IDF+Ridge and frozen all-mpnet-base-v2 embeddings+Ridge. They see the question, document, recorded answer and public target identifier; they do not see report outputs or target-internal activations. The encoder revision is pinned. Its input must remain within the verified 384-token bound; no unlogged truncation is allowed.

Generate out-of-fold development predictions for stacking, holding entire subject/answer-triple groups together. Fit text preprocessing, imputation (if needed), vocabularies, scaling and hyperparameters only inside development folds. Use a small fixed alpha grid [0.1, 1, 10, 100] for text forecasters; choose by development group-CV error with a fixed tie rule favoring larger alpha. Refit on all development data after selection. Retain both forecaster predictions rather than choosing one by fresh-set performance. Unknown target identifiers use a frozen unknown-category rule.

Train a **common prediction decoder** from B and two three-ALR report vectors. Use development data from the three known targets only. Balance native-second-report and independent-second-report cases; use all ordered independent-observer anchors and second observers and both cue levels. Add both report-slot orders with half weight and average their two predictions at evaluation, avoiding an arbitrary first/second-slot advantage. Do not give the decoder the true source/reporter-match indicator. Public target metadata may enter B; reporter-specific identity features are excluded in the primary common decoder, with an identity-aware decoder as a predefined sensitivity. Thus the main comparison uses exactly the same decoder, feature dimension, visible evidence, number of reports and fitted weights for both report choices.

The primary decoder is Ridge alpha=1, free intercept, training-only numeric standardization. Weight views so every development item/target/strength has total weight 1, divided across cues, observer choices, arm and slot order. Fit and freeze it on development data, then hash the full stack before fresh outcomes. A fixed GradientBoosting decoder (100 trees, learning rate .05, depth 2, squared-error loss, seed 20261020) is secondary, without a fresh-data tuning search. Do not use Mistral target outcomes for any fitting or selection. Input-only baselines are finite comparators, not an upper bound on what every possible observer could infer.

## 6. The two primary estimands

For target s and an independent anchor reporter r1, compare:

- Source-corresponding second report: f(B, report_r1, report_s).
- Second independent observer: f(B, report_r1, report_r2), with r2 different from s and r1.

Both use two reports and the same frozen decoder. Average the comparison over all valid ordered independent-observer choices and cue levels, rather than choosing a favorable observer. Evaluate the paired squared-error improvement, normalized by each target's original fresh-set population variance of Y, held fixed during bootstrap.

**P1 — new-item advantage:** equal-target mean R2(source-corresponding second report) − R2(second independent observer), for the three development-known targets on fresh items.

**P2 — unseen-target advantage:** the identical contrast for the reserved Mistral target on fresh items, with every predictor frozen using only known-target development outcomes.

A positive contrast favors a source-corresponding report over an equal-cost extra independent observer under a transferable common decoder. The two primary 97.5% two-sided paired percentile intervals use a Bonferroni family of two, 10,000 relation-stratified fresh-item bootstrap draws, seed 20261020. All models, strengths, cues and report comparisons of an item travel together. The sample is subject/triple-disjoint by construction. These intervals are conditional on the frozen development-trained stack; they do not include uncertainty from reselecting the development set or tuning pipeline. A secondary development-resampling robustness check can quantify that additional uncertainty if feasible, without replacing the primary test.

Because no estimator is refitted on fresh outcomes in the primary evaluation, each bootstrap draw only resamples paired test errors. This avoids millions of fresh-data training refits and preserves a clean prospective holdout. The 100-item development set limits training and remains part of the study's actual design, not an untouched replication sample.

Secondary analyses: one-report source-versus-observer comparison; incremental native-report R2 after one observer; category distributions and cue effects; identity-aware decoding; answer-omitted B; VSA representation; nonlinear decoding; per-target estimates; and fixed-target reporter-precision differences. They are not promoted into the primary family after inspection. A common-decoder null does not rule out information decodable by every possible learner.

## 7. Precision, exclusions and decision rules

600 fresh items is the reference budget, not established power for these new estimands. The old pilot's visibility contrast is a different statistic and cannot supply the new test's power. Before fresh outcomes, use group-held-out development predictions to estimate paired-error variability, and simulate candidate N=400/600/800 under null, small and positive contrasts with variance multipliers 1/1.5/2 for the unseen target. Use at least 1,000 simulation datasets, 1,999 bootstrap draws for calibration and Monte Carlo uncertainty. Freeze the smallest feasible N meeting a provisional interval-halfwidth goal of .01 under the declared variance scenarios. The goal is an engineering resolution, not a validated universal utility or equivalence margin. A substantive equivalence claim needs its own justified margins before fresh execution.

Sample-size selection uses attainable precision and resources, never the sign or significance of development effects. If no candidate fits, revise the design before collecting fresh outcomes rather than running an under-resolved study while promising a decisive result. No sequential significance stopping is planned. Report precision, direction and model heterogeneity for all outcomes. A wide interval is inconclusive; positive statistical significance alone is not a utility claim.

Technical recovery retains exact job/prompt/model/configuration identities, append-only committed batches, explicit OOM batch fallback 8/4/2/1 and no overwrite of completed scores. Stop on nonfinite scores or semantic changes. Retry only operationally equivalent jobs. Exclude an item jointly from the main paired evaluation only for unresolved required main cells; report every reason, no replacements, and flag >5% attrition as incomplete. Secondary precision-bridge failures do not silently change the primary item set; report bridge completeness separately. Fix target normalization on the achieved common main complete-case set. Degenerate target variance is a declared non-estimability, not grounds to replace a target after seeing outcomes.

A useful paper could show robust source-specific gains, robust equal-cost observer matching with sufficiently tight bounds, or a reproducible separation between source identity and ownership-cue effects. It need not tell a uniformly positive story. The design is intended to resolve a substantive alternative, not make acceptance conditional on a favorable pilot result.

## 8. Reference compute and annotation budget

| Component | Logical scoring jobs |
|---|---:|
| Old-100 development answer stage: 3 known targets × 8 × 6 | 14,400 |
| Development reports: 3 targets × 4 reporters × 2 cues × 3 strengths × 4 rotations × 100 | 28,800 |
| Fresh answer stage: 4 targets × 8 × 6 × 600 | 115,200 |
| Fresh reports: 4 targets × 4 reporters × 2 cues × 3 strengths × 4 rotations × 600 | 230,400 |
| Fixed-target NF4 reporter bridge: 3 reporters × 2 cues × 3 strengths × 4 rotations × 100 | 7,200 |
| **Total reference count** | **396,000** |

These are logical jobs, not batch forwards or generated-token counts. Identical rendered prompts can be cached only under a predeclared hash key including model revision, precision, tokenizer/template, label continuation and software semantics, with cache provenance retained. Do not count a cache hit as a new model evaluation. The conservative budget assumes no such savings. Qualification, retries, model loading, CPU embeddings, review and analysis are extra.

Uniform-throughput illustrations: 396,000 jobs take 13.75 GPU-hours at 8 jobs/s, 27.5 at 4, 55 at 2, or 110 at 1. These are scenarios, not forecasts from the earlier NF4 pilot. The original pilot's roughly 71-minute report collection does not establish rates for these FP32 models or the answer phase. Use measured per-configuration, per-phase profiles before a full launch. The supplied synthetic qualification notebook produces those profiles; update with actual prompt-length audits before final budgeting. No money price is assumed; enter your actual GPU rate/credit consumption separately.

Approximate FP32 weight storage alone is 14.9 GiB for 4B parameters, 7.5 for 2B, 14.2 for 3.8B, and 27.0 for 7.248B. Activations, CUDA context and workspace add more. Mistral FP32 requires a larger-memory GPU than the pilot's L4; qualify it on an A100-class runtime with adequate free memory. Do not silently substitute quantization or BF16 if it fails. The other models also require measured memory checks and may need smaller batches.

Annotation is potentially the larger bottleneck. With 900 fresh candidates plus 100 development re-reviews, two reviewers, five minutes per review, and 20% adjudication at six minutes, the reference is about **187 person-hours**. Illustrative fast/central/slow totals are approximately 107/187/317 person-hours (3/5/8 minutes per independent review, disagreement 10/20/30%, adjudication 4/6/10 minutes). These are assumptions, not observed staffing productivity. Timing a 20-item review pilot comes before committing to that workload. An existing validated benchmark can reduce construction effort, but does not automatically validate new distractors or erase a factual audit requirement.

## 9. The immediate next actions

1. Run the supplied **synthetic hardware qualification only**, starting with Granite FP32. Obtain real throughput and memory figures on the available GPU; qualify Mistral on adequate-memory hardware. No scientific item file is accepted by that script.
2. Identify the two reviewers and time a 20-item outcome-blind factual review using the blank forms. Record actual minutes and disagreement. No invented staffing, provenance or reviews.
3. If those budgets are feasible, implement the crossed full runner and development prediction stack, run the old-100 calibration, and finalize N, templates, label scoring, data selection and analysis hashes before fresh outcomes. The supplied package is a design/qualification package, not a finished main-study execution runner.
4. Complete candidate validation and freeze the selected fresh set, precision subset and target holdout. An internal freeze is sufficient for the chosen workflow; external registration remains optional and provides a different timing record. No fake receipt or retrospective preregistration claim is needed.
5. Run the fresh study and analyze the two frozen primary contrasts before deciding how strong the article's conclusions can be. The new evidence, not revised rhetoric, determines the final contribution.
