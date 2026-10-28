# Development implementation decisions

Written before new development scoring. This addendum makes the candidate main design executable for the old-100 stage. It is not a preregistration, a record of completed independent review, a completed power analysis, or a fresh experiment.

## Scope and provenance

The exact old items file is inherited from the user's completed exploratory results package (SHA-256 recorded in protocol/development_config.json). Only item text and identifiers are reused. No NF4 targets or report scores are reused. The new experiment recomputes FP32 answer support using render-then-tokenize(add_special_tokens=False). The scored population remains an originally selected development sample, without claims of representativeness.

The parent plan proposes factual re-review before development. Because new review decisions have not been supplied, this implementation permits provisional old-data diagnostics while review is pending and explicitly prevents treating the resulting draft stack as a confirmatory release. This is an operational staging change, recorded before execution. Full factual re-review, resolution of factual problems and the resulting data/analysis freeze are required before a final confirmatory training artifact is released. Changing reviewed inputs requires a new package/run and a documented impact assessment; committed scores are not overwritten or relabeled as reviewed.

## Prompts and scoring

- Documents reproduce the existing skeptical, provisional and bare claim wording, with the subject and entity substituted. P and C documents at each strength differ only in the factual entity. The original-language bare claim uses lower-case 'the' after 'states that'.
- NO_DOC omits the document block. Answer options are P/C/I under all six A/B/C assignments. The prompt is fixed in common.answer_prompt. Tie-breaking after mean log support uses P, then C, then I.
- Three strengths provide Y = [mean logp(C)-mean logp(P)] under C-document minus the same difference under P-document. The baseline margin is no-document mean logp(P)-mean logp(C). Recorded answers come from the C condition's highest permutation-averaged support. I_PROVISIONAL and all other conditions are retained even when not in the primary target formula.
- Reports use the parent ownership cues and four cyclic document/prior/both/uncertain mappings, with certainty fixed and the recorded answer always shown. Every source/reporter pair receives both cues. Actual source names and true source/reporter-match indicators are absent from the report prompt.
- Report role log supports are averaged across rotations before softmax. Primary ALRs are document/prior, both/prior and uncertain/prior, computed directly as mean-log-support differences, without unstable probability ratios.
- All seven supplied hardware profiles are preserved. Development uses FP32 only. Mistral requires its qualified space-prefixed label IDs; the other systems use their qualified empty-prefix IDs. The model/tokenizer revisions, system message, precision, SDPA choice and TF32 setting are fixed.
- Before any scored development outcomes, all answer prompts and every possible report payload for P, C or I are audited under all four tokenizers. Complete prompt-plus-label tokenization must append exactly the qualified single token. No prompt may exceed 512 tokens; no encoder input may exceed 384. Overlength or failed-continuation cases stop before outcomes and require a documented revision/new freeze; no silent truncation occurs.

## Groups and predictive evaluation

Subjects and ordered normalized P/C/I triples define a connected-component grouping: any shared subject or triple keeps items in one group. The old 100 items produce 57 groups, with relation-specific triple counts developer 20, formation location 22, original language 4, original network 11. Group counts are data properties, not independent human-review counts. Shared answer entities across different triples remain a possible dependency that this grouping does not eliminate.

The deterministic outer five-fold assignment keeps complete groups together. Hyperparameter selection and stack-training text predictions use nested GroupKFold with up to three folds, within each outer training set only. TF-IDF vocabulary/weights, embedding-feature scaling, alpha selection and model fitting never use the outer held-out outcome. Frozen MPNet encoding has no supervised fitting on this study; the encoder's pinned attention-mask mean pooling and L2 normalization follow its model card. Its CPU embeddings are bound to the actual input text/order by a manifest.

Both text forecasters use alpha grid [.1,1,10,100], with a tie favoring the larger value. TF-IDF uses word unigrams/bigrams, min_df=1, at most 10,000 features and sublinear TF. Ridge uses LSQR tolerance 1e-8, maximum 5,000 iterations. Text includes public target identifier, question, document and recorded answer. MPNet embeds that same text. Final text forecasters are refit on all development data after selection, while decoder-training text predictions are generated out of group.

The common decoder receives baseline margin, fixed one-hot relation/strength/answer-role/known-target covariates, the two text-forecaster predictions and two 3-ALR report vectors. Unknown future target categories receive zeros in the known-target one-hot block; unseen words follow the fitted TF-IDF vocabulary. True source/reporter match and reporter identities are not decoder features. Features have training-only weighted standardization. Primary Ridge alpha=1 has a free intercept. Each item/target/strength has total training weight 1 across its views.

## Equal-cost comparison

The same ownership cue is applied to both reports in a pair. This explicitly resolves the parent design's unspecified pairing of cue levels: compare source correspondence while holding displayed ownership constant, then average equally across the two cue levels. Mixed-cue pairs are not part of the primary comparison.

For each target and cue, use all three independent anchors and both possible second independent reporters. Each comparison replaces that second observer with the source-corresponding reporter. The same decoder, feature count and two-report query budget apply to both arms. Both report-slot orders enter training with half weight; evaluation averages their predictions before squaring the residual. Squared errors are then averaged over anchors, second observers and cues. Errors from all models/strengths/cues/views belonging to an item remain linked. A source report reused across comparison views does not count as additional independent evidence or a new GPU call.

The development diagnostic normalizes paired error improvements by each known target's variance over its out-of-group development Y values. Those development variances are not reused as fresh-set primary denominators. Per-target R2 and equal-target mean delta-R2 are exploratory development diagnostics, not primary fresh evidence or a selection gate. All three known targets are retained regardless of direction.

The draft final training artifact is fit on all 100 old items, using group-held-out baseline predictions for decoder training and refitted text models for future prediction. Its manifest binds code, inputs, scores, encoder and analysis versions and marks confirmatory_release_ready=false. Mistral target outcomes never enter model fitting, tuning or calibration. The optional GradientBoosting decoder has 100 trees, learning rate .05, maximum depth 2, squared-error loss and seed 20261020; it cannot replace the Ridge primary based on observed effects.

## Sample-size scenarios

Calibration first averages each known target's normalized paired error differences across an item's strengths, then across items in a subject/triple group. P1 uses the equal-target mean within a group. For P2, the known target with the largest relation-balanced grouped residual variance is a conservative comparator within the observed cohort. It is only a proxy for an unseen target; it does not observe or bound Mistral's variance.

Within each relation, centered group residuals receive equal mass. Artificial future samples draw N/4 hypothetical independent fresh items from this empirical support. This simulates residual variability; it does not claim literal repeated old facts would be eligible fresh data. Each simulated dataset receives 1,999 relation-stratified bootstrap draws and a 97.5% percentile interval. There are 1,000 datasets for each N=400/600/800 and each base metric. Multinomial bin counts implement the same bootstrap efficiently. The variance multipliers 1/1.5/2 scale centered residuals by their square root; shifts 0/.01/.03 translate intervals. These scenarios can reuse draws exactly under this declared location-scale model.

Report median and 90th-percentile interval halfwidth, approximate Monte Carlo uncertainty for that quantile, empirical-location-model coverage and Wilson Monte Carlo intervals for coverage/positive-interval fractions. Provisional precision selection chooses the smallest N whose 90th-percentile halfwidth is at most .01 for P1 and all P2 variance scenarios. This criterion was specified before development outcomes. No sign, p-value or favorable learner is used for sample-size choice. If no N meets it, the report recommends no candidate. --smoke uses 20/199 for software checks only and cannot recommend N.

The four original-language groups make empirical support particularly thin. Population shift, development-selection bias, possible shared-answer dependencies, development-model fitting uncertainty and unseen-target behavior are not removed by a large simulation count. The calibration is an engineering planning exercise, not established power or an equivalence claim. Its candidate_N_by_precision_only is not a final-N freeze or permission to execute fresh items.

## Sources and remaining implementation

Model and encoder pins and primary references are inherited from docs/PARENT_DESIGN.md and the planning package. Frozen encoder usage: https://huggingface.co/sentence-transformers/all-mpnet-base-v2 . The previously checked model card documents attention-mask mean pooling, normalization and the default 384-word-piece text window.

This package completes the old-100 scoring, primary predictive development and precision-scenario implementation. It does not implement fresh collection, the main paired test-error bootstrap, final confirmatory release, or every secondary sensitivity in the parent design. Those remain prospective work after development results, factual review and resource decisions. External registration is optional; no receipt is fabricated.
