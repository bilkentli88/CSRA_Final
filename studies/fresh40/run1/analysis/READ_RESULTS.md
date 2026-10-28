# Fresh40 development qualification

Status: FRESH40_DEVELOPMENT_QUALIFICATION. Forty fresh items, 360 cases, forty clusters.

Positive values below favor the policy named first. Scaling uses the target model’s old100 outcome variance.

| Contrast | Normalized MSE reduction | Descriptive 95% interval | Positive targets | Practical screen |
|---|---:|---|---:|---|
| self_over_visible | 0.4899 | [0.3589, 0.6364] | 3/3 | True |
| self_over_selected_observer | -0.0006 | [-0.0515, 0.0503] | 1/3 | False |
| self_added_to_probe | 0.0719 | [0.0115, 0.1439] | 3/3 | True |
| self_over_probe_cost_comparison | 0.0113 | [-0.0740, 0.1101] | 2/3 | True |

0.01 normalized MSE is a proposed engineering screen, not a validated scientific minimum. At least two targets must have positive improvements. Intervals are descriptive; no multiplicity-adjusted confirmatory decision or publication guarantee.

Purposive relation-balanced convenience sample. After inspection these forty items are development data and must not enter later confirmatory tests.

SELF_PRESENTED and OTHER_PRESENTED are prompt labels. Self means same weights/configuration, not access to retained answer-generation activations.

Training-selected observers: {"granite": "phi", "phi": "qwen", "qwen": "phi"}

See qualification_summary.json for each target and Ridge sensitivity, per_case_predictions.csv for all errors, and actual_costs.json for measured successful forward costs. No predictor was fitted to Fresh40 outcomes.
