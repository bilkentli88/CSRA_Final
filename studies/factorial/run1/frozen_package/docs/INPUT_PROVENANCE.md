# Input provenance

Prepared from the user-supplied recovered repository already included in CSRA_Revision_R1/revision_evidence/source_repo. These are historical, previously used Study 1 records; no new answer or report scoring was performed to prepare these inputs.

`inputs/items.csv` preserves every field, value and row order from data/final_common_items.csv (100 items), with normalized text-file line endings. `inputs/answer_targets.csv` selects the three C_SKEPTICAL/C_PROVISIONAL/C_BARE rows per item from each model's answer_condition_summary.csv, retaining exactly its listed fields and values (900 rows). Both field-level matches were checked locally. CIE is the historical document-substitution effect, not a newly estimated target. All report prompts are newly constructed from these fixed summaries.

Original relative paths and SHA-256:

- `data/final_common_items.csv`: `19c57a4a96db4d8bbe2cb529ed94a74e156eb9f1f2c407660cf2530132c90b98`
- `data/supporting/qwen_answer_condition_summary.csv`: `2edaccf11c050571604e20ad0848dd55eda959c054880f18804f8175025202fd`
- `data/supporting/granite_answer_condition_summary.csv`: `7eff056cc12b281d953d12194d09ea8c3dd201c6bfbf495ae2572a8e517e6822`
- `data/supporting/phi_answer_condition_summary.csv`: `05712981f3202bcf865e84c52ff1ffd92951e16d9e2f9b925881ba437b5aad3b`

Original item metadata, including review_status=APPROVE, is preserved as historical data. It does not establish independent human review, reviewer identities or a factual verification completed during this revision. Those unresolved provenance limits remain. The source repository's MIT license is retained in LICENSE.
