# CSRA Study 2 pilot — exploratory version 2

This version removes the OSF/registration requirement for the existing-item exploratory pilot. It retains the original model, prompt, data and numerical-analysis settings. No new pilot reports were generated during preparation. Report results as an **existing-item exploratory pilot with an internally frozen plan**, without claiming preregistration or an independently verified timestamp.

## What to do now

Open the updated CSRA_Study2_Pilot.ipynb in Colab, choose a CUDA GPU runtime, and follow its numbered steps:

1. Upload the updated ZIP; it extracts into a separate version-specific source location.
2. Install pinned dependencies. The cell removes both unused torchvision and torchaudio, addressing the CUDA extension mismatches encountered in the original notebook. Fresh subprocesses show full errors and perform an early import/CUDA check. No restart is required between cleanup and those checks.
3. Mount Drive and select the new exploratory_v2_run1 folder. Keep the old run1 folder unchanged.
4. Run CPU synthetic tests.
5. Run all three technical preflights automatically. A passing preflight is reused only if it binds this exact new package/model; old-version preflight files are not relabeled. This step tests synthetic prompts only.
6. Save the internal freeze. No OSF account, registration form or receipt upload is needed.
7. Collect/resume Qwen, Granite and Phi sequentially. There are 28,800 logical report jobs, 9,600 per system. On interruption, restore the notebook variables and rerun this cell using the same source/run folder. Completed rows are skipped. One writer per model only.
8. Analyze after all three systems finish, using 10,000 paired bootstrap draws. No interim hypothesis analysis is allowed.
9. Optionally run the costly planned secondary predictive probes now; completion status is recorded. Complete these before reporting them as finished.
10. Download the results ZIP and retain it with the source ZIP. Return that result ZIP for scientific review and the next study decision.

The local freeze copies locked sources and records a local timestamp. Neither it nor these checks proves independent timing, absence of prior knowledge outside the folder, or external registration. Synthetic preflight is not a dependable full-study cost forecast. CUDA inference is not available in the preparation environment; the revised GPU workflow still needs to pass in Colab.

## Files and provenance

- protocol/EXPLORATORY_AMENDMENT.md governs the new workflow and scope.
- protocol/PROTOCOL.md and PILOT_IMPLEMENTATION_ADDENDUM.md retain the detailed estimands, exclusions and numerical choices.
- protocol/history/ preserves the superseded registration-focused drafts and original review PDF.
- protocol/scientific_compatibility.json checks unchanged scientific inputs, prompt construction and analysis calculations against the prior package.
- inputs/ holds 100 old item records and 900 frozen answer summaries; docs/INPUT_PROVENANCE.md records their source matches and limitations.
- code/ contains scoring, freeze, checkpoints and analysis; tests/ contains synthetic validation only.
- docs/VALIDATION_REPORT.md records what was tested and what remains untested.

No real or synthetic report outcomes are included as Study 2 findings. The original MIT license is retained. Existing item review and limited model/precision coverage remain scientific limitations.

## Local CLI

```bash
python code/run_pilot.py validate
python tests/selftest.py
python tests/full_validation.py
python code/run_pilot.py preflight --model qwen --output /durable/exploratory_v2_run1/qwen_preflight.json
python code/run_pilot.py freeze --output /durable/exploratory_v2_run1
python code/run_pilot.py run --mode exploratory --model qwen --output /durable/exploratory_v2_run1 --preflight /durable/exploratory_v2_run1/qwen_preflight.json
python code/analyze_pilot.py --run-dir /durable/exploratory_v2_run1 --out /durable/exploratory_v2_run1/primary_analysis
```

Run preflight and collection for all three systems. The separate registered CLI mode requires --mode registered and a real --registration receipt; the notebook does not use it. Source-lock rebuilding is only for reviewed changes before new collection, with prior sources and amendment history retained. Google Drive is not a transactional database: append/fsync and partial-tail recovery do not guarantee cloud durability or exactly-once inference. Keep backups.
