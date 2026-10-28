# Collecting a separate model-scoring run

The CPU paper route reproduces reported analyses from supplied scores. New inference is a separate run: it downloads the pinned models/tokenizers, requires compatible CUDA hardware and produces a new ledger. Do not change the supplied run1 folders or rebuild their historical freezes. Rerunning an already inspected sample does not produce prospective confirmatory evidence.

## Environment and model identities

Each study's `frozen_package/requirements-scoring.txt` records the scoring packages. The frozen execution records identify PyTorch 2.11.0+cu128, Transformers 5.13.1, bitsandbytes 0.50.0 and Accelerate 1.14.0. Use the actual compatible CUDA runtime and follow the package's startup/token audits. The notebooks include the original Colab setup and environment checks. Model and tokenizer revisions are fixed in each package's protocol JSON; the sentence encoder is separately pinned in the development/Fresh40 encoder records.

The reported modes are NF4 with FP32 compute for the factorial reports, and unquantized FP32 for development/Fresh40. The qualified development/Fresh40 hardware records describe A100 execution. These are execution records, not a guarantee that another GPU/software stack reproduces bit-identical supports. Preserve each new run's manifests and report differences; do not silently change precision or token handling.

Examples below use a Linux shell from the repository root. Set an absolute output directory **outside this repository** and use a new directory for each study. They illustrate the included CLIs; this R6 packaging run did not execute new GPU inference. Install the appropriate scoring/analysis dependencies before each workflow. Do not launch concurrent writers for one output directory.

## Factorial reports: 28,800 jobs

```bash
CSRA_FACTORIAL_RUN=/absolute/new_runs/factorial
CSRA_FACTORIAL_CODE=studies/factorial/run1/frozen_package/code
python "$CSRA_FACTORIAL_CODE/run_pilot.py" validate
for model in qwen granite phi; do
  python "$CSRA_FACTORIAL_CODE/run_pilot.py" preflight --model "$model" --output "$CSRA_FACTORIAL_RUN/${model}_preflight.json"
done
python "$CSRA_FACTORIAL_CODE/run_pilot.py" freeze --output "$CSRA_FACTORIAL_RUN"
for model in qwen granite phi; do
  python "$CSRA_FACTORIAL_CODE/run_pilot.py" run --mode exploratory --model "$model" --output "$CSRA_FACTORIAL_RUN" --preflight "$CSRA_FACTORIAL_RUN/${model}_preflight.json"
done
python "$CSRA_FACTORIAL_CODE/analyze_pilot.py" --run-dir "$CSRA_FACTORIAL_RUN" --out "$CSRA_FACTORIAL_RUN/primary_analysis"
```

The package README and protocol describe the exact eight reporting cells, item groups and primary analysis. The unfinished nonlinear bootstrap remains outside the article evidence.

## Full-precision development: 43,200 jobs

```bash
CSRA_DEV_RUN=/absolute/new_runs/development
python studies/development/run1/frozen_package/code/run_development.py prepare --run-dir "$CSRA_DEV_RUN"
CSRA_DEV_CODE="$CSRA_DEV_RUN/frozen_package/code"
python "$CSRA_DEV_CODE/run_development.py" audit --run-dir "$CSRA_DEV_RUN"
for model in qwen granite phi; do
  python "$CSRA_DEV_CODE/run_development.py" answers --run-dir "$CSRA_DEV_RUN" --model "$model"
done
python "$CSRA_DEV_CODE/run_development.py" targets --run-dir "$CSRA_DEV_RUN"
for model in qwen granite phi mistral; do
  python "$CSRA_DEV_CODE/run_development.py" reports --run-dir "$CSRA_DEV_RUN" --model "$model"
done
python "$CSRA_DEV_CODE/run_development.py" assemble --run-dir "$CSRA_DEV_RUN"
python "$CSRA_DEV_CODE/embed_inputs.py" --run-dir "$CSRA_DEV_RUN"
python "$CSRA_DEV_CODE/analyze_development.py" --run-dir "$CSRA_DEV_RUN" --decoder ridge
python "$CSRA_DEV_CODE/analyze_development.py" --run-dir "$CSRA_DEV_RUN" --decoder gradient_boosting
```

Mistral is a reporter, not an answering target. The parent design contains unexecuted larger-study plans; these commands concern the completed old100 design only.

## Fresh40: 17,280 jobs

The supplied Fresh40 frozen package includes the original old100 training inputs. A separate rerun using it repeats that specified training/calibration procedure. Replacing those inputs with a new development run would define a different training pipeline and requires its own reviewed package and provenance.

```bash
CSRA_FRESH_RUN=/absolute/new_runs/fresh40
python studies/fresh40/run1/frozen_package/code/run_fresh40.py init --run-dir "$CSRA_FRESH_RUN"
CSRA_FRESH_CODE="$CSRA_FRESH_RUN/frozen_package/code"
python "$CSRA_FRESH_CODE/run_fresh40.py" prepare --run-dir "$CSRA_FRESH_RUN"
for model in qwen granite phi; do
  python "$CSRA_FRESH_CODE/run_fresh40.py" answers --run-dir "$CSRA_FRESH_RUN" --model "$model"
done
python "$CSRA_FRESH_CODE/run_fresh40.py" targets --run-dir "$CSRA_FRESH_RUN"
for model in qwen granite phi mistral; do
  python "$CSRA_FRESH_CODE/run_fresh40.py" reports --run-dir "$CSRA_FRESH_RUN" --model "$model"
done
python "$CSRA_FRESH_CODE/run_fresh40.py" analyze --run-dir "$CSRA_FRESH_RUN"
```

The prepare stage audits tokens, encodes inputs, fits old100 predictors and freezes them before new outcome scores. Keep its refusal to train over existing fresh outcomes. The original fitting, embeddings and hardware can affect a new run; the CPU `fresh40` command instead replays the exact supplied fitted assets.

## Original NF4 measurement study

The original scoring implementation and released score summaries are included. The complete 600-candidate snapshot, screening/activation records and original individual scoring checkpoints are not. This repository cannot offer an exact raw-inference reconstruction of that historical collection. The supported original-study reproduction begins with the released summaries and deterministic analysis specifications, as described in the main reproduction guide.
