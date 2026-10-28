# Reproducibility guide

## Scope

These materials reproduce the **CSRA v0.8 R4/A2** analyses reported in the submitted manuscript. Earlier CSRA development/reserve experiments are deliberately excluded from the confirmatory reproduction path.

## Recommended environment

The frozen analysis environment used:

- NumPy 2.0.2
- pandas 2.2.2
- SciPy 1.16.3
- scikit-learn 1.6.1
- statsmodels 0.14.6

For statistical reproduction from the released summaries, a GPU is not required.

## Default quick verification

```bash
python run_reproduction.py
```

This recomputes the point estimates and robustness summaries that can be reproduced quickly on CPU. It reconstructs the frozen fold assignments directly from the deterministic hash rules in the frozen analysis specification and the released 100-item common set.

## Full primary bootstrap

```bash
python run_reproduction.py --bootstrap-primary
```

This reruns the 2,000-replicate paired common-item bootstrap for the prospectively primary predictive estimand, including preprocessing and Ridge refits in every replicate.

## Full alignment bootstrap

```bash
python run_reproduction.py --bootstrap-alignment
```

This reruns the 2,000-replicate paired common-item alignment bootstrap. It is computationally heavier because mixed-effects models are refitted in each replicate.

## TR4 compatibility repair

The frozen `code/csra_v08_analysis.py` contains the pre-execution analysis implementation. The finalized run used the documented TR4 compatibility repair when constructing nuisance design rows for the interaction contrast because the final selected set contained no `country_of_origin` items. The repair substitutes an observed nuisance relation level only for contrast construction. Since the nuisance relation enters additively, it cancels from the CIE-by-SELF contrast. The reproduction wrapper applies exactly this compatibility repair without changing the frozen formula or data.

## Expected outputs

The sealed final summaries are under `expected_results/`. The default reproduction writes new outputs under `reproduced_results/` and compares them to the sealed files using tight numerical tolerances.
