# Final model selection protocol

Stage 7A freezes the final validation-selected protocol before any final test evaluation.

## Frozen model form

The final predictor is the EfficientNetB0 control three-seed equal-probability ensemble:

`ensemble_raw_probability = (p_seed_42 + p_seed_2025 + p_seed_2026) / 3`

Seeds are fixed to 42, 2025, and 2026. No single seed is selected as the final model because single-seed selection would overfit model choice to validation noise. Equal averaging keeps the protocol simple and pre-specified.

## Why control was selected

Stage 6 compared validation-only imbalance strategies at threshold 0.5:

| strategy | mean balanced accuracy | mean sensitivity | mean specificity | mean Brier |
| --- | ---: | ---: | ---: | ---: |
| control | 0.969168291 | 0.976608187 | 0.961728395 | 0.020623490 |
| class weight | 0.963434048 | 0.966374269 | 0.960493827 | 0.025693672 |
| focal | 0.963385315 | 0.955165692 | 0.971604938 | 0.034186917 |

Control had the highest mean balanced accuracy and best mean Brier score among the three strategies. Focal improved specificity but reduced sensitivity and calibration quality; class weighting did not improve the validation trade-off.

## Calibration methods

Three candidates were compared using validation-only out-of-fold predictions:

1. `none`: no calibration, raw ensemble probability is used.
2. `platt`: logistic regression on the raw ensemble probability.
3. `isotonic`: isotonic regression with `out_of_bounds="clip"`.

The comparison uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `patient_id`, so the same patient cannot appear in both calibration train and fold validation.

ECE is computed with 15 equal-width bins over [0, 1]. For every non-empty bin, the absolute difference between mean predicted probability and observed positive fraction is weighted by the bin sample fraction; ECE is the weighted sum.

## Calibration selection rule

The pre-registered rule is: choose the lowest OOF Brier score. If methods are within 0.001 Brier, choose the simpler method in this order: none, Platt, isotonic.

Selected method: `none`.

OOF summary:

| method | brier_score | log_loss | ece | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- |
| none | 0.0201471690675 | 0.0684387652967 | 0.0123020128761 | 0.996729478016 | 0.998750695753 |
| platt | 0.0209774346819 | 0.0894732444124 | 0.0297689492287 | 0.995641108945 | 0.9984186237 |
| isotonic | 0.0213084136431 | 0.108866296628 | 0.0161159579113 | 0.992200021659 | 0.995280808633 |

## Thresholds

Two thresholds are pre-registered:

- benchmark_threshold = 0.5
- balanced_threshold = 0.5618644666666667

The balanced threshold is selected on validation probabilities from the selected calibration method by maximizing balanced accuracy. Ties are resolved by higher sensitivity, then distance closer to 0.5, then smaller threshold.

## Protocol cautions

All results in this report are validation-only. Test data has not been loaded, predicted, or evaluated. The final test evaluation should be run only once after this protocol is frozen, and test results must not trigger retraining, re-selection of calibration, or threshold changes.

This model is not clinically deployable. The dataset is limited, patient IDs are inferred from filenames, and there is no external validation, subgroup analysis, prospective validation, or clinical utility analysis.
