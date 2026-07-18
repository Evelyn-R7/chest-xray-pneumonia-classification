# EfficientNetB0 transfer multiseed validation summary

All values are validation-only. The test split was not loaded, predicted, or evaluated.

## Manifest SHA-256

| manifest | sha256 |
| --- | --- |
| data/splits/v3_clean/train.csv | fac67671d85f11d66bfa87179fb1027cb51da59fa163936102282da31352f566 |
| data/splits/v3_clean/val.csv | ab7dcd0425ec69e379f248ffe1a4875d5f50534c75ff9d8860fd5a7ec3f696f0 |
| data/splits/v3_clean/test.csv | 6ba1a4093a601c590b433e99e8a332b2af3a5f46c29f062dd8e1d283f3a2622d |
| data/splits/v3_clean/excluded_from_development.csv | c9a8284ced63608ceeea95f757d7c8a3d2fdb77bf2e633bacae02d7aa36bd522 |

## Metrics by seed

| seed | best_phase | accuracy | precision | sensitivity | specificity | f1 | balanced_accuracy | roc_auc | pr_auc | npv | brier_score | tn | fp | fn | tp | val_loss | best_epoch | phase1_epochs | phase2_epochs | total_training_epochs | training_duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | phase2 | 0.9549266247379455 | 0.9878234398782344 | 0.9488304093567251 | 0.9703703703703703 | 0.9679343773303505 | 0.9596003898635477 | 0.9950996317955382 | 0.998133951369756 | 0.8821548821548821 | 0.037557737190794915 | 262 | 8 | 35 | 649 | 0.0096498187631368 | 20 | 5 | 20 | 25 | 4082.0 |
| 2025 | phase2 | 0.9570230607966457 | 0.989345509893455 | 0.9502923976608187 | 0.9740740740740741 | 0.9694258016405667 | 0.9621832358674465 | 0.9945798137318604 | 0.9979518443103949 | 0.8855218855218855 | 0.03822572549256156 | 263 | 7 | 34 | 650 | 0.0097850523889064 | 9 | 8 | 6 | 14 | 2335.0 |
| 2026 | phase2 | 0.9675052410901468 | 0.9880418535127056 | 0.966374269005848 | 0.9703703703703703 | 0.9770879526977088 | 0.9683723196881091 | 0.9967457223305176 | 0.9987460940007562 | 0.9192982456140351 | 0.026777287505884716 | 262 | 8 | 23 | 661 | 0.0078110424801707 | 24 | 8 | 20 | 28 | 4678.0 |

## Mean ± sample std

- accuracy: 0.959818 ± 0.006739
- precision: 0.988404 ± 0.000823
- sensitivity: 0.955166 ± 0.009734
- specificity: 0.971605 ± 0.002138
- f1: 0.971483 ± 0.004911
- balanced_accuracy: 0.963385 ± 0.004508
- roc_auc: 0.995475 ± 0.001131
- pr_auc: 0.998277 ± 0.000416
- npv: 0.895658 ± 0.020542
- brier_score: 0.034187 ± 0.006426
- val_loss: 0.009082 ± 0.001103
- best_epoch: 17.666667 ± 7.767453
- phase1_epochs: 7.000000 ± 1.732051
- phase2_epochs: 15.333333 ± 8.082904
- total_training_epochs: 22.333333 ± 7.371115
- training_duration: 3698.333333 ± 1217.707819
- tn: 262.333333 ± 0.577350
- fp: 7.666667 ± 0.577350
- fn: 30.666667 ± 6.658328
- tp: 653.333333 ± 6.658328

## Stability checks

- Most stable metric: pr_auc
- Most variable metric: npv
- Samples incorrect for all three seeds: 27
- Samples with prediction label disagreement: 23
- Samples with probability sample std >= 0.10: 14
- Degenerate all-positive or all-negative runs: none
- NaN, Inf, OOM, or failed registered runs: none detected by aggregation checks
- Threshold remained fixed at 0.5; no threshold tuning was performed

## Descriptive comparison to CNN baseline

This report is descriptive and does not claim statistical superiority without formal testing.
CNN baseline numbers are copied from `reports/cnn_baseline_multiseed_results.md` for validation-only context.

| metric | CNN baseline mean ± std | EfficientNetB0 mean ± std |
| --- | --- | --- |
| accuracy | 0.892383 ± 0.019815 | 0.959818 ± 0.006739 |
| sensitivity | 0.952242 ± 0.060797 | 0.955166 ± 0.009734 |
| specificity | 0.740741 ± 0.213534 | 0.971605 ± 0.002138 |
| f1 | 0.927242 ± 0.009169 | 0.971483 ± 0.004911 |
| roc_auc | 0.974347 ± 0.006318 | 0.995475 ± 0.001131 |
| pr_auc | 0.989708 ± 0.002901 | 0.998277 ± 0.000416 |
| brier_score | 0.080924 ± 0.015574 | 0.034187 ± 0.006426 |

## Limitations

Only three seeds are summarized. Results are validation-only and do not establish final test or clinical performance.
No external validation, confidence intervals, subgroup analysis, calibration study, or clinical utility analysis is included.
