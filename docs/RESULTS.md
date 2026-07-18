# Results

All results are research results from a single public dataset. They are not clinical-performance claims.

## Validation: CNN baseline, three seeds

| metric | mean ± sample std |
| --- | ---: |
| accuracy | 0.8924 ± 0.0198 |
| ROC-AUC | 0.9743 ± 0.0063 |
| PR-AUC | 0.9897 ± 0.0029 |

## Validation: transfer learning

| model | seed(s) | accuracy | sensitivity | specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| VGG16 | 42 | 0.9612 | 0.9503 | 0.9889 | 0.9959 |
| EfficientNetB0 control | 42/2025/2026 | 0.972397 ± 0.001601 | 0.976608 ± 0.001462 | 0.961728 ± 0.004277 | 0.996569 ± 0.000433 |

EfficientNetB0 control also achieved validation balanced accuracy 0.969168 ± 0.002260, PR-AUC 0.998689 ± 0.000170, and Brier score 0.020623 ± 0.001454.

## Validation: imbalance strategies

The control strategy was selected. Class weighting did not improve mean balanced accuracy. Focal loss increased specificity but reduced sensitivity, F1, and calibration quality.

## Frozen validation ensemble

At threshold 0.5, the three-seed EfficientNetB0 control ensemble achieved validation accuracy 0.9738, balanced accuracy 0.9716, and ROC-AUC 0.9967. The selected calibration method was none. The primary balanced threshold was fixed at 0.5618644667.

## Frozen official test

| operating point | accuracy | sensitivity | specificity | balanced accuracy | ROC-AUC | PR-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benchmark threshold 0.5 | 0.8574 | 0.9974 | 0.6239 | 0.8107 | 0.9751 | 0.9829 | 0.1197 | 0.1438 |
| primary balanced threshold 0.5618644667 | 0.8670 | 0.9974 | 0.6496 | 0.8235 | 0.9751 | 0.9829 | 0.1197 | 0.1438 |

Patient-cluster bootstrap 95% intervals are reported in `reports/final_test_results.md`; they are copied from the frozen final-test report and are not recomputed here.

## Validation-test gap

The final test set preserved high sensitivity but showed substantially lower specificity and worse calibration than validation. This gap is emphasized as a key limitation.

## Post-hoc exploratory analysis

At the primary threshold: TN/FP/FN/TP = 152/82/1/389. There were 53 high-confidence false positives and 13 samples with three-seed label disagreement. The unique false negative was `person154_bacteria_728.jpeg` with final probability 0.1944. Grad-CAM is exploratory and explains only the seed 42 single model, not the ensemble.
