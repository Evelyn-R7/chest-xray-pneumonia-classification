# Model card

## Model details

- Model name: EfficientNetB0 control three-seed ensemble
- Version: 0.1.0
- Architecture: EfficientNetB0 transfer learning with a binary classification head
- Ensemble: equal probability average across seeds 42, 2025, and 2026
- Calibration: none
- Thresholds: benchmark 0.5; primary balanced threshold 0.5618644667

## Intended use

Research and education about leakage-aware medical-imaging model evaluation.

## Out-of-scope use

This model must not be used for patient diagnosis, triage, clinical decision-making, or as a substitute for clinicians.

## Data

Training and validation use the v3_clean development split derived from the Kaggle chest X-ray pneumonia dataset. Final evaluation uses the official test split. Patient IDs are inferred from filenames.

## Evaluation

The final official test at the primary threshold achieved accuracy 0.8670, sensitivity 0.9974, specificity 0.6496, balanced accuracy 0.8235, ROC-AUC 0.9751, PR-AUC 0.9829, and Brier score 0.1197.

## Known failure modes

The final test showed many false positives, including 53 high-confidence false positives. Calibration degraded on test relative to validation. No external validation, subgroup analysis, fairness analysis, or device analysis is included.

## Explainability

Grad-CAM visualizations are post-hoc exploratory explanations for the seed 42 model only. They do not prove medical causality or lesion localization correctness.

## Weights

Model weights are not distributed in this repository.

## License and clinical safety

The original code and documentation in this repository use the MIT License. Dataset images, ImageNet pretrained weights, and third-party dependencies remain under their respective licenses. The MIT License is not a clinical-use permission and does not imply medical-device clearance or certification.
