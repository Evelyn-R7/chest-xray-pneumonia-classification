# Experiment protocol

This project follows a leakage-aware protocol:

1. Audit original filenames and inferred patient IDs.
2. Remove development images whose inferred patient IDs overlap the official test set.
3. Create a v3_clean patient-level train/validation split.
4. Verify the TensorFlow data pipeline.
5. Train a CNN baseline.
6. Train VGG16 and EfficientNetB0 transfer-learning models.
7. Run multi-seed validation for the selected EfficientNetB0 control strategy.
8. Compare class weighting and focal loss as imbalance strategies.
9. Freeze the final ensemble, calibration method, and thresholds before final test.
10. Run the final test evaluation once.
11. Perform post-hoc exploratory error analysis and Grad-CAM.

The test set was not used for model selection, calibration selection, threshold selection, or imbalance-strategy selection. After the final test evaluation, the model, threshold, and calibration choices are frozen and must not be changed based on test results.
