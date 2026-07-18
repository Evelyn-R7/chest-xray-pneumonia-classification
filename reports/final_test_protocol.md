# Final test evaluation protocol

Stage 7B defines the one-time final test evaluation framework. Codex created the framework but does not run real test inference in this stage.

The frozen protocol is `results/final_protocol/final_protocol.json` with SHA-256 `6835e0135b286046c37d2fa765aeaea2f564e3b8155e3012c81052088b75885e`.

The final model form is fixed: EfficientNetB0 control seeds 42, 2025, and 2026, combined by equal probability averaging. Calibration is `none`. The benchmark threshold is 0.5 and the primary balanced threshold is 0.5618644666666667.

Preflight may hash `test.csv`, load frozen model files, and run synthetic tensor predictions. It must not parse `test.csv` contents, read test images, create a test Dataset, or create one-time final-test markers.

The final test runner requires both the expected protocol hash and the exact confirmation string. It refuses to run if `TEST_EVALUATED` already exists or if `TEST_EVALUATION_STARTED` exists without `TEST_EVALUATED`.

After all preflight checks and immediately before reading test manifest contents or images, the runner atomically creates `results/final_test/TEST_EVALUATION_STARTED`. Any failure after that point requires manual audit; the code does not delete the lock or automatically retry.

The final evaluation uses the frozen models sequentially, one model in memory at a time. It uses batch size 16, deterministic order, no shuffle, no augmentation, and serial image reads. The three seed probabilities are averaged. No calibration model is fitted and no threshold is changed.

Primary reporting uses the balanced threshold. Benchmark reporting uses threshold 0.5. Threshold-independent metrics are ROC-AUC, PR-AUC, Brier score, log loss, and ECE with 15 equal-width bins.

Uncertainty is estimated with a patient-level stratified cluster bootstrap: 5000 replicates, seed 20260718, 95% percentile intervals. Patients are sampled with replacement within true-label strata, and all images for a sampled patient remain grouped.

Secondary patient-level analysis averages final probabilities within each inferred patient_id and evaluates the same two frozen thresholds. Patient IDs are inferred from filenames and are a limitation.

The final report must state that no retraining, recalibration, threshold optimization, or ensemble reweighting occurred after test evaluation. The model is not clinically deployable and is not a replacement for clinician judgment.
