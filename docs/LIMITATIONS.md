# Limitations

1. Single public pediatric chest X-ray dataset.
2. No external validation.
3. Patient IDs are inferred from filenames rather than external metadata.
4. Limited information about acquisition devices, sites, and patient subgroups.
5. Validation and test distributions differ substantially.
6. Test specificity is much lower than validation specificity.
7. Probability calibration worsens on test.
8. High-confidence false positives are common.
9. No subgroup, fairness, or device-level analysis.
10. Grad-CAM does not prove lesion localization correctness.
11. Only three random seeds were used for the main multi-seed experiments.
12. The model is not suitable for clinical deployment.
