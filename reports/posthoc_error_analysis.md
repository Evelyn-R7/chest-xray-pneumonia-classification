# Post-hoc exploratory final-test error analysis

This analysis is post-hoc exploratory. It was performed after the frozen final test evaluation and must not be used to modify the model, calibration, thresholds, or ensemble strategy.

Primary threshold: `0.5618644666666667`. No new threshold was selected.

## Confusion groups

| error_group | count | patient_count | probability_mean | probability_median | probability_q1 | probability_q3 | std_probability_mean | std_probability_median | seed_label_disagreement_count | high_confidence_error_count | near_threshold_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FN | 1 | 1 | 0.194401326279 | 0.194401326279 | 0.194401326279 | 0.194401326279 | 0.0843091099759 | 0.0843091099759 | 0 | 0 | 0 |
| FP | 82 | 78 | 0.901565774548 | 0.93672122558 | 0.853080521027 | 0.987618391713 | 0.0300923101222 | 0.0188482573929 | 2 | 53 | 2 |
| TN | 152 | 149 | 0.124458696588 | 0.0472736302763 | 0.00997907218213 | 0.167440400148 | 0.0343135668339 | 0.0125235167801 | 11 | 0 | 6 |
| TP | 389 | 201 | 0.994143103419 | 0.999945600828 | 0.99939819177 | 0.999994675318 | 0.00292238439101 | 2.23682925089e-05 | 0 | 0 | 0 |

The dominant error type is false positive: `82` FP versus `1` FN. High-confidence FP is fixed as final_probability >= 0.90; count = `53`. Near-threshold is fixed as abs(probability - threshold) <= 0.05; FP near-threshold count = `2`.

## Unique FN

```json
[
  {
    "filename": "person154_bacteria_728.jpeg",
    "patient_id": "person154",
    "final_probability": 0.1944013262788454,
    "probability_seed_42": 0.1121294125914573,
    "probability_seed_2025": 0.1904648393392563,
    "probability_seed_2026": 0.2806097269058227,
    "std_probability": 0.0843091099759451
  }
]
```

## Seed stability

Three-seed label disagreement count: `13` of `624` images. FP disagreement count: `2`. TN disagreement count: `11`.

These are descriptive associations only and do not establish causality.

## Patient-level exploratory analysis

```json
{
  "patient_count": 427,
  "single_image_patient_count": 345,
  "multi_image_patient_count": 82,
  "single_image_error_rate": 0.21739130434782608,
  "multi_image_error_rate": 0.02867383512544803,
  "mixed_correctness_patient_count": 2,
  "mean_within_patient_probability_std": 0.004587250421349185,
  "fp_patient_count": 78,
  "top_fp_patients": [
    {
      "patient_id": "normal2-im-0246",
      "true_label": 0,
      "image_count": 3,
      "error_count": 3,
      "probability_std": 0.007316266997492954,
      "fp_count": 3,
      "has_mixed_correctness": false
    },
    {
      "patient_id": "im-0011",
      "true_label": 0,
      "image_count": 3,
      "error_count": 2,
      "probability_std": 0.4788884137266592,
      "fp_count": 2,
      "has_mixed_correctness": true
    },
    {
      "patient_id": "normal2-im-0173",
      "true_label": 0,
      "image_count": 2,
      "error_count": 2,
      "probability_std": 0.0,
      "fp_count": 2,
      "has_mixed_correctness": false
    },
    {
      "patient_id": "normal2-im-0374",
      "true_label": 0,
      "image_count": 3,
      "error_count": 1,
      "probability_std": 0.5100773692680398,
      "fp_count": 1,
      "has_mixed_correctness": true
    },
    {
      "patient_id": "im-0010",
      "true_label": 0,
      "image_count": 1,
      "error_count": 1,
      "probability_std": 0.0,
      "fp_count": 1,
      "has_mixed_correctness": false
    },
    {
      "patient_id": "im-0015",
      "true_label": 0,
      "image_count": 1,
      "error_count": 1,
      "probability_std": 0.0,
      "fp_count": 1,
      "has_mixed_correctness": false
    },
    {
      "patient_id": "im-0022",
      "true_label": 0,
      "image_count": 1,
      "error_count": 1,
      "probability_std": 0.0,
      "fp_count": 1,
      "has_mixed_correctness": false
    },
    {
      "patient_id": "im-0028",
      "true_label": 0,
      "image_count": 1,
      "error_count": 1,
      "probability_std": 0.0,
      "fp_count": 1,
      "has_mixed_correctness": false
    },
    {
      "patient_id": "im-0039",
      "true_label": 0,
      "image_count": 1,
      "error_count": 1,
      "probability_std": 0.0,
      "fp_count": 1,
      "has_mixed_correctness": false
    },
    {
      "patient_id": "im-0065",
      "true_label": 0,
      "image_count": 1,
      "error_count": 1,
      "probability_std": 0.0,
      "fp_count": 1,
      "has_mixed_correctness": false
    }
  ]
}
```

FP concentration should be interpreted cautiously because patient_id is inferred from filenames.

## Grad-CAM

Grad-CAM was generated for a deterministic sample list using the frozen seed 42 model only. It explains a single model, not the three-model ensemble. Grad-CAM cannot prove medical causality or confirm lesion localization. Heatmaps must not be used as clinical evidence by themselves.

Grad-CAM outputs generated: `35` samples.

Selection rules: all FN; FP top 5 probability, FP 5 nearest threshold, FP 5 highest seed std; TN lowest 5 probability and 5 nearest threshold; TP top 5 probability and 5 nearest threshold. Duplicates were kept once with all reasons recorded.

No training, recalibration, threshold adjustment, model selection, or protocol change was performed.

Limitations: single public dataset, no external clinical validation, inferred patient IDs, and post-hoc visual explanations. The model is not clinically deployable and is not a replacement for clinicians.
