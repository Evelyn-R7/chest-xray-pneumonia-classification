# Final test results

This report is generated only after the frozen protocol has been evaluated once.

- Frozen protocol SHA-256: `6835e0135b286046c37d2fa765aeaea2f564e3b8155e3012c81052088b75885e`
- Test manifest SHA-256: `6ba1a4093a601c590b433e99e8a332b2af3a5f46c29f062dd8e1d283f3a2622d`
- Ensemble: EfficientNetB0 control seeds 42, 2025, 2026 with equal probability averaging
- Calibration: none
- Primary balanced threshold: `0.5618644666666667`
- Benchmark threshold: `0.5`

## Point estimates

```json
{
  "threshold_0_5": {
    "threshold": 0.5,
    "accuracy": 0.8573717948717948,
    "precision": 0.8155136268343816,
    "sensitivity": 0.9974358974358974,
    "recall": 0.9974358974358974,
    "specificity": 0.6239316239316239,
    "f1": 0.8973471741637832,
    "balanced_accuracy": 0.8106837606837607,
    "roc_auc": 0.9751424501424502,
    "pr_auc": 0.9828825361211988,
    "npv": 0.9931972789115646,
    "brier_score": 0.1196986974403211,
    "tn": 146,
    "fp": 88,
    "fn": 1,
    "tp": 389,
    "log_loss": 0.4948348721015625,
    "ece": 0.14384965376724812
  },
  "balanced_threshold": {
    "threshold": 0.5618644666666667,
    "accuracy": 0.8669871794871795,
    "precision": 0.8259023354564756,
    "sensitivity": 0.9974358974358974,
    "recall": 0.9974358974358974,
    "specificity": 0.6495726495726496,
    "f1": 0.9036004645760743,
    "balanced_accuracy": 0.8235042735042735,
    "roc_auc": 0.9751424501424502,
    "pr_auc": 0.9828825361211988,
    "npv": 0.9934640522875817,
    "brier_score": 0.1196986974403211,
    "tn": 152,
    "fp": 82,
    "fn": 1,
    "tp": 389,
    "log_loss": 0.4948348721015625,
    "ece": 0.14384965376724812
  }
}
```

## Patient-cluster bootstrap 95% CI

```json
{
  "replicates_requested": 5000,
  "successful_replicates": 5000,
  "failed_replicates": 0,
  "confidence_level": 0.95,
  "method": "patient-level stratified cluster bootstrap",
  "intervals": {
    "balanced_threshold": {
      "accuracy": {
        "ci_lower": 0.8400630048465266,
        "ci_upper": 0.8918939773106439
      },
      "precision": {
        "ci_lower": 0.7940503432494279,
        "ci_upper": 0.85625
      },
      "sensitivity": {
        "ci_lower": 0.991666086350975,
        "ci_upper": 1.0
      },
      "specificity": {
        "ci_lower": 0.5857664592497228,
        "ci_upper": 0.7112214839632582
      },
      "f1": {
        "ci_lower": 0.8836158192090395,
        "ci_upper": 0.9217016623866146
      },
      "balanced_accuracy": {
        "ci_lower": 0.7916422914394874,
        "ci_upper": 0.8546064058531258
      },
      "npv": {
        "ci_lower": 0.9791666666666666,
        "ci_upper": 1.0
      },
      "roc_auc": {
        "ci_lower": 0.9623891946212546,
        "ci_upper": 0.9852878217928157
      },
      "pr_auc": {
        "ci_lower": 0.9706529388647238,
        "ci_upper": 0.9912564468503975
      },
      "brier_score": {
        "ci_lower": 0.0989532761839498,
        "ci_upper": 0.14256730807944742
      },
      "log_loss": {
        "ci_lower": 0.3946393764079909,
        "ci_upper": 0.6086288028658173
      },
      "ece": {
        "ci_lower": 0.12253283059800364,
        "ci_upper": 0.16813337923558858
      }
    },
    "threshold_0_5": {
      "accuracy": {
        "ci_lower": 0.8299991582491583,
        "ci_upper": 0.8825260471295713
      },
      "precision": {
        "ci_lower": 0.7837259100642399,
        "ci_upper": 0.8456659619450317
      },
      "sensitivity": {
        "ci_lower": 0.991666086350975,
        "ci_upper": 1.0
      },
      "specificity": {
        "ci_lower": 0.5584415584415584,
        "ci_upper": 0.6866952789699571
      },
      "f1": {
        "ci_lower": 0.8773895303181443,
        "ci_upper": 0.9153606761106186
      },
      "balanced_accuracy": {
        "ci_lower": 0.7781011469299618,
        "ci_upper": 0.8422431058881407
      },
      "npv": {
        "ci_lower": 0.9782608695652174,
        "ci_upper": 1.0
      },
      "roc_auc": {
        "ci_lower": 0.9623891946212546,
        "ci_upper": 0.9852878217928157
      },
      "pr_auc": {
        "ci_lower": 0.9706529388647238,
        "ci_upper": 0.9912564468503975
      },
      "brier_score": {
        "ci_lower": 0.0989532761839498,
        "ci_upper": 0.14256730807944742
      },
      "log_loss": {
        "ci_lower": 0.3946393764079909,
        "ci_upper": 0.6086288028658173
      },
      "ece": {
        "ci_lower": 0.12253283059800364,
        "ci_upper": 0.16813337923558858
      }
    }
  }
}
```

## Secondary patient-level analysis

Patient-level probabilities average all images for each inferred patient_id. No patient-level threshold was re-selected.

```json
{
  "threshold_0_5": {
    "threshold": 0.5,
    "accuracy": 0.8032786885245902,
    "precision": 0.7077464788732394,
    "sensitivity": 0.995049504950495,
    "recall": 0.995049504950495,
    "specificity": 0.6311111111111111,
    "f1": 0.8271604938271605,
    "balanced_accuracy": 0.8130803080308031,
    "roc_auc": 0.9685808580858086,
    "pr_auc": 0.954829171032653,
    "npv": 0.993006993006993,
    "brier_score": 0.16259336297678334,
    "tn": 142,
    "fp": 83,
    "fn": 1,
    "tp": 201,
    "log_loss": 0.6720519030529783,
    "ece": 0.2011281744774771
  },
  "balanced_threshold": {
    "threshold": 0.5618644666666667,
    "accuracy": 0.8173302107728337,
    "precision": 0.7230215827338129,
    "sensitivity": 0.995049504950495,
    "recall": 0.995049504950495,
    "specificity": 0.6577777777777778,
    "f1": 0.8375,
    "balanced_accuracy": 0.8264136413641364,
    "roc_auc": 0.9685808580858086,
    "pr_auc": 0.954829171032653,
    "npv": 0.9932885906040269,
    "brier_score": 0.16259336297678334,
    "tn": 148,
    "fp": 77,
    "fn": 1,
    "tp": 201,
    "log_loss": 0.6720519030529783,
    "ece": 0.2011281744774771
  }
}
```

## Validation-test generalization gap

These are descriptive test metric minus frozen validation metric comparisons only. They are not used to alter the model, thresholds, calibration, or training.

```json
{
  "threshold_0_5": {
    "accuracy": -0.1164227543944526,
    "precision": -0.17119242929560363,
    "sensitivity": 0.02082771030139452,
    "specificity": -0.34273504273504274,
    "f1": -0.08428397939977306,
    "balanced_accuracy": -0.16095366621682405
  },
  "balanced_threshold": {
    "accuracy": -0.10785558780841797,
    "precision": -0.16371190786696654,
    "sensitivity": 0.02228969860548813,
    "specificity": -0.3245014245014245,
    "f1": -0.07872648682304195,
    "balanced_accuracy": -0.15110586294796824
  },
  "threshold_independent": {
    "roc_auc": -0.02158702787357758,
    "pr_auc": -0.015868159632094203,
    "brier_score": 0.09955152837284639,
    "log_loss": 0.4263961068048384
  }
}
```

## Anomalies

[]

No retraining, recalibration, threshold adjustment, or ensemble reweighting was performed after seeing test results.

Limitations: patient_id values are inferred from filenames, the data come from a single public source, no external validation is included, and this model is not clinically deployable or a replacement for clinician judgment.
