# Yogyank Entitlement Score — Fixed Baseline

**Version:** `yogyank_v1.0.0`  
**Assessment:** Arbix AI / SaakhSetu Round 1 Technical Assessment

---

## Timeline

| | |
|---|---|
| Start time | 2026-05-28 (assessment session) |
| Approximate time spent | ~85 minutes |

---

## Setup

### Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
xgboost
shap
joblib
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost shap joblib
```

### Input files required

Place both files in the same directory as `fixed_yogyank_training.py`:

```
farmer_scoring_sample_yogyank_round1.csv
fixed_yogyank_training.py
```

---

## Running the script

```bash
python fixed_yogyank_training.py
```

The script will:
1. Load the CSV and print row/column counts
2. Confirm excluded columns (leakage guard)
3. Perform an out-of-time split (train: 2022–2023, test: 2024)
4. Fit categorical encoders on training data only
5. Train an XGBoost regressor
6. Print OOT R² and MAE
7. Print a fairness/stability slice report
8. Generate top-3 SHAP reason codes for every 2024 test farmer
9. Save all artifacts to `artifacts/`

---

## Output artifacts

```
artifacts/
  xgboost_yogyank.pkl       # Trained model
  encoders.pkl              # Frozen OrdinalEncoder per categorical column
  feature_schema.json       # Feature list, exclusions + reasons, year split
  version_manifest.json     # Version stamp, UTC timestamp, metrics, MD5 checksums
  oot_reason_codes.csv      # Top-3 SHAP reason codes per OOT farmer
```

---

## What was fixed

| # | Trap in original script | Fix applied |
|---|---|---|
| 1 | **Policy mix-up** — `target_entitlement_score` was modified by subtracting 150 for `pm_kisan_status == "No"` before training | Reverted; `pm_kisan_status` is a plain input feature; policy adjustments must live outside the model |
| 2 | **Data leakage** — `defaulted_in_next_12_months` (a future label) was used as a model feature | Removed from feature set; exclusion reason documented in `feature_schema.json` |
| 3 | **Bad encoding** — single `LabelEncoder` instance reused, silently overwriting mappings; no encoders saved | One `OrdinalEncoder` per column, fitted on train only, serialised to `encoders.pkl` |
| 4 | **Invalid validation** — `train_test_split(shuffle=True)` mixed 2022–2024 records randomly | Replaced with OOT split: train 2022+2023, test 2024 |
| 5 | **No auditability** — only `xgboost_baseline.pkl` saved; no schema, encoders, version stamp, or reason codes | Four-artifact bundle + reason codes CSV saved to `artifacts/` |

---

## Assumptions

- `application_year` is reliable and reflects when the application was genuinely submitted (not back-filled).
- The synthetic dataset's `target_entitlement_score` is the raw, policy-neutral entitlement signal (before the junior scientist's 150-point deduction).
- 2024 is the most recent cohort and is treated as the OOT test set. If the dataset is extended with later years, `TEST_YEARS` should be updated accordingly.
- `farmer_id` is an identifier only and carries no predictive signal.

---

## Validation approach

Out-of-time (OOT) validation was chosen specifically because it simulates real deployment: the model trains on historical applications and is evaluated on a strictly future cohort it has never seen. This is far more conservative than a random shuffle split, which inflates metrics by leaking future patterns into training.

The OOT test set for 2024 is small in this synthetic sample (~100–150 rows). R² on small samples has wide variance; MAE should be treated as the primary reliability metric until a larger holdout is available.

---

## Skipped items (due to time constraints)

- **Full `sklearn.Pipeline` wrapping** — Preprocessing and model are separate objects; a single `Pipeline` would make inference more atomic and reduce the risk of step mis-ordering in production. Would add with more time.
- **Human-readable SHAP reason codes** — Reason codes currently show the feature name and numeric SHAP value. A reverse-mapping step (from encoded integer back to original category label) would make them audit- and farmer-communication ready.
- **Hyperparameter tuning** — Model uses reasonable defaults. A cross-validated grid or Bayesian search on the training fold would improve calibration.
- **Confidence intervals / prediction intervals** — A regulated score should ideally carry an uncertainty estimate (e.g. conformal prediction bounds).
- **Unit tests** — Leakage guard, encoder boundary, and artifact checksum checks would benefit from a `pytest` suite.

---

## Notes on AI/LLM tool usage

See `LLM_NOTES.md`.
