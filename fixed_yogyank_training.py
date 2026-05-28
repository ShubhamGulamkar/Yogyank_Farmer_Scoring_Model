import json
import os
import hashlib
import datetime
import warnings
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

VERSION = "yogyank_v1.0.0"
TRAIN_YEARS = [2022, 2023]
TEST_YEARS  = [2024]
ARTIFACTS_DIR = "artifacts"

# Features that are genuinely available at scoring time
# (no future labels, no policy-adjusted targets)
NUMERIC_FEATURES = [
    "land_area_acres",
    "historical_repayment_score",
    "annual_income_inr",
    "liability_ratio_pct",
    "rainfall_deviation_pct",
    "ndvi_score",
]

CATEGORICAL_FEATURES = [
    "district",
    "crop_type",
    "pm_kisan_status",   # input signal only — NOT used to adjust the target
    "irrigation_type",
    "land_ownership",
    "soil_type",
    "sales_channel",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target_entitlement_score"

# Features explicitly excluded and why
EXCLUDED_FEATURES = {
    "farmer_id":                      "identifier — no predictive signal",
    "application_year":               "used only for OOT split, not a model feature",
    "defaulted_in_next_12_months":    "LEAKAGE — future label not available at scoring time",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def md5_of_file(path: str) -> str:
    """Return the MD5 hex digest of a file for artifact integrity checks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str = "farmer_scoring_sample_yogyank_round1.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Basic sanity checks
    assert TARGET in df.columns, f"Target column '{TARGET}' not found."
    for col in EXCLUDED_FEATURES:
        if col in df.columns and col == "defaulted_in_next_12_months":
            print(f"  [LEAKAGE GUARD] Column '{col}' is present — will be excluded.")

    return df


# ── OOT split ─────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    """
    Out-of-time split: train on TRAIN_YEARS, test on TEST_YEARS.
    This simulates real deployment — the model is scored on a future cohort
    it has never seen, which is far more realistic than a random shuffle split.
    """
    df_train = df[df["application_year"].isin(TRAIN_YEARS)].copy()
    df_test  = df[df["application_year"].isin(TEST_YEARS)].copy()

    print(f"  Train: {len(df_train)} rows (years {TRAIN_YEARS})")
    print(f"  Test : {len(df_test)} rows (years {TEST_YEARS})")

    if len(df_test) == 0:
        raise ValueError(
            f"No test rows found for years {TEST_YEARS}. "
            "Check application_year values in the dataset."
        )

    return df_train, df_test


# ── Encoding ──────────────────────────────────────────────────────────────────

def fit_encoders(df_train: pd.DataFrame) -> dict:
    """
    Fit one OrdinalEncoder per categorical column using TRAIN data only.
    handle_unknown='use_encoded_value' + unknown_value=-1 means unseen
    categories at scoring time get a sentinel (-1) rather than crashing
    or silently bleeding future information back into the encoder.
    """
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
        enc.fit(df_train[[col]])
        encoders[col] = enc
        print(f"  Encoder fitted for '{col}': {list(enc.categories_[0])}")
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply frozen encoders to any split (train or test). Never refit."""
    df = df.copy()
    for col, enc in encoders.items():
        df[col] = enc.transform(df[[col]])
    return df


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_X_y(df: pd.DataFrame, encoders: dict):
    df_enc = apply_encoders(df, encoders)
    X = df_enc[ALL_FEATURES].copy()
    y = df_enc[TARGET].copy()
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_train, y_train) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, split_name: str = "OOT Test") -> dict:
    preds = model.predict(X_test)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"\n── {split_name} Metrics ──────────────────────────")
    print(f"  R²  : {r2:.4f}")
    print(f"  MAE : {mae:.2f} score points")
    print("  Note: R² on a synthetic dataset with ~500 rows may still look")
    print("  inflated. Interpret alongside MAE and fairness slices.")
    return {"r2": round(r2, 4), "mae": round(mae, 4), "n": int(len(y_test))}


# ── Fairness slice report ─────────────────────────────────────────────────────

def fairness_slices(model, df_test: pd.DataFrame, encoders: dict):
    """
    Report MAE broken down by key demographic/agronomic slices.
    Flags slices where MAE is >20% worse than the overall MAE.
    These are the slices to monitor post-deployment.
    """
    X_test, y_test = build_X_y(df_test, encoders)
    preds = model.predict(X_test)
    df_eval = df_test.copy()
    df_eval["__pred__"] = preds
    df_eval["__abs_err__"] = np.abs(y_test.values - preds)

    overall_mae = df_eval["__abs_err__"].mean()
    print(f"\n── Fairness / Stability Slices (Overall MAE = {overall_mae:.2f}) ──")

    slices = ["crop_type", "district", "pm_kisan_status",
              "irrigation_type", "land_ownership"]

    results = {}
    for col in slices:
        if col not in df_eval.columns:
            continue
        grp = df_eval.groupby(col)["__abs_err__"].mean().round(2)
        flagged = grp[grp > overall_mae * 1.20].index.tolist()
        print(f"\n  {col}:")
        print(grp.to_string())
        if flagged:
            print(f"  ⚠  Groups with MAE >20% above average: {flagged}")
        results[col] = grp.to_dict()

    return results


# ── Reason codes (top-3 SHAP) ─────────────────────────────────────────────────

def compute_reason_codes(model, X: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    For each farmer row, return the top-N features driving the score
    (signed SHAP values — positive means the feature pushed the score up).
    This is the explainability artifact required for audit and farmer comms.
    """
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)           # shape: (n_rows, n_features)
    shap_df    = pd.DataFrame(shap_vals, columns=X.columns)

    reason_rows = []
    for idx in range(len(shap_df)):
        row = shap_df.iloc[idx].abs().nlargest(top_n)
        signed = {feat: round(float(shap_df.iloc[idx][feat]), 4) for feat in row.index}
        reason_rows.append({
            f"reason_{i+1}_feature": feat
            for i, feat in enumerate(row.index)
        } | {
            f"reason_{i+1}_shap": signed[feat]
            for i, feat in enumerate(row.index)
        })

    return pd.DataFrame(reason_rows)


# ── Artifact saving ───────────────────────────────────────────────────────────

def save_artifacts(model, encoders: dict, metrics: dict, fairness: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Model
    model_path = os.path.join(ARTIFACTS_DIR, "xgboost_yogyank.pkl")
    joblib.dump(model, model_path)

    # 2. Encoders (frozen; must be used identically at scoring time)
    enc_path = os.path.join(ARTIFACTS_DIR, "encoders.pkl")
    joblib.dump(encoders, enc_path)

    # 3. Feature schema (contract between training and scoring)
    schema = {
        "numeric_features":      NUMERIC_FEATURES,
        "categorical_features":  CATEGORICAL_FEATURES,
        "all_features_ordered":  ALL_FEATURES,
        "target":                TARGET,
        "excluded_with_reason":  EXCLUDED_FEATURES,
        "train_years":           TRAIN_YEARS,
        "test_years":            TEST_YEARS,
    }
    schema_path = os.path.join(ARTIFACTS_DIR, "feature_schema.json")
    save_json(schema, schema_path)

    # 4. Version manifest
    manifest = {
        "version":           VERSION,
        "trained_at":        datetime.datetime.utcnow().isoformat() + "Z",
        "oot_test_metrics":  metrics,
        "fairness_slices":   fairness,
        "artifact_checksums": {
            "model":    md5_of_file(model_path),
            "encoders": md5_of_file(enc_path),
            "schema":   md5_of_file(schema_path),
        },
        "policy_note": (
            "This model outputs a bank-agnostic Entitlement Score. "
            "All bank-specific cutoffs, grade mappings, and eligibility "
            "decisions MUST be applied outside this model in a versioned "
            "policy layer. pm_kisan_status is a model input feature only — "
            "it must NOT be used to adjust the raw target during training."
        ),
    }
    manifest_path = os.path.join(ARTIFACTS_DIR, "version_manifest.json")
    save_json(manifest, manifest_path)
    # Update manifest checksum to include itself
    manifest["artifact_checksums"]["manifest"] = md5_of_file(manifest_path)
    save_json(manifest, manifest_path)

    print(f"\n── Artifacts saved to '{ARTIFACTS_DIR}/' ──────────────────────")
    for fname in os.listdir(ARTIFACTS_DIR):
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        print(f"  {fname}  ({os.path.getsize(fpath):,} bytes)")

    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────

def train_pipeline(data_path: str = "farmer_scoring_sample_yogyank_round1.csv"):
    print("=" * 60)
    print(f"  Yogyank Training Pipeline  |  {VERSION}")
    print("=" * 60)

    # ── 1. Load
    df = load_data(data_path)

    # ── 2. Guard: confirm leaky column is never used
    leaky_cols = [c for c in EXCLUDED_FEATURES if c in df.columns]
    print(f"\nExcluded columns confirmed absent from feature set: {leaky_cols}")

    # ── 3. OOT split
    print("\nSplitting data (Out-of-Time) ...")
    df_train, df_test = temporal_split(df)

    # ── 4. Fit encoders on TRAIN only
    print("\nFitting encoders on training data only ...")
    encoders = fit_encoders(df_train)

    # ── 5. Build feature matrices
    X_train, y_train = build_X_y(df_train, encoders)
    X_test,  y_test  = build_X_y(df_test,  encoders)

    # ── 6. Train
    print("\nTraining XGBoost ...")
    model = train_model(X_train, y_train)

    # ── 7. Evaluate
    metrics = evaluate(model, X_test, y_test, split_name="OOT 2024")

    # ── 8. Fairness slices
    fairness = fairness_slices(model, df_test, encoders)

    # ── 9. Reason codes — sample on test set
    print("\nGenerating top-3 reason codes for OOT test farmers ...")
    reason_df = compute_reason_codes(model, X_test)
    reason_out = os.path.join(ARTIFACTS_DIR, "oot_reason_codes.csv")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    # Attach farmer IDs for traceability
    reason_df.insert(0, "farmer_id", df_test["farmer_id"].values)
    reason_df.insert(1, "predicted_score", model.predict(X_test).round(2))
    reason_df.to_csv(reason_out, index=False)
    print(f"  Reason codes written to {reason_out}")
    print(reason_df.head(5).to_string(index=False))

    # ── 10. Save artifacts
    manifest = save_artifacts(model, encoders, metrics, fairness)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Version : {manifest['version']}")
    print(f"  Trained : {manifest['trained_at']}")
    print(f"  OOT R²  : {metrics['r2']}   MAE: {metrics['mae']}")
    print("=" * 60)

    return model, encoders, manifest


if __name__ == "__main__":
    train_pipeline()
