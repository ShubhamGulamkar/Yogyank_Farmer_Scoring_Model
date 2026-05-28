# LLM_NOTES.md — AI / Coding Tool Usage Disclosure

> Required disclosure per assessment policy. Responsible tool use with personal verification.

---

## Tools used

- **Claude (Anthropic)** — used for flaw identification review, code structure planning, and memo drafting.

---

## Where used

| Area | How tool was used |
|---|---|
| Flaw identification | Reviewed the broken script against the five suspected traps; used Claude to cross-check whether any additional leakage vectors were missed |
| Code refactoring | Used Claude to generate initial structure for the fixed training pipeline |
| Memo drafting | Used Claude to draft prose for `audit_memo.md`; reviewed and edited for accuracy and domain alignment |
| Artifact design | Discussed what should go into `version_manifest.json` and `feature_schema.json` |

---

## Representative prompts (paraphrased)

1. "Here is a flawed ML training script for a credit scoring engine used by farmers. Identify all data leakage, policy mixing, encoding, and auditability issues. Explain why each is dangerous in a regulated context."

2. "Rewrite the script with: leakage-safe feature selection, OOT temporal split (train 2022-2023, test 2024), one OrdinalEncoder per categorical column fitted only on train, SHAP-based top-3 reason codes per farmer, and a four-artifact bundle with version manifest and MD5 checksums."

3. "Write an audit memo following the structure: para 1 = what was dangerous, para 2 = what was fixed, limitations = one thing not trusted yet + one improvement, monitoring = at least two fairness slices."

---

## What I accepted

1. **OrdinalEncoder with `handle_unknown='use_encoded_value'` and `unknown_value=-1`** — Claude suggested this as the correct replacement for `LabelEncoder`, and it is the right choice for a regulated scoring context where unseen categories at inference time must be handled gracefully rather than raising an exception or silently re-encoding.

2. **MD5 checksum manifest** — Claude suggested including artifact checksums in the version manifest. This is standard practice for model governance and reproducibility audits; accepted and included.

---

## What I rejected or corrected

1. **Claude initially suggested wrapping everything in a `sklearn.Pipeline`** — This is good engineering but introduces complexity in 90 minutes when artifact serialisation, SHAP integration, and OOT split logic all need to be explicitly visible and auditable. A Pipeline would hide the preprocessing steps inside an opaque object, making it harder for a reviewer to inspect the encoder boundary. I kept preprocessing and model as separate, explicitly saved artifacts. The README notes the Pipeline as a future improvement, but it was deliberately not included in this 90-minute baseline.

---

## What I personally verified

| Check | How verified |
|---|---|
| **Leakage check** | Confirmed `defaulted_in_next_12_months` is not in `ALL_FEATURES`; the runtime guard prints a log line confirming exclusion |
| **OOT split** | Printed row counts for train (2022+2023) and test (2024); verified no year overlap |
| **Encoder boundary** | Confirmed `fit_encoders()` is called only with `df_train`; `apply_encoders()` is called separately for train and test without refitting |
| **Saved artifacts** | Ran the pipeline and confirmed all five files appear in `artifacts/` with non-zero sizes |
| **Reason code logic** | Inspected `oot_reason_codes.csv` first five rows; confirmed three feature columns and three SHAP value columns per row; checked that SHAP signs are directionally plausible (e.g. high `historical_repayment_score` → positive SHAP) |
| **Run output** | Executed `python fixed_yogyank_training.py` end-to-end; confirmed no exceptions, R² printed, fairness slice table printed, artifact checksums in manifest |
