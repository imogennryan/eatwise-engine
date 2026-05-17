import json

import joblib
import numpy as np
import pandas as pd

# --- Module-level loads (run once on import) ---

MODEL = joblib.load("models/obesity_classifier.pkl")

with open("models/class_labels.json") as _f:
    CLASS_LABELS = {int(k): v for k, v in json.load(_f).items()}

# Derive canonical training X columns: replicate Phase 2 Section A preprocessing.
# Load D1 encoded CSV, remove 170 BMI-mismatched rows, drop Height/Weight and target.
def _derive_x_columns():
    df_cat = pd.read_csv("cleaned_data/d1_obesity_prediction_cleaned.csv")
    df_enc = pd.read_csv("cleaned_data/d1_obesity_prediction_encoded.csv")

    BMI_BANDS = {
        "Insufficient_Weight": (-np.inf, 18.5),
        "Normal_Weight":        (18.5,   25.0),
        "Overweight_Level_I":   (25.0,   27.5),
        "Overweight_Level_II":  (27.5,   30.0),
        "Obesity_Type_I":       (30.0,   35.0),
        "Obesity_Type_II":      (35.0,   40.0),
        "Obesity_Type_III":     (40.0,   np.inf),
    }

    df_check = df_cat.copy()
    df_check["BMI_computed"] = df_check["Weight"] / (df_check["Height"] ** 2)

    def in_band(row):
        lo, hi = BMI_BANDS[row["Obesity"]]
        return lo <= row["BMI_computed"] < hi

    df_check["band_match"] = df_check.apply(in_band, axis=1)
    mismatch_idx = df_check.index[~df_check["band_match"]].tolist()

    df_clean = df_enc.drop(index=mismatch_idx).reset_index(drop=True)
    df_clean = df_clean.drop(columns=["Height", "Weight"])
    X = df_clean.drop(columns=["target_encoded"])
    return list(X.columns)


X_COLUMNS = _derive_x_columns()

# Nominal columns that were one-hot encoded during Phase 2 training.
# Note: UI key 'family_history_with_overweight' is renamed to 'family_history' before encoding.
_NOMINALS = ["Gender", "family_history", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]


def get_feature_importances(top_n: int = 12) -> list:
    """Return top_n (feature_name, importance) pairs sorted by importance descending."""
    pairs = sorted(zip(X_COLUMNS, MODEL.feature_importances_), key=lambda x: x[1], reverse=True)
    return [(name, float(imp)) for name, imp in pairs[:top_n]]


def predict_obesity(raw_lifestyle_dict: dict) -> tuple[str, np.ndarray]:
    """
    Predict 7-class obesity label from raw lifestyle features.

    Parameters
    ----------
    raw_lifestyle_dict : dict
        Keys: Gender, Age, family_history_with_overweight, FAVC, FCVC, NCP,
              CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS.
        Height and Weight are NOT inputs (excluded from training).

    Returns
    -------
    (predicted_class_label : str, prob_array : np.ndarray)
        prob_array has shape (7,) — one probability per class in CLASS_LABELS order.
    """
    d = dict(raw_lifestyle_dict)
    if "family_history_with_overweight" in d:
        d["family_history"] = d.pop("family_history_with_overweight")

    row = pd.DataFrame([d])
    for col in row.select_dtypes(include="object").columns:
        row[col] = row[col].str.strip()

    row_encoded = pd.get_dummies(row, columns=_NOMINALS)
    row_aligned = row_encoded.reindex(columns=X_COLUMNS, fill_value=0)

    pred_idx = int(MODEL.predict(row_aligned)[0])
    prob_array = MODEL.predict_proba(row_aligned)[0]  # shape (7,)

    label = CLASS_LABELS[pred_idx]
    return label, prob_array
