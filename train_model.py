"""
train_model.py
--------------
Loads House_Rent_Dataset.csv, preprocesses it, trains a
GradientBoostingRegressor, evaluates it, and saves:
  - model/rent_model.pkl   (trained model)
  - model/encoders.pkl     (LabelEncoders + feature metadata)
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "House_Rent_Dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "rent_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "encoders.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── Feature engineering ──────────────────────────────────────────────────────
def parse_floor(floor_str):
    """Convert 'Ground out of 5' to (0, 5), '3 out of 10' to (3, 10)."""
    if pd.isna(floor_str):
        return 0, 0
    floor_str = str(floor_str).lower().strip()
    if "ground" in floor_str:
        current = 0
    else:
        m = re.match(r"(\d+)", floor_str)
        current = int(m.group(1)) if m else 0
    m_total = re.search(r"out of\s+(\d+)", floor_str)
    total = int(m_total.group(1)) if m_total else current
    return current, total

df[["Floor_Current", "Floor_Total"]] = df["Floor"].apply(
    lambda x: pd.Series(parse_floor(x))
)

# Categorical columns to encode
CAT_COLS = ["Area Type", "City", "Furnishing Status", "Tenant Preferred", "Point of Contact"]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── Feature matrix ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "BHK", "Size", "Bathroom",
    "Floor_Current", "Floor_Total",
    "Area Type_enc", "City_enc", "Furnishing Status_enc",
    "Tenant Preferred_enc", "Point of Contact_enc",
]

X = df[FEATURE_COLS]
y = df["Rent"]

# Store metadata for the API / frontend
encoders["feature_cols"]    = FEATURE_COLS
encoders["cat_cols"]        = CAT_COLS
encoders["unique_values"]   = {col: sorted(df[col].unique().tolist()) for col in CAT_COLS}

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model ────────────────────────────────────────────────────────────────────
print("Training Gradient Boosting Regressor ...")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n-- Model Evaluation --")
print(f"  MAE  : Rs.{mae:,.0f}")
print(f"  RMSE : Rs.{rmse:,.0f}")
print(f"  R2   : {r2:.4f}")
print("----------------------\n")

# ── Save
joblib.dump(model,    MODEL_PATH)
joblib.dump(encoders, ENC_PATH)
print(f"Model    saved -> {MODEL_PATH}")
print(f"Encoders saved -> {ENC_PATH}")
