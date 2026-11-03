"""Feature engineering and the preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn"
ID_COL = "customer_id"

CATEGORICAL = [
    "partner", "dependents", "contract", "internet_service", "online_security",
    "tech_support", "streaming", "paperless_billing", "payment_method", "tenure_group",
]
NUMERIC = [
    "senior_citizen", "tenure_months", "monthly_charges", "total_charges",
    "support_calls", "avg_monthly_spend", "services_count",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features. Pure function: returns a copy."""
    out = df.copy()
    out["tenure_group"] = pd.cut(
        out["tenure_months"], bins=[-1, 12, 24, 48, 72],
        labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr"],
    ).astype(str)
    out["avg_monthly_spend"] = out["total_charges"] / np.maximum(out["tenure_months"], 1)
    out["services_count"] = sum(
        (out[c] == "Yes").astype(int) for c in ["online_security", "tech_support", "streaming"]
    )
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
    ])


def split_xy(df: pd.DataFrame):
    df = add_features(df)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET] if TARGET in df else None
    return X, y
