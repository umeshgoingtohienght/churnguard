"""Generate a realistic synthetic telecom customer dataset.

The churn label is produced by a hidden logistic model whose drivers mirror
patterns seen in real telecom data (short tenure, month-to-month contracts,
fibre plans without support, electronic-check payments, many support calls).
This keeps the project fully reproducible without any download.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    max_tenure = np.select(
        [contract == "Month-to-month", contract == "One year"], [40, 60], default=72
    )
    tenure = np.clip((rng.beta(1.1, 1.6, n) * max_tenure).round().astype(int), 0, 72)

    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    has_net = internet != "No"
    yes_no = lambda p: np.where(rng.random(n) < p, "Yes", "No")  # noqa: E731

    online_security = np.where(has_net, yes_no(0.35), "No internet")
    tech_support = np.where(has_net, yes_no(0.37), "No internet")
    streaming = np.where(has_net, yes_no(0.45), "No internet")

    base = np.select([internet == "Fiber optic", internet == "DSL"], [70, 45], default=20)
    monthly = (
        base
        + (streaming == "Yes") * 15
        + (online_security == "Yes") * 6
        + (tech_support == "Yes") * 6
        + rng.normal(0, 6, n)
    ).clip(18, 120).round(2)
    total = (monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.02, n)).round(2)

    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n,
        p=[0.34, 0.22, 0.22, 0.22],
    )
    senior = (rng.random(n) < 0.16).astype(int)
    partner = yes_no(0.48)
    dependents = np.where(partner == "Yes", yes_no(0.45), yes_no(0.12))
    paperless = yes_no(0.59)
    support_calls = rng.poisson(1.2 + 1.3 * (internet == "Fiber optic"), n)

    # ---- hidden churn model -----------------------------------------
    logit = (
        -1.95
        + 1.5 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 0.045 * tenure
        + 0.9 * (internet == "Fiber optic")
        - 0.6 * (internet == "No")
        - 0.6 * (online_security == "Yes")
        - 0.5 * (tech_support == "Yes")
        + 0.7 * (payment == "Electronic check")
        + 0.35 * senior
        - 0.25 * (dependents == "Yes")
        + 0.3 * (paperless == "Yes")
        + 0.28 * support_calls
        + 0.012 * (monthly - 65)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "senior_citizen": senior,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure,
        "contract": contract,
        "internet_service": internet,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming": streaming,
        "paperless_billing": paperless,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
        "support_calls": support_calls,
        "churn": churn,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic churn data")
    parser.add_argument("--rows", type=int, default=7000)
    parser.add_argument("--out", default="data/customers.csv")
    args = parser.parse_args()
    df = generate(args.rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out} · churn rate {df.churn.mean():.1%}")
