"""ChurnGuard dashboard — predict churn for one customer or a whole CSV."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import split_xy

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")

MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("reports/metrics.json")


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved model; if missing or incompatible (e.g. different
    scikit-learn version on the server), rebuild it automatically."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        with st.spinner("Training model for the first time (≈30 s)…"):
            from src.data_generator import generate
            from src.train import main as train

            data_path = Path("data/customers.csv")
            if not data_path.exists():
                data_path.parent.mkdir(parents=True, exist_ok=True)
                generate().to_csv(data_path, index=False)
            train(str(data_path))
        return joblib.load(MODEL_PATH)


def risk_label(p: float, threshold: float) -> str:
    if p >= max(threshold, 0.7):
        return "🔴 High"
    if p >= threshold:
        return "🟠 Medium"
    return "🟢 Low"


bundle = load_model()
pipe, threshold = bundle["pipeline"], bundle["threshold"]

st.title("📉 ChurnGuard")
st.caption(f"Customer churn prediction · model: **{bundle['model_name']}** · decision threshold {threshold:.2f}")

if METRICS_PATH.exists():
    m = json.loads(METRICS_PATH.read_text())["best_model_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("Recall", f"{m['recall']:.1%}")
    c3.metric("Precision", f"{m['precision']:.1%}")
    c4.metric("F1 score", f"{m['f1']:.3f}")

tab_single, tab_batch, tab_insights = st.tabs(["👤 Single customer", "📂 Batch scoring", "📊 Model insights"])

# ---------------- Single prediction ----------------
with tab_single:
    with st.form("customer"):
        a, b, c = st.columns(3)
        with a:
            tenure = st.slider("Tenure (months)", 0, 72, 6)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment method",
                                   ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
        with b:
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            opts = ["Yes", "No"] if internet != "No" else ["No internet"]
            security = st.selectbox("Online security", opts)
            support = st.selectbox("Tech support", opts)
            streaming = st.selectbox("Streaming", opts)
        with c:
            monthly = st.number_input("Monthly charges (₹ / $)", 18.0, 120.0, 85.0)
            calls = st.number_input("Support calls (last 6 months)", 0, 15, 3)
            senior = st.checkbox("Senior citizen")
            partner = st.radio("Partner", ["Yes", "No"], horizontal=True, index=1)
            dependents = st.radio("Dependents", ["Yes", "No"], horizontal=True, index=1)
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "senior_citizen": int(senior), "partner": partner, "dependents": dependents,
            "tenure_months": tenure, "contract": contract, "internet_service": internet,
            "online_security": security, "tech_support": support, "streaming": streaming,
            "paperless_billing": paperless, "payment_method": payment,
            "monthly_charges": monthly, "total_charges": monthly * max(tenure, 1),
            "support_calls": calls,
        }])
        X, _ = split_xy(row)
        p = float(pipe.predict_proba(X)[0, 1])
        st.subheader(f"Churn probability: {p:.0%}  ·  Risk: {risk_label(p, threshold)}")
        st.progress(p)

        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discounted 1-year contract upgrade.")
        if internet != "No" and security == "No":
            tips.append("Bundle free online security for 3 months.")
        if payment == "Electronic check":
            tips.append("Encourage auto-pay via card or bank transfer.")
        if calls >= 3:
            tips.append("Escalate to a senior support agent — many recent complaints.")
        if tenure < 12:
            tips.append("Enrol in the new-customer onboarding programme.")
        if tips and p >= threshold:
            st.markdown("**Suggested retention actions**")
            for t in tips:
                st.markdown(f"- {t}")

# ---------------- Batch prediction ----------------
with tab_batch:
    st.write("Upload a CSV with the same columns as `data/customers.csv` (the `churn` column is optional).")
    file = st.file_uploader("Customer CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        X, _ = split_xy(df)
        df["churn_probability"] = pipe.predict_proba(X)[:, 1].round(3)
        df["risk"] = df["churn_probability"].apply(lambda p: risk_label(p, threshold))
        df = df.sort_values("churn_probability", ascending=False)
        st.success(f"Scored {len(df):,} customers · "
                   f"{(df.churn_probability >= threshold).mean():.1%} flagged as at-risk")
        st.dataframe(df, width="stretch", height=420)
        st.download_button("⬇️ Download predictions", df.to_csv(index=False),
                           "churn_predictions.csv", "text/csv")

# ---------------- Insights ----------------
with tab_insights:
    fig_dir = Path("reports/figures")
    left, right = st.columns(2)
    for i, name in enumerate(["roc_curves", "feature_importance", "confusion_matrix", "tenure_distribution"]):
        path = fig_dir / f"{name}.png"
        if path.exists():
            (left if i % 2 == 0 else right).image(str(path), width="stretch")
    if (fig_dir / "churn_by_segment.png").exists():
        st.image(str(fig_dir / "churn_by_segment.png"), width="stretch")
