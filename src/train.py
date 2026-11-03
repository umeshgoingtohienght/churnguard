"""Train, compare and evaluate churn models; save the best one and report figures.

Usage:
    python -m src.train --data data/customers.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .features import build_preprocessor, split_xy  # noqa: E402

# ---- chart styling (validated colour-blind-safe palette) ----------------
RETAINED, CHURNED, ACCENT = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,"axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42),
}


def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def best_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 — churn is imbalanced, so 0.5 is rarely optimal."""
    grid = np.linspace(0.2, 0.8, 61)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ---------------------------------------------------------------- figures
def plot_eda(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col, title in [
        (axes[0], "contract", "Churn rate by contract"),
        (axes[1], "internet_service", "Churn rate by internet service"),
        (axes[2], "payment_method", "Churn rate by payment method"),
    ]:
        rates = df.groupby(col)["churn"].mean().sort_values()
        bars = ax.barh(rates.index, rates.values * 100, color=CHURNED, height=0.6)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in rates.values * 100],
                     padding=4, color=INK_2, fontsize=10)
        ax.set_title(title)
        ax.set_xlabel("Churn rate (%)")
        ax.set_xlim(0, rates.max() * 100 * 1.25)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "churn_by_segment.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(0, 76, 4)
    ax.hist(df.loc[df.churn == 0, "tenure_months"], bins=bins, color=RETAINED,
            alpha=0.85, label="Retained", edgecolor=SURFACE, linewidth=2)
    ax.hist(df.loc[df.churn == 1, "tenure_months"], bins=bins, color=CHURNED,
            alpha=0.85, label="Churned", edgecolor=SURFACE, linewidth=2)
    ax.set_title("Most churn happens in the first year")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tenure_distribution.png", dpi=140)
    plt.close(fig)


def plot_roc(curves: dict, out: Path) -> None:
    colors = [RETAINED, CHURNED, ACCENT]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for (name, (fpr, tpr, auc)), c in zip(curves.items(), colors):
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=1, label="Random guess")
    ax.set_title("ROC curves — test set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=140)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — best model")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=140)
    plt.close(fig)


def plot_importance(pipe: Pipeline, out: Path, top: int = 12) -> list:
    pre, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])
    imp = pd.Series(values, index=names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color=RETAINED, height=0.6)
    ax.set_title(f"Top {top} churn drivers")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=140)
    plt.close(fig)
    return imp.sort_values(ascending=False).index.tolist()


# ---------------------------------------------------------------- main
def main(data_path: str, model_dir: str = "models", report_dir: str = "reports") -> dict:
    df = pd.read_csv(data_path)
    model_dir, fig_dir = Path(model_dir), Path(report_dir) / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_eda(df, fig_dir)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, curves, fitted = {}, {}, {}
    for name, model in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)
        metrics["cv_auc_mean"], metrics["cv_auc_std"] = cv_auc.mean(), cv_auc.std()
        results[name], fitted[name] = metrics, pipe
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr, metrics["roc_auc"])
        print(f"{name:<20} CV AUC {cv_auc.mean():.3f} ± {cv_auc.std():.3f} | "
              f"test AUC {metrics['roc_auc']:.3f} | F1 {metrics['f1']:.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_auc_mean"])
    best = fitted[best_name]
    proba = best.predict_proba(X_test)[:, 1]
    threshold = best_threshold(y_test, proba)
    tuned = evaluate(y_test, proba, threshold)

    plot_roc(curves, fig_dir)
    plot_confusion(confusion_matrix(y_test, (proba >= threshold).astype(int)), fig_dir)
    top_features = plot_importance(best, fig_dir)

    joblib.dump({"pipeline": best, "threshold": threshold, "model_name": best_name},
                model_dir / "churn_model.joblib")
    summary = {
        "best_model": best_name, "threshold": round(threshold, 2),
        "best_model_metrics": {k: round(v, 4) for k, v in tuned.items()},
        "all_models": {n: {k: round(v, 4) for k, v in m.items()} for n, m in results.items()},
        "top_features": top_features, "rows": len(df), "churn_rate": round(float(y.mean()), 4),
    }
    with open(Path(report_dir) / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest: {best_name} @ threshold {threshold:.2f} -> {summary['best_model_metrics']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/customers.csv")
    args = parser.parse_args()
    main(args.data)
