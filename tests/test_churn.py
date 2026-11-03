import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data_generator import generate  # noqa: E402
from src.features import CATEGORICAL, NUMERIC, add_features, build_preprocessor, split_xy  # noqa: E402


def test_generator_shape_and_label():
    df = generate(500, seed=1)
    assert len(df) == 500
    assert set(df["churn"].unique()) <= {0, 1}
    assert 0.1 < df["churn"].mean() < 0.5
    assert df["customer_id"].is_unique


def test_generator_is_reproducible():
    assert generate(100, seed=7).equals(generate(100, seed=7))


def test_feature_engineering():
    df = add_features(generate(200))
    assert {"tenure_group", "avg_monthly_spend", "services_count"} <= set(df.columns)
    assert df["services_count"].between(0, 3).all()


def test_split_xy_without_target():
    df = generate(50).drop(columns=["churn"])
    X, y = split_xy(df)
    assert y is None and list(X.columns) == NUMERIC + CATEGORICAL


def test_model_learns_signal():
    df = generate(2000, seed=3)
    X, y = split_xy(df)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X[:1500], y[:1500])
    assert pipe.score(X[1500:], y[1500:]) > 0.7
