import pandas as pd
from src.data import clean_transactions
from src.features import derive_cutoff, make_rfm_features
from src.segmentation import RFMSegmenter
from src.propensity import fit_propensity
from src.recommenders import recommend
from src.evaluation import evaluate_recommendations


def fixture():
    return pd.DataFrame([
        ["a", "2024-01-01", 10, "p1"], ["a", "2024-02-01", 20, "p2"],
        ["b", "2024-01-15", 5, "p1"], ["b", "2024-03-15", 7, "p3"],
        ["c", "2024-02-20", 40, "p2"], ["c", "2024-04-15", 8, "p3"],
        ["a", "2024-05-20", 4, "p3"], ["b", "2024-05-21", 6, "p2"],
    ], columns=["customer_id", "transaction_date", "amount", "product_id"])


def test_clean_and_features_are_cutoff_safe():
    tx = clean_transactions(fixture())
    cutoff = derive_cutoff(tx, holdout_days=10)
    features = make_rfm_features(tx, cutoff)
    assert features.customer_id.tolist() == ["a", "b", "c"]
    assert features.loc[features.customer_id == "a", "frequency"].item() == 2


def test_deterministic_recommendation_and_evaluation():
    tx = clean_transactions(fixture())
    cutoff = pd.Timestamp("2024-04-30")
    assignments = RFMSegmenter(2).fit_predict(make_rfm_features(tx, cutoff))
    prop = fit_propensity(tx, assignments, cutoff)
    rec = recommend(prop, tx, assignments, 2)
    assert (rec.groupby("customer_id").size() <= 2).all()
    assert set(rec.product_id).issubset(set(tx.product_id))
    for customer, group in rec.groupby("customer_id"):
        assert not set(group.product_id) & set(tx.loc[tx.customer_id == customer, "product_id"])
    metrics = evaluate_recommendations(rec, tx, cutoff, 2)
    assert 0 <= metrics["precision_at_k"] <= 1
