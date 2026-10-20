import pandas as pd
from src.data import read_online_retail
from src.features import make_rfm_features
from src.propensity import make_labels
from src.evaluate import evaluate


def online_rows():
    return pd.DataFrame([
        ["1", "100", "Widget", 2, "2024-01-01", 5, "A", "United Kingdom"],
        ["C2", "100", "Widget", -1, "2024-01-02", 5, "A", "United Kingdom"],
        ["2", "POST", "Postage", 1, "2024-01-03", 2, "A", "United Kingdom"],
        ["3", "101", "Thing", 1, "2024-02-01", 10, "B", "France"],
        ["4", "102", "Later", 1, "2024-04-15", 7, "A", "United Kingdom"],
    ], columns=["Invoice","StockCode","Description","Quantity","InvoiceDate","Price","Customer ID","Country"])


def test_cleaning_audit_and_returns(tmp_path):
    path = tmp_path / "retail.csv"
    online_rows().to_csv(path, index=False)
    clean = read_online_retail(path)
    assert len(clean) == 3
    assert len(clean.attrs["returns"]) == 1
    assert clean.attrs["cleaning_log"]["cancellations"] == 1
    assert clean.attrs["cleaning_log"]["return_rows"] == 1
    assert clean.attrs["cleaning_log"]["rows_removed"] == 2


def test_rfm_is_cutoff_safe_and_exact():
    tx = online_rows().query("Invoice not in ['C2', '2']").copy()
    tx["_revenue"] = tx.Quantity * tx.Price
    cutoff = pd.Timestamp("2024-03-01")
    features = make_rfm_features(tx, cutoff)
    a = features.loc[features["Customer ID"] == "A"].iloc[0]
    assert a["frequency"] == 1
    assert a["monetary"] == 10
    assert a["recency"] == 61


def test_labels_only_post_cutoff():
    tx = online_rows()
    tx["_revenue"] = tx.Quantity * tx.Price
    labels = make_labels(tx, ["A"], pd.Timestamp("2024-03-01"), horizon_days=90)
    assert labels.loc[labels.StockCode == "102", "label"].item() == 1
    assert labels.loc[labels.StockCode == "100", "label"].item() == 0


def test_exact_recommendation_metrics():
    tx = online_rows()
    tx["_revenue"] = tx.Quantity * tx.Price
    rec = pd.DataFrame({"Customer ID": ["A", "A"], "StockCode": ["100", "102"], "rank": [1, 2]})
    metrics = evaluate(rec, tx, pd.Timestamp("2024-03-01"), k=2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 1.0
