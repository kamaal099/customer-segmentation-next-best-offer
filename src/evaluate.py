"""Recommendation evaluation."""
import numpy as np
import pandas as pd


def evaluate(recommendations, transactions, cutoff, k=3, bootstrap=200):
    recommendations = recommendations.copy()
    recommendations["Customer ID"] = pd.to_numeric(recommendations["Customer ID"], errors="coerce")
    transactions = transactions.copy()
    transactions["Customer ID"] = pd.to_numeric(transactions["Customer ID"], errors="coerce")
    future = transactions[pd.to_datetime(transactions.InvoiceDate) > pd.Timestamp(cutoff)]
    truth = future.groupby("Customer ID")["StockCode"].agg(lambda x: set(x.astype(str)))
    rec = recommendations[recommendations["rank"] <= k]

    hits = total = rel = 0
    per, ndcgs = [], []
    for c, g in rec.groupby("Customer ID"):
        actual = set(truth.get(c, []))
        items = g.sort_values("rank").StockCode.astype(str).tolist()
        h = sum(i in actual for i in items)
        hits += h
        total += len(items)
        rel += len(actual)
        dcg = sum(1 / np.log2(r + 2) for r, i in enumerate(items) if i in actual)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(actual), len(items))))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        per.append({"Customer ID": c, "hits": h,
                    "precision_at_k": h / len(items) if items else 0,
                    "recall_at_k": h / len(actual) if actual else 0})
    p = hits / total if total else 0.0
    r = hits / rel if rel else 0.0
    return {"precision_at_k": p, "recall_at_k": r,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "per_customer": pd.DataFrame(per)}


evaluate_recommendations = evaluate