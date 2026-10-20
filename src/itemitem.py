"""Item-item collaborative filtering next-best-offer (cosine similarity)."""
import numpy as np
import pandas as pd


def recommend_item_item(history, assigned, top_k=3, max_items=2500):
    h = history.copy()
    h["cid"] = pd.to_numeric(h["Customer ID"], errors="coerce")
    h = h.dropna(subset=["cid"])
    if "_revenue" in h.columns:
        h = h[h["_revenue"] > 0]
    h["sc"] = h["StockCode"].astype(str)

    items = h.groupby("sc").cid.nunique().sort_values(ascending=False).index[:max_items]
    h = h[h["sc"].isin(items)]
    custs = np.sort(h["cid"].unique())
    c_idx = {c: i for i, c in enumerate(custs)}
    i_idx = {s: j for j, s in enumerate(items)}

    U = np.zeros((len(custs), len(items)), dtype=np.float32)
    pairs = h[["cid", "sc"]].drop_duplicates()
    U[pairs["cid"].map(c_idx).to_numpy(), pairs["sc"].map(i_idx).to_numpy()] = 1.0

    co = U.T @ U
    norms = np.sqrt(np.diag(co))
    norms[norms == 0] = 1.0
    S = co / np.outer(norms, norms)
    np.fill_diagonal(S, 0.0)

    scores = U @ S
    scores[U > 0] = -np.inf  # never recommend an item already bought

    seg = {}
    if "segment" in assigned.columns:
        seg = dict(zip(pd.to_numeric(assigned["Customer ID"], errors="coerce"), assigned["segment"]))

    rows = []
    for i, c in enumerate(custs):
        top = np.argsort(-scores[i])[:top_k]
        for r, j in enumerate(top, start=1):
            if np.isfinite(scores[i, j]):
                rows.append({"Customer ID": float(c), "segment": seg.get(c, -1),
                             "StockCode": items[j], "rank": r, "score": float(scores[i, j]),
                             "method": "item_item",
                             "reason": "similar to items this customer already bought"})
    return pd.DataFrame(rows)