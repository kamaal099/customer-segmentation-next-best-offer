import pandas as pd

tx = pd.read_parquet("artifacts/transactions.parquet")
rec = pd.read_csv("artifacts/recommendations.csv")
tx["cid"] = pd.to_numeric(tx["Customer ID"], errors="coerce")
tx["sc"] = tx.StockCode.astype(str)
cut = tx.InvoiceDate.max() - pd.Timedelta(days=90)
hist = tx[tx.InvoiceDate <= cut]
fut = tx[tx.InvoiceDate > cut]

popular = hist.groupby("sc").cid.nunique().sort_values(ascending=False).index.tolist()[:200]
seen = hist.groupby("cid").sc.apply(set).to_dict()
bought = set(zip(fut.cid, fut.sc))

custs = rec["Customer ID"].unique()
active = set(fut.cid.dropna().unique()) & set(custs)

base_hits = base_hits_active = 0
for c in custs:
    picks = [p for p in popular if p not in seen.get(c, set())][:3]
    h = sum((c, p) in bought for p in picks)
    base_hits += h
    if c in active:
        base_hits_active += h

eng_hits = sum((c, s) in bought for c, s in zip(rec["Customer ID"], rec.StockCode.astype(str)))
eng_hits_active = sum((c, s) in bought for c, s in zip(rec["Customer ID"], rec.StockCode.astype(str)) if c in active)

n_all = len(custs) * 3
n_act = len(active) * 3
print("baseline precision@3, all customers:", base_hits / n_all)
print("engine   precision@3, all customers:", eng_hits / n_all)
print("baseline precision@3, active only:  ", base_hits_active / n_act)
print("engine   precision@3, active only:  ", eng_hits_active / n_act)
print("customers who bought in holdout:", len(active))
