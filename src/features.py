"""Leakage-safe customer and product features."""
from __future__ import annotations
import pandas as pd

def derive_cutoff(transactions: pd.DataFrame, holdout_days: int = 90, cutoff=None) -> pd.Timestamp:
    date_col = "InvoiceDate" if "InvoiceDate" in transactions else "transaction_date"
    latest = pd.to_datetime(transactions[date_col]).max()
    value = pd.Timestamp(cutoff) if cutoff is not None else latest - pd.Timedelta(days=holdout_days)
    if value >= latest: raise ValueError("cutoff must precede latest InvoiceDate")
    return value

def make_rfm_features(transactions: pd.DataFrame, cutoff, as_of=None) -> pd.DataFrame:
    cut = pd.Timestamp(cutoff); t=transactions.copy()
    ren={k:v for k,v in {"transaction_date":"InvoiceDate","customer_id":"Customer ID","amount":"_revenue","product_id":"StockCode"}.items() if k in t and v not in t}
    t=t.rename(columns=ren); hist = t[pd.to_datetime(t.InvoiceDate) <= cut].copy()
    if hist.empty: raise ValueError("No history at or before cutoff")
    ref = pd.Timestamp(as_of) if as_of is not None else cut + pd.Timedelta(days=1)
    g = hist.groupby("Customer ID", as_index=False).agg(
        recency=("InvoiceDate", lambda x: (ref-pd.to_datetime(x).max()).days),
        frequency=("Invoice", "nunique"), monetary=("_revenue", "sum"),
        total_quantity=("Quantity", "sum"), unique_products=("StockCode", "nunique"),
        avg_order_value=("_revenue", "sum"))
    orders = hist.groupby("Customer ID")["_revenue"].sum() / hist.groupby("Customer ID")["Invoice"].nunique()
    g["avg_order_value"] = g["Customer ID"].map(orders)
    g["purchase_frequency"] = g["frequency"]
    g = g.sort_values("Customer ID").reset_index(drop=True)
    g["customer_id"] = g["Customer ID"]
    return g
