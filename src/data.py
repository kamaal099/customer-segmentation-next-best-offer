"""Online Retail II ingestion, normalization and auditable cleaning."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

CANONICAL = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    mapping = {}
    for wanted in CANONICAL:
        key = wanted.lower()
        for alias, original in aliases.items():
            if alias == key or alias.replace(" ", "") == key.replace(" ", ""):
                mapping[original] = wanted; break
    return df.rename(columns=mapping)

def read_online_retail(path: str | Path, uk_only: bool = False) -> pd.DataFrame:
    """Read CSV or every sheet in XLS/XLSX; raise actionable missing-file errors."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Online Retail II file not found: {p}. Download it manually from the UCI repository and pass --input.")
    if p.suffix.lower() == ".csv":
        raw = pd.read_csv(p, encoding="latin1")
    elif p.suffix.lower() in {".xls", ".xlsx"}:
        sheets = pd.read_excel(p, sheet_name=None)
        raw = pd.concat(sheets.values(), ignore_index=True)
    else:
        raise ValueError("Input must be CSV, XLS, or XLSX")
    df = _normalize_columns(raw)
    missing = set(CANONICAL) - set(df.columns)
    if missing:
        raise ValueError(f"Missing Online Retail II columns: {sorted(missing)}")
    for c in ["Quantity", "Price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Customer ID"] = df["Customer ID"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["Invoice"] = df["Invoice"].astype("string").str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["_revenue"] = df["Quantity"] * df["Price"]
    df["_is_cancellation"] = df["Invoice"].str.upper().str.startswith("C")
    df["_is_return"] = df["Quantity"] < 0
    before = len(df)
    returns = df[df["_is_return"] | df["_is_cancellation"]].copy()
    valid = df.dropna(subset=["InvoiceDate", "Customer ID", "StockCode", "Quantity", "Price"])
    valid = valid[valid["Customer ID"].ne("<NA>") & valid["Customer ID"].ne("")]
    valid = valid[~valid["_is_cancellation"] & ~valid["_is_return"]]
    # UCI non-product/service codes are not sellable products.
    valid = valid[~valid["StockCode"].str.upper().str.match(r"^(POST|DOT|M|D|C2|BANK|AMAZON|S)$", na=False)]
    valid = valid[valid["Quantity"] > 0].copy()
    if uk_only:
        valid = valid[valid["Country"].eq("United Kingdom")]
    valid.attrs["returns"] = returns.drop(columns=["_is_cancellation", "_is_return"], errors="ignore")
    valid.attrs["cleaning_log"] = {
        "rows_before": int(before), "rows_after": int(len(valid)),
        "rows_removed": int(before-len(valid)),
        "revenue_before": float(df["_revenue"].sum(min_count=1)),
        "revenue_after": float(valid["_revenue"].sum(min_count=1)),
        "cancellations": int(df["_is_cancellation"].sum()), "returns": int(df["_is_return"].sum()),
        "return_revenue": float(returns["_revenue"].sum(min_count=1)),
        "return_rows": int(len(returns)),
        "uk_only": uk_only,
    }
    return valid.drop(columns=["_is_cancellation", "_is_return"])

def load_transactions(path: str | Path, uk_only: bool = False, cache: str | Path | None = None) -> pd.DataFrame:
    if cache and Path(cache).exists():
        return pd.read_parquet(cache)
    df = read_online_retail(path, uk_only)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        _out = df.copy()
        _out.attrs = {}
        _out.to_parquet(cache, index=False)
    return df

def clean_transactions(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compatibility cleaner for canonical frames and the original tiny fixture."""
    if {"customer_id", "transaction_date", "amount"}.issubset(frame.columns):
        out = frame.rename(columns={"customer_id":"Customer ID","transaction_date":"InvoiceDate","amount":"_revenue","product_id":"StockCode"}).copy()
        out["Invoice"] = out.groupby("Customer ID").cumcount().astype(str)
        out["Quantity"] = 1
        out["Price"] = out["_revenue"]
        out["Description"] = ""
        out["Country"] = ""
        out["product_id"] = out["StockCode"]
        out["customer_id"] = out["Customer ID"]
        return out
    return read_online_retail(frame, **kwargs) if isinstance(frame, (str, Path)) else frame
