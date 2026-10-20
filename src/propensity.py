"""Propensity models and exact post-cutoff labels."""
from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

def make_labels(transactions, customers, cutoff, horizon_days=90):
    end=pd.Timestamp(cutoff)+pd.Timedelta(days=horizon_days)
    future=transactions[(pd.to_datetime(transactions.InvoiceDate)>pd.Timestamp(cutoff))&(pd.to_datetime(transactions.InvoiceDate)<=end)]
    bought=future.groupby("Customer ID")["StockCode"].agg(lambda x:set(x.astype(str)))
    products=sorted(transactions.StockCode.astype(str).unique())
    return pd.DataFrame([{"Customer ID":c,"StockCode":p,"label":int(p in bought.get(c,set()))} for c in customers for p in products])

def fit_propensity(transactions, assignments, cutoff, smoothing=1.0):
    hist=transactions[pd.to_datetime(transactions.InvoiceDate)<=pd.Timestamp(cutoff)]
    j=hist.merge(assignments[["Customer ID","segment"]],on="Customer ID")
    products=sorted(hist.StockCode.astype(str).unique()); rows=[]
    for seg in sorted(assignments.segment.unique()):
        counts=j.loc[j.segment==seg,"StockCode"].astype(str).value_counts(); total=counts.sum()
        for p in products: rows.append({"segment":seg,"product_id":p,"propensity":(counts.get(p,0)+smoothing)/(total+smoothing*len(products))})
    return pd.DataFrame(rows)

class PropensityModel:
    def fit(self,X,y): self.model=LogisticRegression(max_iter=500).fit(X,y); return self
    def predict_proba(self,X): return self.model.predict_proba(X)[:,1]
    def metrics(self,X,y):
        p=self.predict_proba(X)
        return {"roc_auc":float(roc_auc_score(y,p)) if len(set(y))>1 else None,"log_loss":float(log_loss(y,p))}
