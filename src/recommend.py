"""Segment propensity recommender."""
import pandas as pd
def recommend(propensity,transactions,assignments,top_k=3,method="segment"):
    seen=transactions.groupby("Customer ID")["StockCode"].agg(lambda x:set(x.astype(str))); rows=[]
    for c,s in assignments[["Customer ID","segment"]].itertuples(index=False):
        cand=propensity[propensity.segment==s].copy()
        if method == "popularity" and "global_propensity" in cand:
            cand["propensity"] = cand["global_propensity"]
        elif method == "hybrid" and "global_propensity" in cand:
            cand["propensity"] = .7*cand["propensity"] + .3*cand["global_propensity"]
        cand=cand[~cand.product_id.astype(str).isin(seen.get(c,set()))]
        for rank,r in enumerate(cand.sort_values(["propensity","product_id"],ascending=[False,True]).head(top_k).itertuples(),1):
            rows.append({"Customer ID":c,"segment":s,"StockCode":r.product_id,"rank":rank,"score":r.propensity,"method":method,"reason":"unseen top segment propensity"})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["customer_id"] = out["Customer ID"]
        out["product_id"] = out["StockCode"]
    return out
