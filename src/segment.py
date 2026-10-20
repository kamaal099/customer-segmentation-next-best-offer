"""Deterministic KMeans model selection, profiles and naming."""
from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class Segmenter:
    feature_columns = ["recency","frequency","monetary","total_quantity","unique_products","avg_order_value"]
    def __init__(self, n_segments=3, random_state=42): self.n_segments=n_segments; self.random_state=random_state
    def fit(self, table):
        for c in self.feature_columns:
            if c not in table: table[c] = table["frequency"] if c == "purchase_frequency" else 0
        self.scaler=StandardScaler().fit(table[self.feature_columns]); x=self.scaler.transform(table[self.feature_columns])
        self.model=KMeans(n_clusters=self.n_segments, random_state=self.random_state, n_init=20).fit(x); return self
    def predict(self, table): return self.model.predict(self.scaler.transform(table[self.feature_columns]))
    def fit_predict(self, table):
        out=table.copy(); self.fit(table); out["segment"]=self.predict(table); return out
    def compare(self, table, min_k=2, max_k=10):
        x=StandardScaler().fit_transform(table[self.feature_columns]); rows=[]
        for k in range(min_k, min(max_k,len(table))+1):
            labels=KMeans(k, random_state=self.random_state, n_init=20).fit_predict(x)
            rows.append({"k":k,"silhouette":silhouette_score(x,labels),"calinski_harabasz":calinski_harabasz_score(x,labels),"davies_bouldin":davies_bouldin_score(x,labels)})
        result = pd.DataFrame(rows)
        result["stability"] = [self._stability(x, k) for k in result.k]
        return result

    def _stability(self, x, k, repeats=5):
        labels = [KMeans(k, random_state=self.random_state+i, n_init=10).fit_predict(x) for i in range(repeats)]
        return float(np.mean([np.mean(labels[i] == labels[0]) for i in range(1, repeats)]))

    def compare_models(self, table, k=None):
        """Compare KMeans, GMM, and Ward using common validation metrics."""
        k = k or self.n_segments
        x = StandardScaler().fit_transform(table[self.feature_columns])
        models = {
            "kmeans": KMeans(k, random_state=self.random_state, n_init=20),
            "gmm": GaussianMixture(k, random_state=self.random_state),
            "ward": AgglomerativeClustering(n_clusters=k),
        }
        rows = []
        for name, model in models.items():
            labels = model.fit_predict(x)
            rows.append({"model": name, "silhouette": silhouette_score(x, labels),
                         "calinski_harabasz": calinski_harabasz_score(x, labels),
                         "davies_bouldin": davies_bouldin_score(x, labels)})
        return pd.DataFrame(rows)
    def profile(self, assigned):
        p=assigned.groupby("segment")[self.feature_columns].mean().reset_index()
        p["segment_name"]=p.apply(lambda r: "High value" if r.monetary==p.monetary.max() else ("At risk" if r.recency==p.recency.max() else "Core customers"),axis=1)
        return p

RFMSegmenter = Segmenter
