import argparse
import pandas as pd
from .data import load_transactions
from .features import derive_cutoff, make_rfm_features
from .segment import Segmenter
from .propensity import fit_propensity
from .recommend import recommend
from .itemitem import recommend_item_item
from .evaluate import evaluate
from .reports import write_reports


def run(input_path, output_dir="artifacts", uk_only=False, segments=3, top_k=3):
    tx = load_transactions(input_path, uk_only=uk_only, cache=f"{output_dir}/transactions.parquet")
    cut = derive_cutoff(tx)
    table = make_rfm_features(tx, cut)
    seg = Segmenter(segments)
    assigned = seg.fit_predict(table)
    prop = fit_propensity(tx, assigned, cut)
    # Leakage-safe: recommenders only see history up to the cutoff.
    history = tx[pd.to_datetime(tx.InvoiceDate) <= pd.Timestamp(cut)]
    rec_seg = recommend(prop, history, assigned, top_k)
    m_seg = evaluate(rec_seg, tx, cut, top_k)
    rec = recommend_item_item(history, assigned, top_k)
    metrics = evaluate(rec, tx, cut, top_k)
    metrics["segment_engine_precision_at_k"] = m_seg["precision_at_k"]
    write_reports(output_dir, assigned, rec, metrics, seg.profile(assigned), seg)
    return metrics


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--input", required=True)
    a.add_argument("--output", default="artifacts")
    a.add_argument("--uk-only", action="store_true")
    a.add_argument("--segments", type=int, default=3)
    a.add_argument("--top-k", type=int, default=3)
    x = a.parse_args()
    print(run(x.input, x.output, x.uk_only, x.segments, x.top_k))
    