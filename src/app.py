from pathlib import Path
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Customer Engine",layout="wide"); st.title("Customer Segmentation + Next-Best-Offer")
root=Path(st.sidebar.text_input("Artifacts directory","artifacts"))
if not (root/"customer_table.parquet").exists(): st.warning("Artifacts not found. Run the pipeline first.")
else:
    a,b,c=st.tabs(["Segments","Recommendations","Metrics"])
    with a: st.dataframe(pd.read_parquet(root/"customer_table.parquet"))
    with b:
        f=root/"recommendations.csv"; st.dataframe(pd.read_csv(f)) if f.exists() else st.info("Recommendations missing.")
    with c:
        f=root/"metrics.json"; st.json(f.read_text()) if f.exists() else st.info("Metrics missing.")
