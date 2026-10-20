from pathlib import Path
import json,pandas as pd
import joblib
def write_reports(output_dir,assignments,recommendations,metrics,profiles=None,model=None):
    p=Path(output_dir); (p/"models").mkdir(parents=True,exist_ok=True)
    a=assignments.copy(); a.attrs={}
    a.to_parquet(p/"customer_table.parquet",index=False); a.to_csv(p/"customer_table.csv",index=False)
    recommendations.to_csv(p/"recommendations.csv",index=False)
    if profiles is not None: profiles.to_csv(p/"segment_profiles.csv",index=False)
    if model is not None: joblib.dump(model, p/"models/segmenter.joblib")
    (p/"metrics.json").write_text(json.dumps({k:v for k,v in metrics.items() if not isinstance(v,pd.DataFrame)},indent=2,default=str),encoding="utf-8")
