# core/pipeline/base_pipeline.py
import pandas as pd, requests, hashlib, json, datetime

def sha256(x:str): return hashlib.sha256(x.encode()).hexdigest()

def fetch_csv(url:str)->pd.DataFrame:
    df=pd.read_csv(url)
    df["source_hash"]=sha256(url)
    df["timestamp"]=datetime.datetime.utcnow().isoformat()
    return df

def normalize(df:pd.DataFrame, mappings:dict)->list:
    """Map external columns to internal Tessrax schema fields."""
    records=[]
    for _,row in df.iterrows():
        rec={k:row[v] for k,v in mappings.items() if v in row}
        rec["source_hash"]=sha256(json.dumps(rec,sort_keys=True))
        records.append(rec)
    return records