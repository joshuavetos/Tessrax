"""
Tessrax Anchor Service
Posts Merkle roots to public timestamping networks (OpenTimestamps + IPFS fallback).
"""

import os, json, time, hashlib, requests, subprocess
from core.ledger.merkle_api import build_ledger_cache, merkle_root

ANCHOR_LOG = "logs/anchor_log.jsonl"
OTS_SERVER = "https://a.pool.opentimestamps.org"
IPFS_API = "https://ipfs.io/api/v0/add"

def sha(x:str)->str:
    return hashlib.sha256(x.encode()).hexdigest()

def anchor_to_opentimestamps(root:str)->dict:
    """Create an OpenTimestamps proof (requires ots client in PATH)."""
    fname=f"/tmp/{root}.ots"
    with open(fname,"w") as f: f.write(root)
    try:
        subprocess.run(["ots", "stamp", fname], check=True)
        return {"method":"OpenTimestamps","proof":fname}
    except Exception as e:
        return {"method":"OpenTimestamps","error":str(e)}

def anchor_to_ipfs(root:str)->dict:
    """Store Merkle root as small text object on IPFS."""
    try:
        r=requests.post(IPFS_API, files={"file":(f"{root}.txt",root)})
        cid=r.json().get("Hash")
        return {"method":"IPFS","cid":cid}
    except Exception as e:
        return {"method":"IPFS","error":str(e)}

def anchor_today():
    ledger=build_ledger_cache()
    root=merkle_root(ledger)
    timestamp=int(time.time())
    record={"root":root,"timestamp":timestamp}
    proofs=[]
    proofs.append(anchor_to_opentimestamps(root))
    proofs.append(anchor_to_ipfs(root))
    record["proofs"]=proofs
    os.makedirs(os.path.dirname(ANCHOR_LOG),exist_ok=True)
    with open(ANCHOR_LOG,"a") as f: f.write(json.dumps(record)+"\n")
    print(f"Anchored root {root[:12]}… at {time.ctime(timestamp)}")
    for p in proofs: print(f"  {p['method']}: {p.get('cid',p.get('proof'))}")

if __name__=="__main__":
    anchor_today()