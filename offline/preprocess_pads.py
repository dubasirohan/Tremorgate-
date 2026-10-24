#!/usr/bin/env python3
"""
PADS -> NPZ windows.

Label proxy used by default:
  positive: non-Healthy condition AND tremor-oriented rest/postural task
  negative: Healthy condition AND kinetic task
This is intentionally NOT a clinical tremor label. It is a dataset-level proxy.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

POS_TASKS=("Relaxed","RelaxedTask","StretchHold","HoldWeight","Entrainment")
NEG_TASKS=("DrinkGlas","CrossArms","TouchNose")
POS_CONDS=("PD","Differential","DD","Parkinson")
NEG_CONDS=("Healthy","HC")
def norm_task(s):
    s=s.lower().replace("_","").replace(" ","")
    if s.startswith("relaxedtask"): return "RelaxedTask"
    if s.startswith("relaxed"): return "Relaxed"
    if "stretch" in s: return "StretchHold"
    if "holdweight" in s or "weight" in s: return "HoldWeight"
    if "entrain" in s: return "Entrainment"
    if "drink" in s: return "DrinkGlas"
    if "cross" in s: return "CrossArms"
    if "touch" in s or "nose" in s: return "TouchNose"
    return s
def condition_label(c):
    c=str(c).lower()
    if "healthy" in c: return "Healthy"
    if "parkinson" in c: return "PD"
    if c in ("dd","differential") or "differential" in c: return "DD"
    return c
def load_patient(path):
    with open(path) as f:return json.load(f)
def find_file(pads, fn):
    candidates=[pads/"movement"/"timeseries"/fn,
                pads/"movement"/fn,
                pads/"preprocessed"/"movement"/fn]
    for p in candidates:
        if p.exists(): return p
    return None
def extract_records(obs):
    for sess in obs.get("session",[]):
        task=sess.get("record_name","")
        for rec in sess.get("records",[]):
            loc=rec.get("device_location","")
            ch=rec.get("channels",[])
            fn=rec.get("file_name","")
            if loc and fn and len(ch)>=6:
                yield task,loc,ch,fn
def read_txt(path):
    x=np.loadtxt(path,delimiter=",",ndmin=2)
    if x.shape[1]<6: raise ValueError(f"{path}: expected >=6 columns")
    return x[:,:6].astype(np.float32)
def label_for(cond,task):
    t=norm_task(task); c=condition_label(cond)
    if c in POS_CONDS and t in POS_TASKS:return 1
    if c in NEG_CONDS and t in NEG_TASKS:return 0
    return None
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pads",required=True)
    ap.add_argument("--out",default="pads_windows.npz")
    ap.add_argument("--fs",type=float,default=100.0)
    ap.add_argument("--window",type=int,default=200)
    ap.add_argument("--hop",type=int,default=100)
    ap.add_argument("--cut-sec",type=float,default=.5)
    args=ap.parse_args()
    pads=Path(args.pads)
    X=[]; y=[]; groups=[]; meta=[]
    for op in sorted((pads/"movement").glob("observation_*.json")):
        sid=op.stem.split("_")[-1]
        patient_path=pads/"patients"/f"patient_{sid}.json"
        if not patient_path.exists(): continue
        patient=load_patient(patient_path); cond=patient.get("condition","")
        with open(op) as f: obs=json.load(f)
        srcfs=float(obs.get("sampling_rate",100))
        for task,loc,ch,fn in extract_records(obs):
            if loc.lower()!="rightwrist": continue
            lab=label_for(cond,task)
            if lab is None: continue
            path=find_file(pads,Path(fn).name)
            if path is None:
                # v1 examples use timeseries names; bins are also accepted.
                continue
            try:x=read_txt(path)
            except Exception as e:
                print("SKIP",path,e); continue
            ncut=int(args.cut_sec*srcfs)
            if ncut and len(x)>ncut:x=x[ncut:]
            if abs(srcfs-args.fs)>1e-6:
                # scipy resample_poly for rational ratio
                from fractions import Fraction
                r=Fraction(args.fs/srcfs).limit_denominator(1000)
                x=resample_poly(x,r.numerator,r.denominator,axis=0).astype(np.float32)
            for st in range(0,len(x)-args.window+1,args.hop):
                w=x[st:st+args.window]
                if len(w)!=args.window:continue
                X.append(w); y.append(lab); groups.append(int(sid))
                meta.append((sid,task,loc,cond))
    if not X: raise RuntimeError("No windows found. Inspect PADS file layout and adjust parser.")
    X=np.stack(X); y=np.asarray(y,np.int64); groups=np.asarray(groups,np.int64)
    np.savez_compressed(args.out,X=X,y=y,groups=groups,meta=np.asarray(meta,dtype=str))
    print("saved",args.out,"X",X.shape,"positive",int(y.sum()),"negative",int((y==0).sum()),
          "subjects",len(np.unique(groups)))
if __name__=="__main__":main()
