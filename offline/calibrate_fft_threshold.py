#!/usr/bin/env python3
import argparse,numpy as np
from scipy.signal import welch
from sklearn.metrics import f1_score,balanced_accuracy_score
def bp(w,fs=100):
    vals=[]
    for c in range(6):
        f,p=welch(w[:,c]-w[:,c].mean(),fs=fs,nperseg=min(128,len(w)))
        m=(f>=4)&(f<=8);vals.append(np.trapezoid(p[m],f[m]))
    return float(np.mean(vals))
ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);args=ap.parse_args()
z=np.load(args.data);X=z["X"];y=z["y"]
s=np.asarray([bp(w) for w in X]); qs=np.quantile(s,np.linspace(.01,.99,199))
best=None
for t in qs:
 p=(s>=t).astype(int);score=balanced_accuracy_score(y,p)
 if best is None or score>best[0]:best=(score,t,f1_score(y,p,zero_division=0))
print({"balanced_accuracy":best[0],"threshold":best[1],"f1":best[2]})
