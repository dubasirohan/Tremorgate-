#!/usr/bin/env python3
"""Fallback only if CNN training is unusable. Uses spectral features, not raw synthetic data."""
import argparse, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from scipy.signal import welch
def feat(w,fs=100):
    out=[]
    for c in range(6):
        f,p=welch(w[:,c],fs=fs,nperseg=min(128,len(w)))
        bands=[(0.5,3),(3,4),(4,8),(8,15),(15,30)]
        for a,b in bands:
            m=(f>=a)&(f<b);out.append(np.trapezoid(p[m],f[m]) if m.any() else 0)
        out += [w[:,c].mean(),w[:,c].std(),np.sqrt(np.mean(w[:,c]**2))]
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);args=ap.parse_args()
    z=np.load(args.data);X=z["X"];y=z["y"];g=z["groups"];F=np.asarray([feat(w) for w in X])
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(F,y,g))
    clf=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1)
    clf.fit(F[a],y[a]);p=clf.predict_proba(F[b])[:,1]
    print(classification_report(y[b],p>=.5,digits=4));print(confusion_matrix(y[b],p>=.5))
    print("AUC",roc_auc_score(y[b],p))
if __name__=="__main__":main()
