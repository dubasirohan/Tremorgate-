#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve,auc,accuracy_score,precision_recall_fscore_support
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--pred",required=True);ap.add_argument("--out",default="plots");args=ap.parse_args()
    p=Path(args.out);p.mkdir(exist_ok=True)
    z=np.load(args.pred);y=z["y"];s=z["p"];pred=(s>=.5).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    print({"accuracy":accuracy_score(y,pred),"precision":pr,"recall":re,"f1":f1,
           "confusion":confusion_matrix(y,pred).tolist()})
    fpr,tpr,_=roc_curve(y,s);plt.figure();plt.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.3f}");plt.plot([0,1],[0,1],"--")
    plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend();plt.tight_layout();plt.savefig(p/"roc.png",dpi=180);plt.close()
    ConfusionMatrixDisplay(confusion_matrix(y,pred),display_labels=["voluntary-proxy","tremor-proxy"]).plot()
    plt.tight_layout();plt.savefig(p/"confusion.png",dpi=180);plt.close()
if __name__=="__main__":main()
