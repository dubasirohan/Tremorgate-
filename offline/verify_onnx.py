#!/usr/bin/env python3
import argparse
import numpy as np, torch, onnxruntime as ort
from train_cnn import CNN1D
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--pt",required=True);ap.add_argument("--onnx",required=True)
    ap.add_argument("--norm",required=True);args=ap.parse_args()
    m=CNN1D();m.load_state_dict(torch.load(args.pt,map_location="cpu"));m.eval()
    x=np.random.default_rng(0).normal(size=(4,6,200)).astype(np.float32)
    with torch.no_grad():a=m(torch.from_numpy(x)).numpy()
    s=ort.InferenceSession(args.onnx,providers=["CPUExecutionProvider"]);b=s.run(None,{s.get_inputs()[0].name:x})[0].reshape(-1)
    print("max_abs_logit_error",float(np.max(np.abs(a-b))))
    assert np.allclose(a,b,rtol=1e-3,atol=1e-4)
if __name__=="__main__":main()
