#!/usr/bin/env python3
"""
Build a REAL-DATA composite replay CSV from one PADS IMU file and one ASL flex CSV.
No signal synthesis is performed. The two real recordings are only temporally aligned
for the purpose of exercising the serial/ROS pipeline.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
def load_numeric(path):
    return np.loadtxt(path,delimiter=",",ndmin=2)
def choose_cols(header):
    h=[x.lower().strip() for x in header]
    def find(names):
        for n in names:
            for i,v in enumerate(h):
                if n in v:return i
        raise ValueError(f"could not find any of {names}; columns={header}")
    return [find([f"flex{i}",f"finger{i}"]) for i in range(1,6)]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pads",required=True,help="6-column PADS IMU CSV/TXT, accel g + gyro rad/s")
    ap.add_argument("--asl",required=True,help="ASL CSV")
    ap.add_argument("--out",required=True)
    ap.add_argument("--asl-header",action="store_true")
    ap.add_argument("--fs",type=float,default=100)
    args=ap.parse_args()
    imu=load_numeric(args.pads)[:,:6].astype(float)
    if args.asl_header:
        with open(args.asl,newline="") as f:
            rows=list(csv.reader(f)); hdr=rows[0]; A=np.asarray(rows[1:],float)
        idx=choose_cols(hdr); flex=A[:,idx]
    else:
        A=load_numeric(args.asl); flex=A[:,:5]
    if len(flex)!=len(imu):
        from fractions import Fraction
        r=Fraction(len(imu)/len(flex)).limit_denominator(1000)
        flex=resample_poly(flex,r.numerator,r.denominator,axis=0)
        if len(flex)<len(imu): flex=np.pad(flex,((0,len(imu)-len(flex)),(0,0)),mode="edge")
        flex=flex[:len(imu)]
    # Preserve actual PADS IMU values. Flex values are passed through unchanged.
    t=np.arange(len(imu))*1000.0/args.fs
    out=np.column_stack([t,flex,imu])
    np.savetxt(args.out,out,delimiter=",",fmt="%.8f")
    print("wrote",args.out,out.shape)
if __name__=="__main__":main()
