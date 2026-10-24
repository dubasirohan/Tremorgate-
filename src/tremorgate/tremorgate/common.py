import json, math, time
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt, welch

CSV_NAMES = ["timestamp_ms","flex1","flex2","flex3","flex4","flex5",
             "ax_g","ay_g","az_g","gx_rad_s","gy_rad_s","gz_rad_s"]

def parse_csv_line(line):
    p = [x.strip() for x in line.strip().split(",")]

    if len(p) != 12:
        raise ValueError(f"expected 12 CSV fields, got {len(p)}")

    if p[0] == "time_ms":
        raise ValueError("CSV header")

    vals = [float(x) for x in p]

    if not np.all(np.isfinite(vals)):
        raise ValueError("non-finite value")

    if vals[11] != 1:
        raise ValueError("IMU sample invalid")

    return vals

def bandpower_4_8hz(x, fs=100.0):
    x=np.asarray(x,dtype=np.float64)
    x=x-x.mean(axis=0,keepdims=True)
    freqs, psd=welch(x,fs=fs,nperseg=min(128,len(x)),axis=0)
    mask=(freqs>=4.0)&(freqs<=8.0)
    return float(np.trapezoid(psd[mask],freqs[mask],axis=0).mean())

class HysteresisDwell:
    """Binary tremor state from a probability/score.
    Enter tremor after enter_dwell consecutive positive samples.
    Leave tremor after exit_dwell consecutive samples below exit threshold.
    """
    def __init__(self, enter=0.65, exit=0.45, enter_dwell=2, exit_dwell=2):
        if not 0 <= exit < enter <= 1:
            raise ValueError("need 0 <= exit < enter <= 1")
        self.enter=enter; self.exit=exit
        self.enter_dwell=int(enter_dwell); self.exit_dwell=int(exit_dwell)
        self.state=False; self.hi=0; self.lo=0
    def update(self, p):
        if not self.state:
            self.lo=0
            self.hi=self.hi+1 if p>=self.enter else 0
            if self.hi>=self.enter_dwell:
                self.state=True; self.hi=0
        else:
            self.hi=0
            self.lo=self.lo+1 if p<=self.exit else 0
            if self.lo>=self.exit_dwell:
                self.state=False; self.lo=0
        return self.state

def flex_to_arm(flex, mins=None, maxs=None):
    """Map five flex ADC values to two RRBot joint targets.
    Defaults assume normalized [0,1] flex. Calibrate on real glove.
    """
    f=np.asarray(flex,dtype=np.float64)
    if mins is not None and maxs is not None:
        mins=np.asarray(mins); maxs=np.asarray(maxs)
        f=(f-mins)/np.maximum(maxs-mins,1e-6)
    else:
        # ESP32 replay should provide normalized flex values if possible.
        f=np.clip(f/4095.0,0,1)
    q1=float(np.clip(np.mean(f[:3])*1.8-0.9,-1.5,1.5))
    q2=float(np.clip(np.mean(f[3:])*1.8-0.9,-1.5,1.5))
    return [q1,q2]

def json_log(path, obj):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(obj)+"\n")
