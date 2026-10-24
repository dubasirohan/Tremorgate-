#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument("pads");a=ap.parse_args()
p=Path(a.pads); obs=sorted((p/"movement").glob("observation_*.json"))
print("observations",len(obs))
if obs:
  j=json.loads(obs[0].read_text())
  print(json.dumps(j,indent=2)[:12000])
print("patient examples:")
for q in sorted((p/"patients").glob("patient_*.json"))[:3]:
 print(q, q.read_text()[:1000])
