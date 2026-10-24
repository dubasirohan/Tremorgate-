#!/usr/bin/env python3
import argparse,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument("file");a=ap.parse_args()
df=pd.read_csv(a.file)
print("shape",df.shape);print("columns:");print("\n".join(map(str,df.columns)))
print(df.head())
print(df.dtypes)
