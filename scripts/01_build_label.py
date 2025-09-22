# -*- coding: utf-8 -*-
import argparse, pathlib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/raw/Data_Carrard_2022_MedTeach.csv", help="our dataset CSV")
    ap.add_argument("--output", default="data/processed/_labeled.csv",            help="where to save the tagged date set")
    ap.add_argument("--t", type=int, default=27, help="T threshold for MBI-EX (default is 27)")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "mbi_ex" not in df.columns:
        raise SystemExit("Column 'mbi_ex' not found.")

    # step 1: construct the target variable label (1 = high risk by MBI-EX >= T)
    T = args.t
    df["label"] = (df["mbi_ex"] >= T).astype(int)

    # EDA
    n = len(df)
    pos = int(df["label"].sum())
    neg = int(n - pos)
    print(f"[INFO] threshold T={T}. Dataset size: {n}. positive (1): {pos} | negative (0): {neg}")

    # saved
    df.to_csv(out_path, index=False)
    print(f"[OK] Tagged dataset saved: {out_path}")

if __name__ == "__main__":
    main()


