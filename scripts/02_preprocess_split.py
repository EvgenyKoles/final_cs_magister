
import argparse, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/_labeled.csv", help="file from step 1")
    ap.add_argument("--outdir", default="data/processed", help="куда класть X_train/X_val/X_test и y_*")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size",  type=float, default=0.15) 
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" not in df.columns:
        raise SystemExit("Нет столбца 'label'. Сначала запусти 01_build_label.py.")

    # --- min list of parameters ---
    numeric_expected = {
        "age","year","stud_h","psyt","jspe","qcae_cog","qcae_aff",
        "amsp","erec_mean","cesd","stai_t"
    }
    categorical_expected = {"sex","glang","job","health"}

    numeric = [c for c in df.columns if c in numeric_expected]
    categorical = [c for c in df.columns if c in categorical_expected]

    # if something not found, take it anyway
    if not numeric:
        numeric = [c for c in df.columns if c not in {"label"} and pd.api.types.is_numeric_dtype(df[c])]

    X = df[[c for c in numeric + categorical if c in df.columns]].copy()
    y = df["label"].astype(int)

    # delete rows where is empty labels

    data = X.join(y.rename("label"))
    before = len(data)
    data = data.dropna()
    after = len(data)
    if after < before:
        print(f"[INFO] deleted rows: {before - after}")

    X, y = data.drop(columns=["label"]), data["label"]

    # --- split 70/15/15 ---
    X = shuffle(X, random_state=args.seed)
    y = y.loc[X.index]

    # test first
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    # validation
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=args.seed
    )

    print(f"[INFO] partitioning: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # --- preprocessing: OneHot for categores + StandardScaler for numbers ---
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop"
    )
    pipe = Pipeline([("prep", ct)])

    X_train_t = pipe.fit_transform(X_train)
    X_val_t   = pipe.transform(X_val)
    X_test_t  = pipe.transform(X_test)

    feature_names = []
    if numeric:
        feature_names += numeric
    if categorical:
        ohe = pipe.named_steps["prep"].named_transformers_["cat"]
        feature_names += list(ohe.get_feature_names_out(categorical))

    # saving
    np.savetxt(outdir/"X_train.csv", X_train_t, delimiter=",")
    np.savetxt(outdir/"X_val.csv",   X_val_t,   delimiter=",")
    np.savetxt(outdir/"X_test.csv",  X_test_t,  delimiter=",")
    y_train.to_csv(outdir/"y_train.csv", index=False)
    y_val.to_csv(outdir/"y_val.csv",     index=False)
    y_test.to_csv(outdir/"y_test.csv",   index=False)
    pd.Series(feature_names).to_csv(outdir/"feature_names.csv", index=False, header=False)

    # disbalance check
    def distr(s): 
        return dict(zip(*np.unique(s, return_counts=True)))
    print(f"[INFO] Class balance (train): {distr(y_train)} | (val): {distr(y_val)} | (test): {distr(y_test)}")

    print("[OK] data saved:", outdir)

if __name__ == "__main__":
    main()
