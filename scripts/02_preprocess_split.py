# -*- coding: utf-8 -*-
"""
02_preprocess_split.py

Подготовка признаков и стратифицированный сплит 70/15/15.
Сохраняет:
- X_train.csv, X_val.csv, X_test.csv  (OHE + стандартизация)
- y_train.csv, y_val.csv, y_test.csv
- feature_names.csv
- subgroups_{train,val,test}.csv  (колонки: gender, year, employment) в:
    artifacts/metadata/  и дублирует в  data/processed/

Совместимо с последующими скриптами 03/04/05/06.
"""

import argparse, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/_labeled.csv", help="файл из шага 01")
    ap.add_argument("--outdir", default="data/processed", help="куда класть X_*/y_* и feature_names")
    ap.add_argument("--artifacts_dir", default="artifacts", help="куда класть subgroups_* для агрегатора 06")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size",  type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    art_meta = pathlib.Path(args.artifacts_dir) / "metadata"; art_meta.mkdir(parents=True, exist_ok=True)

    # ---- load
    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" not in df.columns:
        raise SystemExit("Нет столбца 'label'. Сначала запусти 01_build_label.py.")

    # ---- ожидаемые признаки
    numeric_expected = {
        "age","year","stud_h","psyt","jspe","qcae_cog","qcae_aff",
        "amsp","erec_mean","cesd","stai_t"
    }
    categorical_expected = {"sex","glang","job","health"}

    numeric = [c for c in df.columns if c in numeric_expected]
    categorical = [c for c in df.columns if c in categorical_expected]

    # если список пуст, fallback: все числовые, кроме label
    if not numeric:
        numeric = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    # категориальные оставим только из известного набора, чтобы избежать взрыва OHE
    categorical = [c for c in categorical if c in df.columns]

    # ---- формируем X, y
    used_cols = [c for c in numeric + categorical if c in df.columns]
    X = df[used_cols].copy()
    y = df["label"].astype(int)

    # ---- чистка: удаляем строки с пропусками по используемым признакам/лейблу
    data = X.join(y.rename("label"))
    before = len(data)
    data = data.dropna()
    if len(data) < before:
        print(f"[INFO] удалены строки из-за NaN: {before - len(data)}")

    X, y = data.drop(columns=["label"]), data["label"]

    # ---- сплит 70/15/15 (стратифицированный)
    X = shuffle(X, random_state=args.seed)   # фиксируем порядок для воспроизводимости
    y = y.loc[X.index]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=args.seed
    )

    print(f"[INFO] partitioning: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    def distr(s): 
        v, c = np.unique(s, return_counts=True); return {int(k): int(vv) for k, vv in zip(v, c)}
    print(f"[INFO] Class balance (train): {distr(y_train)} | (val): {distr(y_val)} | (test): {distr(y_test)}")

    # ---- подгруппы для RQ3: gender/year/employment из «сырых» колонок до OHE
    def extract_subgroups(df_part: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df_part.index)
        # gender
        if "sex" in df_part.columns:
            out["gender"] = df_part["sex"]
        elif "gender" in df_part.columns:
            out["gender"] = df_part["gender"]
        else:
            out["gender"] = pd.NA
        # year of study
        if "year" in df_part.columns:
            out["year"] = df_part["year"]
        elif "study_year" in df_part.columns:
            out["year"] = df_part["study_year"]
        else:
            out["year"] = pd.NA
        # employment
        if "job" in df_part.columns:
            out["employment"] = df_part["job"]
        elif "employment" in df_part.columns:
            out["employment"] = df_part["employment"]
        else:
            out["employment"] = pd.NA

        # приведение к строкам, чтобы значения стабильно группировались
        for c in ["gender","year","employment"]:
            out[c] = out[c].astype(str)
        return out.reset_index(drop=True)

    sub_train = extract_subgroups(X_train)
    sub_val   = extract_subgroups(X_val)
    sub_test  = extract_subgroups(X_test)

    # ---- препроцессинг: StandardScaler (num) + OneHotEncoder (cat)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # совместимость со старыми sklearn: параметр назывался sparse
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [c for c in numeric if c in X_train.columns]),
            ("cat", ohe, [c for c in categorical if c in X_train.columns]),
        ],
        remainder="drop"
    )
    pipe = Pipeline([("prep", ct)])

    X_train_t = pipe.fit_transform(X_train)
    X_val_t   = pipe.transform(X_val)
    X_test_t  = pipe.transform(X_test)

    # ---- имена фич
    feature_names = []
    if numeric:
        feature_names += [c for c in numeric if c in X_train.columns]
    if categorical:
        ohe_step = pipe.named_steps["prep"].named_transformers_.get("cat", None)
        if ohe_step is not None:
            cats_cols = [c for c in categorical if c in X_train.columns]
            feature_names += list(ohe_step.get_feature_names_out(cats_cols))

    # ---- сохранение матриц/лейблов/имен
    np.savetxt(outdir/"X_train.csv", X_train_t, delimiter=",")
    np.savetxt(outdir/"X_val.csv",   X_val_t,   delimiter=",")
    np.savetxt(outdir/"X_test.csv",  X_test_t,  delimiter=",")
    y_train.to_csv(outdir/"y_train.csv", index=False)
    y_val.to_csv(outdir/"y_val.csv",     index=False)
    y_test.to_csv(outdir/"y_test.csv",   index=False)
    pd.Series(feature_names).to_csv(outdir/"feature_names.csv", index=False, header=False)

    # ---- сохранение подгрупп (в том же порядке, что X_*)
    # в artifacts/metadata (основной путь для 06)
    sub_train.to_csv(art_meta/"subgroups_train.csv", index=False)
    sub_val.to_csv(  art_meta/"subgroups_val.csv",   index=False)
    sub_test.to_csv( art_meta/"subgroups_test.csv",  index=False)
    # дублируем в processed для удобства
    sub_train.to_csv(outdir/"subgroups_train.csv", index=False)
    sub_val.to_csv(  outdir/"subgroups_val.csv",   index=False)
    sub_test.to_csv( outdir/"subgroups_test.csv",  index=False)

    print("[OK] data saved:", outdir)
    print("[OK] subgroups saved:", art_meta)

if __name__ == "__main__":
    main()
