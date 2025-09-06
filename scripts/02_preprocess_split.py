# -*- coding: utf-8 -*-
import argparse, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/_labeled.csv", help="входной размеченный файл из шага 1")
    ap.add_argument("--outdir", default="data/processed", help="куда класть X_train/X_val/X_test и y_*")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size",  type=float, default=0.15)  # итоговая доля в полном наборе
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" not in df.columns:
        raise SystemExit("Нет столбца 'label'. Сначала запусти 01_build_label.py.")

    # --- минимальный список ожидаемых признаков (возьмем то, что реально присутствует) ---
    numeric_expected = {
        "age","year","stud_h","psyt","jspe","qcae_cog","qcae_aff",
        "amsp","erec_mean","cesd","stai_t"
    }
    categorical_expected = {"sex","glang","job","health"}

    numeric = [c for c in df.columns if c in numeric_expected]
    categorical = [c for c in df.columns if c in categorical_expected]

    # если что-то названо иначе — всё равно работаем: авто-добавим числовые столбцы по типу
    if not numeric:
        numeric = [c for c in df.columns if c not in {"label"} and pd.api.types.is_numeric_dtype(df[c])]

    X = df[[c for c in numeric + categorical if c in df.columns]].copy()
    y = df["label"].astype(int)

    # простая чистка: удалим строки, где есть пропуски в выбранных фичах/лейбле
    data = X.join(y.rename("label"))
    before = len(data)
    data = data.dropna()
    after = len(data)
    if after < before:
        print(f"[INFO] Удалено строк с пропусками: {before - after}")

    X, y = data.drop(columns=["label"]), data["label"]

    # --- стратифицированный сплит 70/15/15 ---
    X = shuffle(X, random_state=args.seed)
    y = y.loc[X.index]

    # сначала test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    # доля валидации из оставшегося куска
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=args.seed
    )

    print(f"[INFO] Разбиение: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # --- препроцессинг: OneHot для категорий + StandardScaler для чисел ---
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

    # имена фич после трансформации (по желанию — пригодится позже)
    feature_names = []
    if numeric:
        feature_names += numeric
    if categorical:
        ohe = pipe.named_steps["prep"].named_transformers_["cat"]
        feature_names += list(ohe.get_feature_names_out(categorical))

    # сохраняем
    np.savetxt(outdir/"X_train.csv", X_train_t, delimiter=",")
    np.savetxt(outdir/"X_val.csv",   X_val_t,   delimiter=",")
    np.savetxt(outdir/"X_test.csv",  X_test_t,  delimiter=",")
    y_train.to_csv(outdir/"y_train.csv", index=False)
    y_val.to_csv(outdir/"y_val.csv",     index=False)
    y_test.to_csv(outdir/"y_test.csv",   index=False)
    pd.Series(feature_names).to_csv(outdir/"feature_names.csv", index=False, header=False)

    # быстрый чек дисбаланса
    def distr(s): 
        return dict(zip(*np.unique(s, return_counts=True)))
    print(f"[INFO] Баланс классов (train): {distr(y_train)} | (val): {distr(y_val)} | (test): {distr(y_test)}")

    print("[OK] Данные сохранены в:", outdir)

if __name__ == "__main__":
    main()
