# -*- coding: utf-8 -*-
import pathlib, sys
import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]

def check_split(processed_dir, metadata_dir, split):
    p = pathlib.Path(processed_dir)
    m = pathlib.Path(metadata_dir) / f"subgroups_{split}.csv"

    # данные
    X = np.loadtxt(p / f"X_{split}.csv", delimiter=",")
    y = np.loadtxt(p / f"y_{split}.csv", delimiter=",", skiprows=1).astype(int)

    # метаданные подгрупп: первая строка — заголовок
    df = pd.read_csv(m, header=0)  # явно указываем header
    assert set(["gender", "year", "employment"]).issubset(df.columns), f"{m} -> нет нужных колонок"

    # базовые проверки
    assert len(df) == X.shape[0] == y.shape[0], f"{split}: len(meta)={len(df)} != X={X.shape[0]} или y={y.shape[0]}"
    assert df[["gender","year","employment"]].isna().sum().sum() == 0, f"{split}: NaN в метаданных"

    # типы/значения
    for col in ["gender","year","employment"]:
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"{split}: колонка {col} не числовая (dtype={df[col].dtype})")
        uniq = sorted(df[col].unique().tolist())
        print(f"[{split}] {col}: levels={uniq}")

    # проверка наличии обоих классов в подгруппах (важно для метрик)
    def warn_rare(gname, lv, idx):
        yg = y[idx]
        pos = int(yg.sum())
        n = len(yg)
        if n < 10 or pos == 0 or pos == n:
            print(f"[WARN] {split}/{gname}={lv}: n={n}, positives={pos} -> метрики ROC/PR/F1 будут нестабильны/NaN")

    for gname in ["gender","year","employment"]:
        for lv, idx in df.groupby(gname).groups.items():
            warn_rare(gname, lv, df.index.get_indexer_for(idx))

    print(f"[OK] {split}: всё согласовано. n={X.shape[0]}, d={X.shape[1]}")
    return True

def main():
    processed_dir = "data/processed"
    metadata_dir = "artifacts/metadata"
    ok = True
    for split in SPLITS:
        try:
            check_split(processed_dir, metadata_dir, split)
        except AssertionError as e:
            ok = False; print("[ERROR]", e)
        except Exception as e:
            ok = False; print("[ERROR]", type(e).__name__, e)
    if not ok:
        sys.exit(1)
    print("[DONE] Subgroups integrity check passed for all splits.")

if __name__ == "__main__":
    main()
