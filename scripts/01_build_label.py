# -*- coding: utf-8 -*-
import argparse, pathlib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/raw/Data_Carrard_2022_MedTeach.csv", help="путь к исходному CSV")
    ap.add_argument("--output", default="data/processed/_labeled.csv",            help="куда сохранить размеченный датасет")
    ap.add_argument("--t", type=int, default=27, help="порог T для MBI-EX (по умолчанию 27)")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "mbi_ex" not in df.columns:
        raise SystemExit("Не найден столбец 'mbi_ex'. Проверь названия колонок в CSV (они чувствительны к символам).")

    # шаг 1: строим целевую переменную label (1 = высокий риск по MBI-EX >= T)
    T = args.t
    df["label"] = (df["mbi_ex"] >= T).astype(int)

    # немного простого EDA
    n = len(df)
    pos = int(df["label"].sum())
    neg = int(n - pos)
    print(f"[INFO] Порог T={T}. Размер датасета: {n}. Положительных (1): {pos} | Отрицательных (0): {neg}")

    # сохраняем
    df.to_csv(out_path, index=False)
    print(f"[OK] Размеченный датасет сохранён: {out_path}")

if __name__ == "__main__":
    main()


