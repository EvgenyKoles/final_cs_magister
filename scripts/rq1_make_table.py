# -*- coding: utf-8 -*-
"""
RQ1 summary table:
Собирает PR-AUC, ROC-AUC, F1, Recall@Thr для валид. и теста из *_eval.json
и сохраняет один CSV-файл: artifacts/metrics/rq1_summary.csv
"""

import argparse, json, pathlib, sys
import pandas as pd

HUMAN_NAMES = {
    "logistic_regression": "Logistic Regression",
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "xgb_tuned": "XGBoost (tuned)",
    "mlp": "Artificial Neural Network",
}

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def model_label(d):
    # используем 'model' и/или 'variant'
    m = d.get("model", "")
    v = d.get("variant", "")
    key = (v or m or "").lower()
    name = HUMAN_NAMES.get(key, HUMAN_NAMES.get(m, m))
    return name or "Model"

def extract_rows(d, fname):
    rows = []
    name = model_label(d)
    thr_global = d.get("threshold_for_report", None)
    prev = d.get("prevalence", {})
    for split in ["val", "test"]:
        if split not in d:
            continue
        m = d[split] or {}
        row = {
            "Model": name,
            "Split": split.upper(),
            "PR_AUC": round(float(m.get("PR_AUC", float("nan"))), 4) if m.get("PR_AUC") is not None else None,
            "ROC_AUC": round(float(m.get("ROC_AUC", float("nan"))), 4) if m.get("ROC_AUC") is not None else None,
            "F1": round(float(m.get("F1", float("nan"))), 4) if m.get("F1") is not None else None,
            "Recall@Thr": round(float(m.get("Recall@Thr", float("nan"))), 4) if m.get("Recall@Thr") is not None else None,
            "Thr": m.get("Thr", thr_global),
            "Prevalence": round(float(prev.get(split, float("nan"))), 4) if prev else None,
            "Source": fname.name
        }
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="artifacts/metrics", help="где лежат *_eval.json")
    ap.add_argument("--out_csv", default="artifacts/metrics/rq1_summary.csv", help="куда сохранить итоговую таблицу")
    args = ap.parse_args()

    md = pathlib.Path(args.metrics_dir)
    files = sorted(list(md.glob("*_eval.json")))
    if not files:
        print(f"[ERROR] Не найдены *_eval.json в {md}")
        sys.exit(1)

    all_rows = []
    for f in files:
        try:
            d = load_json(f)
            all_rows.extend(extract_rows(d, f))
        except Exception as e:
            print(f"[WARN] Пропускаю {f.name}: {type(e).__name__}: {e}")

    if not all_rows:
        print("[ERROR] Нет валидных метрик для сборки.")
        sys.exit(1)

    # упорядочим: по Split (VAL, TEST), затем по модели
    df = pd.DataFrame(all_rows)
    split_order = {"VAL": 0, "TEST": 1}
    df["split_order"] = df["Split"].map(split_order).fillna(99)
    df.sort_values(["split_order", "Model"], inplace=True)
    df.drop(columns=["split_order"], inplace=True)

    out_path = pathlib.Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] RQ1 summary saved: {out_path}")

    # одновременный «сухой» вывод в консоль
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print("\n=== RQ1 summary (VAL & TEST) ===")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
