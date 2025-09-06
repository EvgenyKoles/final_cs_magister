# -*- coding: utf-8 -*-
import json, argparse, pathlib
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="artifacts/metrics", help="где лежат *_eval.json")
    ap.add_argument("--out_csv", default="artifacts/metrics/model_comparison.csv")
    ap.add_argument("--out_md",  default="artifacts/metrics/model_comparison.md")
    args = ap.parse_args()

    mdir = pathlib.Path(args.metrics_dir)
    files = sorted(mdir.glob("*_eval.json"))
    if not files:
        raise SystemExit(f"Не найдено JSON-файлов в {mdir}")

    rows = []
    for fp in files:
        try:
            js = json.loads(fp.read_text())
        except Exception as e:
            print(f"[WARN] пропускаю {fp.name}: {e}")
            continue

        name = js.get("model", fp.stem.replace("_eval",""))
        thr  = js.get("threshold", js.get("val", {}).get("Thr", None))
        val  = js.get("val", {}) if isinstance(js.get("val", {}), dict) else {}
        test = js.get("test", {}) if isinstance(js.get("test", {}), dict) else {}

        def g(d, k):
            return float(d.get(k)) if k in d and d.get(k) is not None else np.nan

        rows.append({
            "model": name,
            "thr": thr,
            "val_PR_AUC":  g(val,  "PR_AUC"),
            "val_ROC_AUC": g(val,  "ROC_AUC"),
            "val_F1":      g(val,  "F1"),
            "val_Recall":  g(val,  "Recall@Thr"),
            "val_Brier":   g(val,  "Brier"),
            "test_PR_AUC":  g(test, "PR_AUC"),
            "test_ROC_AUC": g(test, "ROC_AUC"),
            "test_F1":      g(test, "F1"),
            "test_Recall":  g(test, "Recall@Thr"),
            "test_Brier":   g(test, "Brier"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["test_PR_AUC", "val_PR_AUC"], ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # ранг

    out_csv = pathlib.Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md  = pathlib.Path(args.out_md);  out_md.parent.mkdir(parents=True,  exist_ok=True)

    df.to_csv(out_csv, index=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Model comparison (sorted by test PR-AUC)\n\n")
        f.write(df.to_markdown())

    print("\n=== Model comparison (sorted by test PR-AUC) ===")
    print(df[["model","test_PR_AUC","val_PR_AUC","test_Recall","val_Recall","test_F1","val_F1","test_ROC_AUC","val_ROC_AUC","test_Brier","val_Brier","thr"]].to_string())
    print(f"\n[OK] Сохранено: {out_csv} и {out_md}")

if __name__ == "__main__":
    main()
