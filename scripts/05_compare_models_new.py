
# -*- coding: utf-8 -*-
"""
Universal metrics aggregator for IRP model comparisons.

Scans a metrics/ directory for any JSON files (robust to heterogeneous schemas),
normalizes them, and produces a single comparison table.

Usage:
    python 05_compare_models.py --metrics_dir metrics --out_dir reports

Outputs:
    reports/metrics_summary.csv
    reports/metrics_summary.md
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# --------- JSON utilities ---------
def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract one or more top-level JSON objects from raw text.
    Handles cases where files contain concatenated JSON objects without commas.
    """
    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:i+1]
                try:
                    objs.append(json.loads(chunk))
                except json.JSONDecodeError:
                    # Try to clean trailing commas or control chars
                    try:
                        cleaned = re.sub(r",\s*}", "}", chunk)
                        objs.append(json.loads(cleaned))
                    except Exception:
                        # Ignore malformed chunk but continue
                        pass
                start = None
    # Fallback: if nothing parsed, try json.loads whole text
    if not objs:
        try:
            objs = [json.loads(text)]
        except Exception:
            pass
    return objs

# --------- Normalization ---------
def _get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return default if cur is None else cur

def normalize_record(src: Dict[str, Any], filename: str) -> Dict[str, Any]:
    model = src.get("model") or src.get("Model") or "unknown"
    # Threshold info
    selected_threshold = src.get("selected_threshold", src.get("threshold"))
    # Threshold mode
    thr_mode = _get(src, "threshold_selection", "mode", default="fixed" if "threshold" in src else None)
    beta = _get(src, "threshold_selection", "beta")
    target_recall = _get(src, "threshold_selection", "target_recall")

    # SMOTE info
    smote = src.get("smote") or {}
    smote_flag = True if smote else False
    smote_strategy = smote.get("sampling_strategy")
    smote_k = smote.get("k_neighbors") or _get(src, "threshold_selection", "smote_k")

    # Metrics (val/test) — tolerate both "Recall" and "Recall@Thr"
    def grab_metrics(split: str) -> Dict[str, Any]:
        m = src.get(split, {}) or {}
        recall = m.get("Recall", m.get("Recall@Thr"))
        return {
            f"{split}_Precision": m.get("Precision"),
            f"{split}_Recall": recall,
            f"{split}_F1": m.get("F1"),
            f"{split}_PR_AUC": m.get("PR_AUC"),
            f"{split}_ROC_AUC": m.get("ROC_AUC"),
            f"{split}_Brier": m.get("Brier"),
            f"{split}_Thr": m.get("Thr"),
        }

    row = {
        "file": filename,
        "model": model,
        "smote": smote_flag,
        "smote_strategy": smote_strategy,
        "smote_k": smote_k,
        "threshold_mode": thr_mode,
        "beta": beta,
        "target_recall": target_recall,
        "selected_threshold": selected_threshold,
    }
    row.update(grab_metrics("val"))
    row.update(grab_metrics("test"))
    return row

# --------- Main pipeline ---------
def build_table(metrics_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            raw = path.read_text(encoding="latin-1").strip()
        objs = _extract_json_objects(raw)
        if not objs:
            # skip empty/unparseable file
            continue
        for obj in objs:
            records.append(normalize_record(obj, path.name))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # Arrange columns nicely
    preferred = [
        "file","model","smote","smote_strategy","smote_k",
        "threshold_mode","beta","target_recall","selected_threshold",
        "val_PR_AUC","val_ROC_AUC","val_Recall","val_Precision","val_F1","val_Brier",
        "test_PR_AUC","test_ROC_AUC","test_Recall","test_Precision","test_F1","test_Brier",
        "val_Thr","test_Thr",
    ]
    # Ensure all columns exist
    for c in preferred:
        if c not in df.columns:
            df[c] = np.nan

    # Sort for readability: by test_PR_AUC desc, then test_Recall desc
    with np.errstate(invalid="ignore"):
        df_sorted = df.sort_values(
            by=["test_PR_AUC","test_Recall","val_PR_AUC"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    # Round numeric columns
    num_cols = [c for c in df_sorted.columns if df_sorted[c].dtype.kind in "fc"]
    df_sorted[num_cols] = df_sorted[num_cols].round(4)

    return df_sorted[preferred]

def save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_summary.csv"
    md_path = out_dir / "metrics_summary.md"
    df.to_csv(csv_path, index=False)
    # Markdown table
    md = df.to_markdown(index=False)
    md_path.write_text(md, encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")

    # Also print quick leaders
    if not df.empty:
        print("\nTop by test PR_AUC:")
        print(df.sort_values("test_PR_AUC", ascending=False).head(5).to_string(index=False))
        print("\nTop by test Recall:")
        print(df.sort_values("test_Recall", ascending=False).head(5).to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=Path, default=Path("metrics"), help="Directory with JSON metric files")
    ap.add_argument("--out_dir", type=Path, default=Path("reports"), help="Where to write summary files")
    args = ap.parse_args()

    if not args.metrics_dir.exists():
        print(f"[WARN] Metrics dir not found: {args.metrics_dir}")
        sys.exit(0)

    df = build_table(args.metrics_dir)
    if df.empty:
        print("[INFO] No metrics found or could not parse JSON files.")
        sys.exit(0)

    save_outputs(df, args.out_dir)

if __name__ == "__main__":
    main()
