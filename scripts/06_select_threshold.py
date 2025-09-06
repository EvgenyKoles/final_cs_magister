# -*- coding: utf-8 -*-
"""
06_select_threshold.py
Подбор порога для XGBoost по Fβ с использованием OOF-предсказаний (Repeated Stratified CV),
затем финальное обучение и оценка на test при найденном пороге.

Требует: artifacts/metrics/xgb_tuning_summary.json (из 05_tune_xgb.py), data/processed/* из 02_preprocess_split.py.

Пример:
  python scripts/06_select_threshold.py --beta 2.0 --cv_folds 5 --cv_repeats 5 \
      --train_on trainval --save_curves
"""

import argparse, json, pathlib, math
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_recall_fscore_support,
    brier_score_loss, precision_recall_curve
)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception as e:
    raise SystemExit("xgboost не установлен: " + str(e))


# ---------- IO ----------
def load_xy(processed_dir: str):
    p = pathlib.Path(processed_dir)
    X_train = np.loadtxt(p/"X_train.csv", delimiter=",")
    X_val   = np.loadtxt(p/"X_val.csv",   delimiter=",")
    X_test  = np.loadtxt(p/"X_test.csv",  delimiter=",")
    y_train = np.loadtxt(p/"y_train.csv", delimiter=",", skiprows=1).astype(int)
    y_val   = np.loadtxt(p/"y_val.csv",   delimiter=",", skiprows=1).astype(int)
    y_test  = np.loadtxt(p/"y_test.csv",  delimiter=",", skiprows=1).astype(int)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_best_config(summary_path: pathlib.Path) -> Dict[str, Any]:
    if not summary_path.exists():
        raise SystemExit(f"Не найдено: {summary_path}")
    data = json.loads(summary_path.read_text())
    cfg = data.get("best_config")
    if not cfg:
        raise SystemExit("В файле нет поля 'best_config'. Убедись, что 05_tune_xgb.py отработал.")
    return cfg


# ---------- Helpers ----------
def neg_pos_stats(y: np.ndarray) -> Tuple[int,int,float]:
    pos = int(y.sum()); neg = int(len(y) - pos)
    spw = (neg / max(pos,1)) if pos>0 else 1.0
    return neg, pos, spw

def build_model(cfg: Dict[str, Any], y_tr: np.ndarray, seed: int) -> XGBClassifier:
    # scale_pos_weight режимы
    _, _, spw_auto = neg_pos_stats(y_tr)
    m = cfg.get("spw_mode", "sqrt")
    if m == "auto":
        spw = spw_auto
    elif m == "sqrt":
        spw = math.sqrt(spw_auto)
    else:
        spw = 1.0  # one

    params = dict(
        n_estimators=int(cfg.get("n_estimators", 1500)),
        learning_rate=float(cfg.get("lr", 0.02)),
        max_depth=int(cfg.get("depth", 3)),
        min_child_weight=float(cfg.get("mcw", 2.0)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample", 0.9)),
        reg_alpha=float(cfg.get("alpha", 0.1)),
        reg_lambda=float(cfg.get("lambda", 2.0)),
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=float(spw),
        tree_method="hist",
        booster=cfg.get("booster", "gbtree"),
        random_state=seed,
        n_jobs=-1
    )
    # поддержка dart-параметров при необходимости
    if params["booster"] == "dart":
        params.update(
            rate_drop=float(cfg.get("rate_drop", 0.1)),
            skip_drop=float(cfg.get("skip_drop", 0.1))
        )
    # никакого ES здесь: порог подбираем на OOF, ES мешает на маленьких валках
    return XGBClassifier(**params)


def metrics_at_thr(y_true, proba, thr):
    y_hat = (proba >= thr).astype(int)
    # базовые метрики
    ap  = average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    f1  = f1_score(y_true, y_hat, zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, average=None, zero_division=0)
    rec_pos = float(rec[1]) if len(rec) > 1 else 0.0
    prec_pos = float(prec[1]) if len(prec) > 1 else 0.0
    brier = brier_score_loss(y_true, proba)
    return {
        "PR_AUC": float(ap), "ROC_AUC": float(roc), "F1": float(f1),
        "Precision@Thr": float(prec_pos), "Recall@Thr": float(rec_pos),
        "Brier": float(brier), "Thr": float(thr)
    }


def fbeta_score_pos(y_true, y_hat, beta=2.0):
    # считаем precision/recall только для положительного класса
    pr, rc, _, _ = precision_recall_fscore_support(y_true, y_hat, average=None, zero_division=0)
    if len(pr) <= 1:
        return 0.0
    p, r = float(pr[1]), float(rc[1])
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--summary_path", default="artifacts/metrics/xgb_tuning_summary.json")
    ap.add_argument("--beta", type=float, default=2.0, help="β для Fβ (приоритет recall при β>1)")
    ap.add_argument("--thr_grid", default="0.01:0.60:0.005", help="min:max:step для поиска порога")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--cv_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_on", choices=["train","trainval"], default="trainval",
                    help="на чем учим финальную модель после выбора порога")
    ap.add_argument("--save_curves", action="store_true")
    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)
    cfg = load_best_config(pathlib.Path(args.summary_path))

    # --- OOF предсказания на train
    rs = np.random.RandomState(args.seed)
    rskf = RepeatedStratifiedKFold(n_splits=args.cv_folds, n_repeats=args.cv_repeats, random_state=args.seed)
    oof_proba = np.zeros_like(y_train, dtype=float)
    oof_mask  = np.zeros_like(y_train, dtype=bool)

    for rep_fold, (idx_tr, idx_va) in enumerate(rskf.split(X_train, y_train), 1):
        model = build_model(cfg, y_train[idx_tr], seed=args.seed + rep_fold)
        model.fit(X_train[idx_tr], y_train[idx_tr])
        oof_proba[idx_va] = model.predict_proba(X_train[idx_va])[:, 1]
        oof_mask[idx_va]  = True

    assert oof_mask.all(), "Не все OOF-значения посчитаны; проверь CV настройки."
    oof_ap = average_precision_score(y_train, oof_proba)
    print(f"[INFO] OOF PR-AUC on train = {oof_ap:.4f}")

    # --- поиск порога по Fβ
    mn, mx, st = [float(x) for x in args.thr_grid.split(":")]
    grid = np.arange(mn, mx + 1e-9, st)
    best_thr, best_fbeta = None, -1.0
    for thr in grid:
        y_hat = (oof_proba >= thr).astype(int)
        f_b = fbeta_score_pos(y_train, y_hat, beta=args.beta)
        if f_b > best_fbeta:
            best_fbeta, best_thr = f_b, float(thr)

    print(f"[INFO] Best threshold by F{args.beta:.1f} on OOF: thr={best_thr:.3f}, Fβ={best_fbeta:.4f}")

    # --- финальное обучение и оценка на test при найденном пороге
    if args.train_on == "trainval":
        X_final = np.vstack([X_train, X_val]); y_final = np.hstack([y_train, y_val])
    else:
        X_final, y_final = X_train, y_train

    final_model = build_model(cfg, y_final, seed=args.seed + 999)
    final_model.fit(X_final, y_final)

    proba_val = final_model.predict_proba(X_val)[:, 1]
    proba_te  = final_model.predict_proba(X_test)[:, 1]

    m_val  = metrics_at_thr(y_val,  proba_val, best_thr)
    m_test = metrics_at_thr(y_test, proba_te,  best_thr)

    print("\n=== XGB with threshold selected by CV ===")
    print(f"[VAL ] PR-AUC={m_val['PR_AUC']:.4f} | F1={m_val['F1']:.4f} | Prec@{best_thr:.2f}={m_val['Precision@Thr']:.4f} | Recall@{best_thr:.2f}={m_val['Recall@Thr']:.4f}")
    print(f"[TEST] PR-AUC={m_test['PR_AUC']:.4f} | F1={m_test['F1']:.4f} | Prec@{best_thr:.2f}={m_test['Precision@Thr']:.4f} | Recall@{best_thr:.2f}={m_test['Recall@Thr']:.4f}")

    # --- сохранить артефакты
    art = pathlib.Path(args.artifacts_dir); (art/"metrics").mkdir(parents=True, exist_ok=True)
    out = {
        "beta": args.beta,
        "best_threshold": best_thr,
        "oof_train_ap": oof_ap,
        "config": cfg,
        "val": m_val,
        "test": m_test
    }
    (art/"metrics"/"xgb_thresholded_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    if args.save_curves:
        (art/"metrics").mkdir(parents=True, exist_ok=True)
        pr_val = precision_recall_curve(y_val, proba_val)
        pr_tst = precision_recall_curve(y_test, proba_te)
        pd.DataFrame({"precision": pr_val[0], "recall": pr_val[1], "thresholds": np.append(pr_val[2], np.nan)}).to_csv(art/"metrics"/"pr_curve_val.csv", index=False)
        pd.DataFrame({"precision": pr_tst[0], "recall": pr_tst[1], "thresholds": np.append(pr_tst[2], np.nan)}).to_csv(art/"metrics"/"pr_curve_test.csv", index=False)

    print(f"\n[OK] Результаты сохранены: {art/'metrics'/'xgb_thresholded_eval.json'}")
    if args.save_curves:
        print(f"[OK] PR-кривые сохранены в: {art/'metrics'}")


if __name__ == "__main__":
    main()
    