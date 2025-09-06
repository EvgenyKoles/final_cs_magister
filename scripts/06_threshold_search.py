# -*- coding: utf-8 -*-
import argparse, json, pathlib, numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ---------- utils ----------
def _load_y(path):
    try:
        return np.loadtxt(path, delimiter=",").astype(int)
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=1).astype(int)

def load_xy(processed_dir):
    p = pathlib.Path(processed_dir)
    X_train = np.loadtxt(p/"X_train.csv", delimiter=",")
    X_val   = np.loadtxt(p/"X_val.csv",   delimiter=",")
    X_test  = np.loadtxt(p/"X_test.csv",  delimiter=",")
    y_train = _load_y(p/"y_train.csv")
    y_val   = _load_y(p/"y_val.csv")
    y_test  = _load_y(p/"y_test.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def f_beta(prec, rec, beta):
    if prec == 0 and rec == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * prec + rec
    return (1 + b2) * prec * rec / denom if denom > 0 else 0.0

def metrics_from_probs(y_true, probs, thr):
    y_hat = (probs >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    roc    = roc_auc_score(y_true, probs)            if len(np.unique(y_true)) > 1 else float("nan")
    brier  = brier_score_loss(y_true, probs)
    return {
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc),
        "Brier": float(brier),
        "Thr": float(thr),
    }

# ---------- models ----------
def get_model(name, y_train, seed=42, scale_pos_weight=None):
    name = name.lower()
    if name == "lr":
        return LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=seed
        )
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=seed)
    if name == "xgb":
        # импортируем лениво — чтобы не требовать xgboost при других моделях
        from xgboost import XGBClassifier
        # auto scale_pos_weight по train, если не задан явно
        if scale_pos_weight is None:
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            scale_pos_weight = max(1, neg // max(1, pos))  # грубая оценка #neg/#pos
        return XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=seed, n_jobs=4
        )
    raise ValueError("unknown model: " + name)

# ---------- threshold search ----------
def grid_thresholds(probs, resolution=1001):
    # равномерная сетка [0,1]; можно заменить на уникальные значения probs при желании
    return np.linspace(0.0, 1.0, resolution)

def select_thr_by_recall(y_true, probs, target_recall=0.80):
    for thr in grid_thresholds(probs):
        y_hat = (probs >= thr).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        if rec >= target_recall:
            return float(thr)
    # fallback: thr с максимальным recall
    best_thr = 0.0
    best_rec = -1.0
    for thr in grid_thresholds(probs):
        y_hat = (probs >= thr).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        if rec > best_rec:
            best_rec = rec
            best_thr = thr
    return float(best_thr)

def select_thr_by_fbeta(y_true, probs, beta=2.0):
    best_thr, best_fb = 0.5, -1.0
    for thr in grid_thresholds(probs):
        y_hat = (probs >= thr).astype(int)
        prec, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        fb = f_beta(prec, rec, beta)
        if fb > best_fb:
            best_fb = fb
            best_thr = thr
    return float(best_thr), float(best_fb)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lr", "rf", "xgb", "mlp"], help="модель для обучения")
    ap.add_argument("--processed_dir", default="data/processed", help="где лежат X_*/y_*")
    ap.add_argument("--artifacts_dir", default="artifacts", help="куда складывать json-метрики")
    ap.add_argument("--mode", choices=["recall", "fbeta"], default="recall", help="критерий выбора порога на валидации")
    ap.add_argument("--target_recall", type=float, default=0.80, help="целевой recall для mode=recall")
    ap.add_argument("--beta", type=float, default=2.0, help="beta для F_beta при mode=fbeta")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--xgb_scale_pos_weight", type=float, default=None, help="если задано — переопределяет авто-расчет")
    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)

    model = get_model(args.model, y_train, seed=args.seed, scale_pos_weight=args.xgb_scale_pos_weight)
    model.fit(X_train, y_train)

    # вероятности положительного класса
    if hasattr(model, "predict_proba"):
        p_val  = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
    else:
        # на случай моделей без predict_proba (не ожидается, но оставим безопасный путь)
        scores_val  = model.decision_function(X_val)
        scores_test = model.decision_function(X_test)
        # ранговая нормализация в [0,1]
        r_val  = scores_val.argsort().argsort().astype(float);  p_val  = r_val  / (len(r_val)  - 1 + 1e-9)
        r_test = scores_test.argsort().argsort().astype(float); p_test = r_test / (len(r_test) - 1 + 1e-9)

    # выбор порога по валидации
    sel_info = {"mode": args.mode}
    if args.mode == "recall":
        thr = select_thr_by_recall(y_val, p_val, target_recall=args.target_recall)
        sel_info["target_recall"] = args.target_recall
    else:
        thr, best_fb = select_thr_by_fbeta(y_val, p_val, beta=args.beta)
        sel_info["beta"] = args.beta
        sel_info["val_best_Fbeta"] = best_fb

    # метрики на val/test при выбранном пороге
    m_val  = metrics_from_probs(y_val,  p_val,  thr)
    m_test = metrics_from_probs(y_test, p_test, thr)

    print(f"\n[{args.model.upper()}] Selected threshold on VAL: {thr:.3f}  ({sel_info})")
    print("[VAL ]", m_val)
    print("[TEST]", m_test)

    # сохранить артефакт
    out_dir = pathlib.Path(args.artifacts_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "threshold_selection": sel_info,
        "selected_threshold": float(thr),
        "val":  m_val,
        "test": m_test,
    }
    (out_dir / f"{args.model}_thresholded_{args.mode}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[OK] Saved: {out_dir / f'{args.model}_thresholded_{args.mode}.json'}")

if __name__ == "__main__":
    main()
