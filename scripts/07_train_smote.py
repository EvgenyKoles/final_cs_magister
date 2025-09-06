# -*- coding: utf-8 -*-
# Train with SMOTE + threshold selection on validation (recall or F_beta), then evaluate on test.
import argparse, json, pathlib, numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------- IO ----------
def _load_y(path: pathlib.Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",").astype(int)
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=1).astype(int)

def load_xy(processed_dir: str):
    p = pathlib.Path(processed_dir)
    X_train = np.loadtxt(p/"X_train.csv", delimiter=",")
    X_val   = np.loadtxt(p/"X_val.csv",   delimiter=",")
    X_test  = np.loadtxt(p/"X_test.csv",  delimiter=",")
    y_train = _load_y(p/"y_train.csv")
    y_val   = _load_y(p/"y_val.csv")
    y_test  = _load_y(p/"y_test.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ---------- metrics ----------
def f_beta(prec: float, rec: float, beta: float) -> float:
    if prec == 0 and rec == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * prec + rec
    return (1 + b2) * prec * rec / denom if denom > 0 else 0.0

def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict:
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

def grid_thresholds(resolution: int = 1001) -> np.ndarray:
    return np.linspace(0.0, 1.0, resolution)

def select_thr_by_recall(y_true: np.ndarray, probs: np.ndarray, target_recall: float = 0.80) -> float:
    # минимальный порог, дающий требуемый Recall
    for thr in grid_thresholds():
        y_hat = (probs >= thr).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        if rec >= target_recall:
            return float(thr)
    # если недостижимо — берём порог с максимальным Recall
    best_thr, best_rec = 0.0, -1.0
    for thr in grid_thresholds():
        y_hat = (probs >= thr).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        if rec > best_rec:
            best_rec, best_thr = rec, thr
    return float(best_thr)

def select_thr_by_fbeta(y_true: np.ndarray, probs: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    best_thr, best_fb = 0.5, -1.0
    for thr in grid_thresholds():
        y_hat = (probs >= thr).astype(int)
        prec, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        fb = f_beta(prec, rec, beta)
        if fb > best_fb:
            best_fb, best_thr = fb, thr
    return float(best_thr), float(best_fb)

# ---------- models (без встроенной балансировки: SMOTE уже балансирует) ----------
def get_model(name: str, seed: int = 42):
    name = name.lower()
    if name == "lr":
        return LogisticRegression(max_iter=2000, solver="liblinear", class_weight=None, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            class_weight=None, random_state=seed
        )
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier  # ленивый импорт
        return XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.0,  # ВАЖНО: при SMOTE отключаем взвешивание
            eval_metric="logloss", random_state=seed, n_jobs=4
        )
    raise ValueError("unknown model: " + name)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lr", "rf", "xgb", "mlp"], help="модель")
    ap.add_argument("--processed_dir", default="data/processed", help="где лежат X_*/y_*")
    ap.add_argument("--artifacts_dir", default="artifacts", help="куда складывать json-метрики")
    ap.add_argument("--mode", choices=["recall", "fbeta"], default="fbeta", help="критерий подбора порога на val")
    ap.add_argument("--target_recall", type=float, default=0.80, help="целевой recall (для mode=recall)")
    ap.add_argument("--beta", type=float, default=2.0, help="β для Fβ (для mode=fbeta)")
    ap.add_argument("--seed", type=int, default=42)
    # SMOTE options
    ap.add_argument("--sampling_strategy", type=float, default=1.0,
                    help="доля миноритарного к мажоритарному после oversampling (1.0 = выровнять классы)")
    ap.add_argument("--smote_k_neighbors", type=int, default=5, help="k для SMOTE (обычно 3–10)")
    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)

    # Подготовим пайплайн: SMOTE -> model
    smote = SMOTE(
        sampling_strategy=args.sampling_strategy,
        k_neighbors=args.smote_k_neighbors,
        random_state=args.seed
    )
    model = get_model(args.model, seed=args.seed)
    pipe = ImbPipeline(steps=[("smote", smote), ("model", model)])

    # Обучение (SMOTE применяется ТОЛЬКО к train внутри пайплайна)
    pipe.fit(X_train, y_train)

    # Вероятности (валидация/тест — без SMOTE!)
    if hasattr(pipe, "predict_proba"):
        p_val  = pipe.predict_proba(X_val)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]
    else:
        # на всякий случай fallback
        scores_val  = pipe.decision_function(X_val)
        scores_test = pipe.decision_function(X_test)
        r_val  = scores_val.argsort().argsort().astype(float);  p_val  = r_val  / (len(r_val)  - 1 + 1e-9)
        r_test = scores_test.argsort().argsort().astype(float); p_test = r_test / (len(r_test) - 1 + 1e-9)

    # Выбор порога по валидации
    sel_info = {"mode": args.mode, "sampling_strategy": args.sampling_strategy, "smote_k": args.smote_k_neighbors}
    if args.mode == "recall":
        thr = select_thr_by_recall(y_val, p_val, target_recall=args.target_recall)
        sel_info["target_recall"] = args.target_recall
    else:
        thr, best_fb = select_thr_by_fbeta(y_val, p_val, beta=args.beta)
        sel_info["beta"] = args.beta
        sel_info["val_best_Fbeta"] = best_fb

    # Метрики на val/test при выбранном пороге
    m_val  = metrics_from_probs(y_val,  p_val,  thr)
    m_test = metrics_from_probs(y_test, p_test, thr)

    print(f"\n[{args.model.upper()} + SMOTE] Selected threshold on VAL: {thr:.3f}  ({sel_info})")
    print("[VAL ]", m_val)
    print("[TEST]", m_test)

    # Сохранение артефактов
    out_dir = pathlib.Path(args.artifacts_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "smote": {"sampling_strategy": args.sampling_strategy, "k_neighbors": args.smote_k_neighbors, "seed": args.seed},
        "threshold_selection": sel_info,
        "selected_threshold": float(thr),
        "val":  m_val,
        "test": m_test,
    }
    out_path = out_dir / f"{args.model}_smote_thresholded_{args.mode}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
