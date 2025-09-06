# -*- coding: utf-8 -*-
"""
04_train_model.py
Единый скрипт обучения LR / RF / XGB / MLP с честной оценкой на val/test.
Особенности:
- XGBoost: eval_metric='aucpr', попытка early_stopping по валидке; если версия xgboost не поддерживает,
  делаем фоллбэк без early stopping.
- RF: адекватные дефолты для дисбаланса.
- MLP: учитываем дисбаланс через sample_weight; если версия sklearn не поддерживает sample_weight,
  выполняем мягкий апсемплинг класса 1 до заданной целевой доли (по умолчанию ~25%).
  Есть простой перебор гиперпараметров по валидке с целевой метрикой AP (PR-AUC).
- Сохраняем метрики в JSON, по флагу сохраняем вероятности.
"""

import argparse, json, pathlib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_fscore_support, brier_score_loss
)

# xgboost (установлен?)
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------- IO ----------
def load_xy(processed_dir):
    p = pathlib.Path(processed_dir)
    X_train = np.loadtxt(p/"X_train.csv", delimiter=",")
    X_val   = np.loadtxt(p/"X_val.csv",   delimiter=",")
    X_test  = np.loadtxt(p/"X_test.csv",  delimiter=",")
    y_train = np.loadtxt(p/"y_train.csv", delimiter=",", skiprows=1).astype(int)
    y_val   = np.loadtxt(p/"y_val.csv",   delimiter=",", skiprows=1).astype(int)
    y_test  = np.loadtxt(p/"y_test.csv",  delimiter=",", skiprows=1).astype(int)

    assert X_train.shape[0] == y_train.shape[0], f"Mismatch train: {X_train.shape[0]} vs {y_train.shape[0]}"
    assert X_val.shape[0]   == y_val.shape[0],   f"Mismatch val: {X_val.shape[0]} vs {y_val.shape[0]}"
    assert X_test.shape[0]  == y_test.shape[0],  f"Mismatch test: {X_test.shape[0]} vs {y_test.shape[0]}"
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------- Utils & Metrics ----------
def prevalence(y):
    y = np.asarray(y).astype(int)
    return float(y.mean()) if y.size else float("nan")

def compute_metrics(y_true, proba, thr):
    y_hat = (proba >= thr).astype(int)
    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, proba)
        roc    = roc_auc_score(y_true, proba)
    else:
        pr_auc, roc = float("nan"), float("nan")
    f1     = f1_score(y_true, y_hat, zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0, average=None)
    recall_pos = float(rec[1]) if len(rec) > 1 else 0.0
    brier = brier_score_loss(y_true, proba)
    return {"PR_AUC": float(pr_auc), "ROC_AUC": float(roc), "F1": float(f1),
            "Recall@Thr": recall_pos, "Brier": float(brier), "Thr": float(thr)}

def majority_metrics(y_true, thr):
    proba = np.zeros_like(y_true, dtype=float)
    return compute_metrics(y_true, proba, thr)

def neg_pos_ratio(y):
    y = np.asarray(y).astype(int)
    pos = y.sum()
    neg = len(y) - pos
    return (neg / max(pos, 1)), int(neg), int(pos)

def upsample_to_target(X, y, target_frac=0.25, seed=42, k_cap=20):
    """
    Дублирует класс 1 так, чтобы доля позитивов стала ~ target_frac (<=0.5).
    Возвращает X_up, y_up (перемешанные).
    """
    y = np.asarray(y).astype(int)
    pos = int(y.sum()); neg = int(len(y) - pos)
    if pos == 0:
        return X, y
    tf = float(np.clip(target_frac, 0.05, 0.5))
    # найти k: prevalence' = pos*k / (neg + pos*k) ~= tf  =>  k = tf*neg / (pos*(1-tf))
    k = int(np.ceil((tf * neg) / (pos * (1.0 - tf))))
    k = int(np.clip(k, 1, k_cap))
    if k <= 1:
        return X, y

    X_pos = X[y == 1]; y_pos = y[y == 1]
    X_neg = X[y == 0]; y_neg = y[y == 0]

    X_pos_up = np.repeat(X_pos, repeats=k, axis=0)
    y_pos_up = np.repeat(y_pos, repeats=k, axis=0)

    Xb = np.vstack([X_neg, X_pos_up])
    yb = np.hstack([y_neg, y_pos_up])

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(yb))
    return Xb[idx], yb[idx]


# ---------- Model Factory ----------
def get_model(name, y_train=None, seed=42):
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=800,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed
        )
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost не установлен. Установи пакет xgboost.")
        spw, neg, pos = neg_pos_ratio(y_train)
        return XGBClassifier(
            n_estimators=3000,            # много деревьев + (если получится) early stopping остановит раньше
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=spw,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=30,
            random_state=seed
        )
    if name == "lr":
        return LogisticRegression(
            solver="liblinear",
            penalty="l2",
            class_weight="balanced",
            max_iter=10000,
            random_state=seed
        )
    raise ValueError("Неизвестная модель: " + name)


# ---------- Train/Eval ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lr","rf","xgb","mlp"])
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--save_probas", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Настройки (и мини-грид) для MLP
    ap.add_argument("--mlp_search", action="store_true", help="перебор небольшой сетки гиперпараметров MLP по валидке (скoring=AP)")
    ap.add_argument("--mlp_layers_grid", default="128,64;64,32;64;32",
                    help="варианты слоёв через ';' (пример: '128,64;64,32;64;32')")
    ap.add_argument("--mlp_alpha_grid", default="0.0001,0.0003,0.001,0.003",
                    help="варианты alpha через запятую")
    ap.add_argument("--mlp_lr_grid", default="0.001,0.003",
                    help="варианты learning_rate_init через запятую")
    ap.add_argument("--mlp_target_pos_frac", type=float, default=0.25,
                    help="целевая доля позитивов после апсемплинга (при отсутствии sample_weight)")

    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)

    prev_tr, prev_va, prev_te = prevalence(y_train), prevalence(y_val), prevalence(y_test)
    print(f"[INFO] Prevalence: train={prev_tr:.4f} val={prev_va:.4f} test={prev_te:.4f} (AP бейзлайн ~= prevalence)")

    model = get_model(args.model, y_train=y_train, seed=args.seed)

    # --- fit (с особенностями по моделям) ---
    if args.model == "xgb":
        eval_set = [(X_train, y_train), (X_val, y_val)]
        # Пытаемся использовать early stopping (если версия xgboost поддерживает этот аргумент в fit)
        try:
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=200
            )
        except TypeError:
            print("[WARN] Эта версия xgboost не поддерживает early_stopping_rounds. Обучаю без early stopping.")
            model.set_params(n_estimators=800)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )

    elif args.model == "mlp":
        # Если просили мини-грид — перебираем гиперпараметры по валидке с метрикой AP.
        def parse_layers_grid(s):
            outs = []
            for part in s.split(";"):
                part = part.strip()
                if not part:
                    continue
                outs.append(tuple(int(x) for x in part.split(",") if x.strip()))
            return outs

        layers_grid = parse_layers_grid(args.mlp_layers_grid)
        alpha_grid  = [float(x) for x in args.mlp_alpha_grid.split(",") if x.strip()]
        lr_grid     = [float(x) for x in args.mlp_lr_grid.split(",") if x.strip()]

        def fit_one_mlp(hls, alpha, lr):
            m = MLPClassifier(
                hidden_layer_sizes=hls,
                activation="relu",
                alpha=alpha,
                learning_rate_init=lr,
                batch_size=64,
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=30,
                random_state=args.seed
            )
            spw, neg, pos = neg_pos_ratio(y_train)
            try:
                w = np.ones_like(y_train, dtype=float)
                w[y_train == 1] = spw  # веса ~ neg/pos
                m.fit(X_train, y_train, sample_weight=w)
            except TypeError:
                # мягкий апсемплинг до целевой доли
                Xb, yb = upsample_to_target(X_train, y_train, target_frac=args.mlp_target_pos_frac, seed=args.seed)
                m.fit(Xb, yb)
            return m

        if args.mlp_search:
            best = None
            best_ap = -1.0
            tried = 0
            for hls in layers_grid:
                for a in alpha_grid:
                    for lr in lr_grid:
                        tried += 1
                        m = fit_one_mlp(hls=hls, alpha=a, lr=lr)
                        proba_val = m.predict_proba(X_val)[:, 1]
                        ap = average_precision_score(y_val, proba_val) if len(np.unique(y_val)) > 1 else float("nan")
                        print(f"[MLP-SEARCH] hls={hls} alpha={a} lr={lr} | val AP={ap:.4f}")
                        if np.isfinite(ap) and ap > best_ap:
                            best_ap = ap
                            best = m
            print(f"[MLP-SEARCH] tried={tried} | best val AP={best_ap:.4f}")
            model = best
        else:
            # одна базовая конфигурация (как раньше)
            spw, neg, pos = neg_pos_ratio(y_train)
            try:
                w = np.ones_like(y_train, dtype=float)
                w[y_train == 1] = spw
                model.fit(X_train, y_train, sample_weight=w)
            except TypeError:
                Xb, yb = upsample_to_target(X_train, y_train, target_frac=args.mlp_target_pos_frac, seed=args.seed)
                print(f"[WARN] MLPClassifier.fit(sample_weight=...) не поддерживается. "
                      f"Апсемплю класс 1 до доли ~{args.mlp_target_pos_frac:.2f} (после апсемплинга N={len(yb)}).")
                model.fit(Xb, yb)

    else:
        # RF/LR — обычный fit (class_weight уже задан где нужно)
        model.fit(X_train, y_train)

    # --- predict proba ---
    if hasattr(model, "predict_proba"):
        proba_val  = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
    else:
        # fallback для моделей без predict_proba
        scores_val  = model.decision_function(X_val)
        scores_test = model.decision_function(X_test)
        r_val  = scores_val.argsort().argsort().astype(float);  proba_val  = r_val  / (len(r_val)  - 1 + 1e-9)
        r_test = scores_test.argsort().argsort().astype(float); proba_test = r_test / (len(r_test) - 1 + 1e-9)

    # --- metrics ---
    m_val  = compute_metrics(y_val,  proba_val,  args.thr)
    m_test = compute_metrics(y_test, proba_test, args.thr)
    base_val  = majority_metrics(y_val,  args.thr)
    base_test = majority_metrics(y_test, args.thr)

    print(f"\n=== {args.model.upper()} ===")
    print(f"[VAL ] PR-AUC={m_val['PR_AUC']:.4f} | ROC-AUC={m_val['ROC_AUC']:.4f} | F1={m_val['F1']:.4f} | Recall@{args.thr:.2f}={m_val['Recall@Thr']:.4f} | Brier={m_val['Brier']:.4f}")
    print(f"[TEST] PR-AUC={m_test['PR_AUC']:.4f} | ROC-AUC={m_test['ROC_AUC']:.4f} | F1={m_test['F1']:.4f} | Recall@{args.thr:.2f}={m_test['Recall@Thr']:.4f} | Brier={m_test['Brier']:.4f}")

    print("\n--- Majority baseline (always 0) ---")
    print(f"[VAL ] PR-AUC={base_val['PR_AUC']:.4f} | ROC-AUC={base_val['ROC_AUC']:.4f} | F1={base_val['F1']:.4f} | Recall@{args.thr:.2f}={base_val['Recall@Thr']:.4f} | Brier={base_val['Brier']:.4f}")
    print(f"[TEST] PR-AUC={base_test['PR_AUC']:.4f} | ROC-AUC={base_test['ROC_AUC']:.4f} | F1={base_test['F1']:.4f} | Recall@{args.thr:.2f}={base_test['Recall@Thr']:.4f} | Brier={base_test['Brier']:.4f}")

    # --- save artifacts ---
    out_dir = pathlib.Path(args.artifacts_dir); (out_dir/"metrics").mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir/"metrics"/f"{args.model}_eval.json"
    metrics_json = {
        "model": args.model,
        "threshold": args.thr,
        "prevalence": {"train": prev_tr, "val": prev_va, "test": prev_te},
        "val": m_val,
        "test": m_test,
        "baseline_val": base_val,
        "baseline_test": base_test
    }
    metrics_path.write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False))

    if args.save_probas:
        prob_dir = out_dir/"probas"; prob_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y": y_val, "proba": proba_val}).to_csv(prob_dir/f"{args.model}_val.csv", index=False)
        pd.DataFrame({"y": y_test, "proba": proba_test}).to_csv(prob_dir/f"{args.model}_test.csv", index=False)

    print(f"\n[OK] Метрики сохранены: {metrics_path}")
    if args.save_probas:
        print(f"[OK] Вероятности сохранены в: {out_dir/'probas'}")


if __name__ == "__main__":
    main()
