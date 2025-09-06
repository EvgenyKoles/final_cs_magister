# -*- coding: utf-8 -*-
"""
05_tune_xgb.py
Рандомный поиск XGBoost по CV с целевой метрикой PR-AUC (average precision).

Фишки:
- RepeatedStratifiedKFold: больше независимых сплитов => стабильнее оценка.
- Агрегатор метрики: mean / median / mean_minus_std (штраф за разброс).
- Гибкие флаги: booster, scale_pos_weight, ранняя остановка в поиске/финале.
- Артефакты: лидерборд CSV, сводка JSON, лучшая модель, вероятности, важность фич.

Примеры:
  # Стабильно и без ES, ищем только gbtree, spw среди {auto,sqrt,one}
  python scripts/05_tune_xgb.py --trials 80 --cv_folds 3 --cv_repeats 5 \
         --agg mean_minus_std --agg_alpha 0.5 \
         --booster_mode gbtree --spw_mode search --es_rounds 0 \
         --train_on trainval --save_model --save_probas
"""

import argparse, json, pathlib, math
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, \
                            precision_recall_fscore_support, brier_score_loss

# XGBoost
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

def load_feature_names(processed_dir: str):
    fn = pathlib.Path(processed_dir)/"feature_names.csv"
    if fn.exists():
        try:
            return pd.read_csv(fn, header=None).iloc[:,0].tolist()
        except Exception:
            return None
    return None


# ---------- Metrics ----------
def prevalence(y): return float(np.asarray(y).mean()) if len(y) else float("nan")

def metrics_from_proba(y_true, proba, thr=0.35) -> Dict[str, float]:
    y_hat = (proba >= thr).astype(int)
    if len(np.unique(y_true)) > 1:
        ap  = average_precision_score(y_true, proba)
        roc = roc_auc_score(y_true, proba)
    else:
        ap, roc = float("nan"), float("nan")
    f1  = f1_score(y_true, y_hat, zero_division=0)
    pr, rc, _, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0, average=None)
    rec_pos = float(rc[1]) if len(rc) > 1 else 0.0
    br = brier_score_loss(y_true, proba)
    return {"PR_AUC": float(ap), "ROC_AUC": float(roc), "F1": float(f1),
            "Recall@Thr": float(rec_pos), "Brier": float(br), "Thr": float(thr)}

def majority_metrics(y_true, thr=0.35) -> Dict[str, float]:
    return metrics_from_proba(y_true, np.zeros_like(y_true, dtype=float), thr)


# ---------- Helpers ----------
def neg_pos_stats(y: np.ndarray) -> Tuple[int,int,float]:
    pos = int(y.sum()); neg = int(len(y) - pos)
    spw = (neg / max(pos,1)) if pos>0 else 1.0
    return neg, pos, spw

def build_model(config: Dict[str, Any], y_tr: np.ndarray, seed: int) -> XGBClassifier:
    # scale_pos_weight modes
    neg, pos, spw_auto = neg_pos_stats(y_tr)
    m = config["spw_mode"]
    if m == "auto":
        spw = spw_auto
    elif m == "sqrt":
        spw = math.sqrt(spw_auto)
    else:
        spw = 1.0  # one

    kwargs = dict(
        n_estimators=config["n_estimators"],
        learning_rate=config["lr"],
        max_depth=config["depth"],
        min_child_weight=config["mcw"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample"],
        reg_alpha=config["alpha"],
        reg_lambda=config["lambda"],
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        tree_method="hist",
        booster=config["booster"],
        random_state=seed,
        n_jobs=-1
    )
    if config["booster"] == "dart":
        kwargs.update(rate_drop=config["rate_drop"], skip_drop=config["skip_drop"])
    # ES в конструкторе (для XGBoost >=2.1/3.x)
    if config["es_rounds"] > 0:
        kwargs["early_stopping_rounds"] = int(config["es_rounds"])
    return XGBClassifier(**kwargs)


def sample_config(rng: np.random.RandomState,
                  booster_mode: str,
                  spw_mode_flag: str,
                  es_rounds: int,
                  space: str) -> Dict[str, Any]:
    """Сэмплируем конфиг из разумных диапазонов; часть вещей фиксируем флагами."""
    if space == "light":
        depth_choices = [3,4]
        lr_choices    = [0.01, 0.02, 0.03]
        n_estim_choices = [800, 1200, 1500, 2000]
        mcw_choices   = [1,2,3]
        lam_choices   = [1.0, 2.0, 3.0, 5.0, 10.0]
        alp_choices   = [0.0, 0.05, 0.1, 0.3]
        subs_lo, subs_hi = 0.8, 0.95
        cols_lo, cols_hi = 0.8, 0.95
    else:  # wide
        depth_choices = [3,4,5]
        lr_choices    = [0.01, 0.02, 0.03, 0.05]
        n_estim_choices = [800, 1200, 1500, 2000, 2500]
        mcw_choices   = [1,2,3,5]
        lam_choices   = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        alp_choices   = [0.0, 0.05, 0.1, 0.3, 0.5]
        subs_lo, subs_hi = 0.75, 0.95
        cols_lo, cols_hi = 0.75, 0.95

    depth     = rng.choice(depth_choices)
    mcw       = rng.choice(mcw_choices)
    lr        = rng.choice(lr_choices)
    n_estim   = rng.choice(n_estim_choices)
    subsample = rng.uniform(subs_lo, subs_hi)
    colsample = rng.uniform(cols_lo, cols_hi)
    reg_lam   = rng.choice(lam_choices)
    reg_alp   = rng.choice(alp_choices)

    if booster_mode == "gbtree":
        booster = "gbtree"; rate_drop = 0.0; skip_drop = 0.0
    elif booster_mode == "dart":
        booster = "dart"
        rate_drop = rng.uniform(0.05, 0.3)
        skip_drop = rng.uniform(0.0, 0.6)
    else:
        booster = rng.choice(["gbtree","dart"])
        if booster == "dart":
            rate_drop = rng.uniform(0.05, 0.3)
            skip_drop = rng.uniform(0.0, 0.6)
        else:
            rate_drop = 0.0; skip_drop = 0.0

    if spw_mode_flag == "search":
        spw_mode = rng.choice(["auto","sqrt","one"])
    else:
        spw_mode = spw_mode_flag

    return {
        "depth": int(depth), "mcw": float(mcw), "lr": float(lr), "n_estimators": int(n_estim),
        "subsample": float(subsample), "colsample": float(colsample),
        "lambda": float(reg_lam), "alpha": float(reg_alp),
        "booster": booster, "rate_drop": float(rate_drop), "skip_drop": float(skip_drop),
        "spw_mode": spw_mode,
        "es_rounds": int(es_rounds)  # 0 = выключен
    }


def aggregate(scores: List[float], mode: str, alpha: float) -> float:
    arr = np.array(scores, dtype=float)
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    # penalized
    return float(np.mean(arr) - alpha * np.std(arr, ddof=0))


def cv_ap_for_config(X: np.ndarray, y: np.ndarray, config: Dict[str,Any],
                     cv_folds: int, cv_repeats: int, seed: int,
                     agg_mode: str, agg_alpha: float) -> Tuple[float,float,float,int]:
    scores, rounds = [], []
    # Повторяем k-fold с разными сидами
    for rep in range(cv_repeats):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 1000*rep)
        for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X[idx_tr], X[idx_va]
            y_tr, y_va = y[idx_tr], y[idx_va]

            model = build_model(config, y_tr, seed + 17*(rep+1) + fold)
            if config["es_rounds"] > 0:
                model.fit(X_tr, y_tr, eval_set=[(X_tr,y_tr),(X_va,y_va)], verbose=False)
                best = getattr(model, "best_score", None)
                ap = float(best) if best is not None else average_precision_score(y_va, model.predict_proba(X_va)[:,1])
                rounds.append(getattr(model, "best_iteration", model.get_params().get("n_estimators", 0)))
            else:
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_va)[:,1]
                ap = average_precision_score(y_va, proba)
                rounds.append(model.get_params().get("n_estimators", 0))
            scores.append(ap)

    ap_mean = float(np.mean(scores))
    ap_std  = float(np.std(scores, ddof=0))
    ap_med  = float(np.median(scores))
    ap_agg  = aggregate(scores, mode=agg_mode, alpha=agg_alpha)
    return ap_agg, ap_mean, ap_std, int(np.mean(rounds))


def train_final_and_eval(X_tr: np.ndarray, y_tr: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         X_te: np.ndarray,  y_te: np.ndarray,
                         config: Dict[str,Any], seed: int,
                         train_on: str = "train",
                         es_rounds_final: int = 0,
                         es_split: float = 0.1) -> Dict[str,Any]:
    # финальное обучение: train или train+val
    if train_on == "trainval":
        X_final = np.vstack([X_tr, X_val]); y_final = np.hstack([y_tr, y_val])
    else:
        X_final, y_final = X_tr, y_tr

    cfg = dict(config)
    cfg["es_rounds"] = int(es_rounds_final)

    model = build_model(cfg, y_final, seed)

    if es_rounds_final > 0:
        # внутренняя маленькая валидка из final набора
        rs = np.random.RandomState(seed+777)
        idx = rs.permutation(len(y_final))
        cut = max(1, int(len(y_final) * es_split))
        idx_tr, idx_va = idx[cut:], idx[:cut]
        model.fit(X_final[idx_tr], y_final[idx_tr],
                  eval_set=[(X_final[idx_tr], y_final[idx_tr]), (X_final[idx_va], y_final[idx_va])],
                  verbose=False)
    else:
        model.fit(X_final, y_final)

    proba_val = model.predict_proba(X_val)[:,1]
    proba_te  = model.predict_proba(X_te)[:,1]
    return {
        "model": model,
        "val": metrics_from_proba(y_val, proba_val),
        "test": metrics_from_proba(y_te,  proba_te),
        "proba_val": proba_val,
        "proba_test": proba_te
    }


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--cv_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_on", choices=["train","trainval"], default="trainval",
                    help="на чем обучать финальную модель")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--save_probas", action="store_true")

    # поиск: контроль пространства
    ap.add_argument("--search_space", choices=["light","wide"], default="light")
    ap.add_argument("--spw_mode", choices=["search","auto","sqrt","one"], default="search",
                    help="scale_pos_weight режим: искать среди auto/sqrt/one или зафиксировать")
    ap.add_argument("--booster_mode", choices=["both","gbtree","dart"], default="gbtree",
                    help="искать среди обоих бустеров или фиксировать один")
    ap.add_argument("--es_rounds", type=int, default=0,
                    help="early_stopping_rounds в поиске (0 = выключить)")

    # агрегатор
    ap.add_argument("--agg", choices=["mean","median","mean_minus_std"], default="mean_minus_std",
                    help="как агрегировать AP по всем фолдам/повторам")
    ap.add_argument("--agg_alpha", type=float, default=0.5,
                    help="штрафной коэффициент для mean_minus_std")

    # финальное обучение
    ap.add_argument("--es_rounds_final", type=int, default=0,
                    help="ES на финальном обучении (0 = выключить)")
    ap.add_argument("--es_split", type=float, default=0.1,
                    help="доля внутренней валидации при ES финала")

    args = ap.parse_args()

    # load
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)
    feat_names = load_feature_names(args.processed_dir)

    print(f"[INFO] Prevalence: train={prevalence(y_train):.4f} | val={prevalence(y_val):.4f} | test={prevalence(y_test):.4f}")

    rng = np.random.RandomState(args.seed)
    trials: List[Dict[str,Any]] = []
    best = {"ap_agg": -1.0}

    # random search
    for t in range(1, args.trials+1):
        cfg = sample_config(rng,
                            booster_mode=args.booster_mode,
                            spw_mode_flag=args.spw_mode,
                            es_rounds=args.es_rounds,
                            space=args.search_space)
        ap_agg, ap_mean, ap_std, mean_rounds = cv_ap_for_config(
            X_train, y_train, cfg,
            cv_folds=args.cv_folds, cv_repeats=args.cv_repeats, seed=args.seed+123,
            agg_mode=args.agg, agg_alpha=args.agg_alpha
        )
        row = dict(cfg); row.update({"cv_ap_agg": ap_agg, "cv_ap_mean": ap_mean, "cv_ap_std": ap_std, "mean_rounds": mean_rounds})
        trials.append(row)
        print(f"[{t:03d}/{args.trials}] CV(AP) agg={ap_agg:.4f} | mean={ap_mean:.4f} ± {ap_std:.4f} | cfg={cfg}")
        if ap_agg > best.get("ap_agg", -1.0):
            best = {"ap_agg": ap_agg, "ap_mean": ap_mean, "ap_std": ap_std, "cfg": cfg, "mean_rounds": mean_rounds}

    print(f"\n[INFO] BEST CV AP agg={best['ap_agg']:.4f} | mean={best['ap_mean']:.4f} ± {best['ap_std']:.4f}")
    print(f"CFG={best['cfg']}\n")

    # final train & eval
    result = train_final_and_eval(
        X_tr=X_train, y_tr=y_train, X_val=X_val, y_val=y_val, X_te=X_test, y_te=y_test,
        config=best["cfg"], seed=args.seed+999,
        train_on=args.train_on, es_rounds_final=args.es_rounds_final, es_split=args.es_split
    )

    # baselines
    base_val  = majority_metrics(y_val,  args.thr)
    base_test = majority_metrics(y_test, args.thr)

    print("=== Final XGB (best CV config) ===")
    print(f"[VAL ] PR-AUC={result['val']['PR_AUC']:.4f} | ROC-AUC={result['val']['ROC_AUC']:.4f} | F1={result['val']['F1']:.4f} | Recall@{args.thr:.2f}={result['val']['Recall@Thr']:.4f} | Brier={result['val']['Brier']:.4f}")
    print(f"[TEST] PR-AUC={result['test']['PR_AUC']:.4f} | ROC-AUC={result['test']['ROC_AUC']:.4f} | F1={result['test']['F1']:.4f} | Recall@{args.thr:.2f}={result['test']['Recall@Thr']:.4f} | Brier={result['test']['Brier']:.4f}")
    print("\n--- Majority baseline (always 0) ---")
    print(f"[VAL ] PR-AUC={base_val['PR_AUC']:.4f} | ROC-AUC={base_val['ROC_AUC']:.4f} | F1={base_val['F1']:.4f} | Recall@{args.thr:.2f}={base_val['Recall@Thr']:.4f} | Brier={base_val['Brier']:.4f}")
    print(f"[TEST] PR-AUC={base_test['PR_AUC']:.4f} | ROC-AUC={base_test['ROC_AUC']:.4f} | F1={base_test['F1']:.4f} | Recall@{args.thr:.2f}={base_test['Recall@Thr']:.4f} | Brier={base_test['Brier']:.4f}")

    # save artifacts
    art = pathlib.Path(args.artifacts_dir); (art/"metrics").mkdir(parents=True, exist_ok=True)

    # leaderboard
    lb = pd.DataFrame(trials)
    lb_path = art/"metrics"/"xgb_tuning_leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    # best config + metrics
    out = {
        "cv_folds": args.cv_folds, "cv_repeats": args.cv_repeats,
        "trials": args.trials,
        "best_cv_ap_agg": best["ap_agg"],
        "best_cv_ap_mean": best["ap_mean"],
        "best_cv_ap_std": best["ap_std"],
        "best_config": best["cfg"],
        "final_train_on": args.train_on,
        "val_metrics": result["val"],
        "test_metrics": result["test"],
        "baseline_val": base_val,
        "baseline_test": base_test,
        "xgboost_version": xgb.__version__
    }
    (art/"metrics"/"xgb_tuning_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # save probas
    if args.save_probas:
        (art/"probas").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y": y_val,  "proba": result["proba_val"]}).to_csv(art/"probas"/"xgb_best_val.csv", index=False)
        pd.DataFrame({"y": y_test, "proba": result["proba_test"]}).to_csv(art/"probas"/"xgb_best_test.csv", index=False)

    # save model
    if args.save_model:
        (art/"models").mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(result["model"], art/"models"/"xgb_best.joblib")

    # feature importances (gain)
    try:
        booster = result["model"].get_booster()
        score = booster.get_score(importance_type="gain")
        imp = pd.DataFrame(sorted(score.items(), key=lambda kv: kv[1], reverse=True), columns=["feature","gain"])
        feat_names = load_feature_names(args.processed_dir)
        if feat_names:
            def map_name(s):
                if s.startswith("f"):
                    idx = int(s[1:])
                    return feat_names[idx] if 0 <= idx < len(feat_names) else s
                return s
            imp["feature_name"] = imp["feature"].map(map_name)
            imp = imp[["feature_name","feature","gain"]]
        imp.to_csv(art/"metrics"/"xgb_feature_importance_gain.csv", index=False)
    except Exception:
        pass

    print(f"\n[OK] Лидерборд: {lb_path}")
    print(f"[OK] Сводка:    {art/'metrics'/'xgb_tuning_summary.json'}")
    if args.save_model:
        print(f"[OK] Модель:    {art/'models'/'xgb_best.joblib'}")
    if args.save_probas:
        print(f"[OK] Пробасы:   {art/'probas'}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
05_tune_xgb.py
Рандомный поиск XGBoost по CV с целевой метрикой PR-AUC (average precision).

Фишки:
- RepeatedStratifiedKFold: больше независимых сплитов => стабильнее оценка.
- Агрегатор метрики: mean / median / mean_minus_std (штраф за разброс).
- Гибкие флаги: booster, scale_pos_weight, ранняя остановка в поиске/финале.
- Артефакты: лидерборд CSV, сводка JSON, лучшая модель, вероятности, важность фич.

Примеры:
  # Стабильно и без ES, ищем только gbtree, spw среди {auto,sqrt,one}
  python scripts/05_tune_xgb.py --trials 80 --cv_folds 3 --cv_repeats 5 \
         --agg mean_minus_std --agg_alpha 0.5 \
         --booster_mode gbtree --spw_mode search --es_rounds 0 \
         --train_on trainval --save_model --save_probas
"""

import argparse, json, pathlib, math
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, \
                            precision_recall_fscore_support, brier_score_loss

# XGBoost
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

def load_feature_names(processed_dir: str):
    fn = pathlib.Path(processed_dir)/"feature_names.csv"
    if fn.exists():
        try:
            return pd.read_csv(fn, header=None).iloc[:,0].tolist()
        except Exception:
            return None
    return None


# ---------- Metrics ----------
def prevalence(y): return float(np.asarray(y).mean()) if len(y) else float("nan")

def metrics_from_proba(y_true, proba, thr=0.35) -> Dict[str, float]:
    y_hat = (proba >= thr).astype(int)
    if len(np.unique(y_true)) > 1:
        ap  = average_precision_score(y_true, proba)
        roc = roc_auc_score(y_true, proba)
    else:
        ap, roc = float("nan"), float("nan")
    f1  = f1_score(y_true, y_hat, zero_division=0)
    pr, rc, _, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0, average=None)
    rec_pos = float(rc[1]) if len(rc) > 1 else 0.0
    br = brier_score_loss(y_true, proba)
    return {"PR_AUC": float(ap), "ROC_AUC": float(roc), "F1": float(f1),
            "Recall@Thr": float(rec_pos), "Brier": float(br), "Thr": float(thr)}

def majority_metrics(y_true, thr=0.35) -> Dict[str, float]:
    return metrics_from_proba(y_true, np.zeros_like(y_true, dtype=float), thr)


# ---------- Helpers ----------
def neg_pos_stats(y: np.ndarray) -> Tuple[int,int,float]:
    pos = int(y.sum()); neg = int(len(y) - pos)
    spw = (neg / max(pos,1)) if pos>0 else 1.0
    return neg, pos, spw

def build_model(config: Dict[str, Any], y_tr: np.ndarray, seed: int) -> XGBClassifier:
    # scale_pos_weight modes
    neg, pos, spw_auto = neg_pos_stats(y_tr)
    m = config["spw_mode"]
    if m == "auto":
        spw = spw_auto
    elif m == "sqrt":
        spw = math.sqrt(spw_auto)
    else:
        spw = 1.0  # one

    kwargs = dict(
        n_estimators=config["n_estimators"],
        learning_rate=config["lr"],
        max_depth=config["depth"],
        min_child_weight=config["mcw"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample"],
        reg_alpha=config["alpha"],
        reg_lambda=config["lambda"],
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        tree_method="hist",
        booster=config["booster"],
        random_state=seed,
        n_jobs=-1
    )
    if config["booster"] == "dart":
        kwargs.update(rate_drop=config["rate_drop"], skip_drop=config["skip_drop"])
    # ES в конструкторе (для XGBoost >=2.1/3.x)
    if config["es_rounds"] > 0:
        kwargs["early_stopping_rounds"] = int(config["es_rounds"])
    return XGBClassifier(**kwargs)


def sample_config(rng: np.random.RandomState,
                  booster_mode: str,
                  spw_mode_flag: str,
                  es_rounds: int,
                  space: str) -> Dict[str, Any]:
    """Сэмплируем конфиг из разумных диапазонов; часть вещей фиксируем флагами."""
    if space == "light":
        depth_choices = [3,4]
        lr_choices    = [0.01, 0.02, 0.03]
        n_estim_choices = [800, 1200, 1500, 2000]
        mcw_choices   = [1,2,3]
        lam_choices   = [1.0, 2.0, 3.0, 5.0, 10.0]
        alp_choices   = [0.0, 0.05, 0.1, 0.3]
        subs_lo, subs_hi = 0.8, 0.95
        cols_lo, cols_hi = 0.8, 0.95
    else:  # wide
        depth_choices = [3,4,5]
        lr_choices    = [0.01, 0.02, 0.03, 0.05]
        n_estim_choices = [800, 1200, 1500, 2000, 2500]
        mcw_choices   = [1,2,3,5]
        lam_choices   = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        alp_choices   = [0.0, 0.05, 0.1, 0.3, 0.5]
        subs_lo, subs_hi = 0.75, 0.95
        cols_lo, cols_hi = 0.75, 0.95

    depth     = rng.choice(depth_choices)
    mcw       = rng.choice(mcw_choices)
    lr        = rng.choice(lr_choices)
    n_estim   = rng.choice(n_estim_choices)
    subsample = rng.uniform(subs_lo, subs_hi)
    colsample = rng.uniform(cols_lo, cols_hi)
    reg_lam   = rng.choice(lam_choices)
    reg_alp   = rng.choice(alp_choices)

    if booster_mode == "gbtree":
        booster = "gbtree"; rate_drop = 0.0; skip_drop = 0.0
    elif booster_mode == "dart":
        booster = "dart"
        rate_drop = rng.uniform(0.05, 0.3)
        skip_drop = rng.uniform(0.0, 0.6)
    else:
        booster = rng.choice(["gbtree","dart"])
        if booster == "dart":
            rate_drop = rng.uniform(0.05, 0.3)
            skip_drop = rng.uniform(0.0, 0.6)
        else:
            rate_drop = 0.0; skip_drop = 0.0

    if spw_mode_flag == "search":
        spw_mode = rng.choice(["auto","sqrt","one"])
    else:
        spw_mode = spw_mode_flag

    return {
        "depth": int(depth), "mcw": float(mcw), "lr": float(lr), "n_estimators": int(n_estim),
        "subsample": float(subsample), "colsample": float(colsample),
        "lambda": float(reg_lam), "alpha": float(reg_alp),
        "booster": booster, "rate_drop": float(rate_drop), "skip_drop": float(skip_drop),
        "spw_mode": spw_mode,
        "es_rounds": int(es_rounds)  # 0 = выключен
    }


def aggregate(scores: List[float], mode: str, alpha: float) -> float:
    arr = np.array(scores, dtype=float)
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    # penalized
    return float(np.mean(arr) - alpha * np.std(arr, ddof=0))


def cv_ap_for_config(X: np.ndarray, y: np.ndarray, config: Dict[str,Any],
                     cv_folds: int, cv_repeats: int, seed: int,
                     agg_mode: str, agg_alpha: float) -> Tuple[float,float,float,int]:
    scores, rounds = [], []
    # Повторяем k-fold с разными сидами
    for rep in range(cv_repeats):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 1000*rep)
        for fold, (idx_tr, idx_va) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X[idx_tr], X[idx_va]
            y_tr, y_va = y[idx_tr], y[idx_va]

            model = build_model(config, y_tr, seed + 17*(rep+1) + fold)
            if config["es_rounds"] > 0:
                model.fit(X_tr, y_tr, eval_set=[(X_tr,y_tr),(X_va,y_va)], verbose=False)
                best = getattr(model, "best_score", None)
                ap = float(best) if best is not None else average_precision_score(y_va, model.predict_proba(X_va)[:,1])
                rounds.append(getattr(model, "best_iteration", model.get_params().get("n_estimators", 0)))
            else:
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_va)[:,1]
                ap = average_precision_score(y_va, proba)
                rounds.append(model.get_params().get("n_estimators", 0))
            scores.append(ap)

    ap_mean = float(np.mean(scores))
    ap_std  = float(np.std(scores, ddof=0))
    ap_med  = float(np.median(scores))
    ap_agg  = aggregate(scores, mode=agg_mode, alpha=agg_alpha)
    return ap_agg, ap_mean, ap_std, int(np.mean(rounds))


def train_final_and_eval(X_tr: np.ndarray, y_tr: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         X_te: np.ndarray,  y_te: np.ndarray,
                         config: Dict[str,Any], seed: int,
                         train_on: str = "train",
                         es_rounds_final: int = 0,
                         es_split: float = 0.1) -> Dict[str,Any]:
    # финальное обучение: train или train+val
    if train_on == "trainval":
        X_final = np.vstack([X_tr, X_val]); y_final = np.hstack([y_tr, y_val])
    else:
        X_final, y_final = X_tr, y_tr

    cfg = dict(config)
    cfg["es_rounds"] = int(es_rounds_final)

    model = build_model(cfg, y_final, seed)

    if es_rounds_final > 0:
        # внутренняя маленькая валидка из final набора
        rs = np.random.RandomState(seed+777)
        idx = rs.permutation(len(y_final))
        cut = max(1, int(len(y_final) * es_split))
        idx_tr, idx_va = idx[cut:], idx[:cut]
        model.fit(X_final[idx_tr], y_final[idx_tr],
                  eval_set=[(X_final[idx_tr], y_final[idx_tr]), (X_final[idx_va], y_final[idx_va])],
                  verbose=False)
    else:
        model.fit(X_final, y_final)

    proba_val = model.predict_proba(X_val)[:,1]
    proba_te  = model.predict_proba(X_te)[:,1]
    return {
        "model": model,
        "val": metrics_from_proba(y_val, proba_val),
        "test": metrics_from_proba(y_te,  proba_te),
        "proba_val": proba_val,
        "proba_test": proba_te
    }


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--cv_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_on", choices=["train","trainval"], default="trainval",
                    help="на чем обучать финальную модель")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--save_probas", action="store_true")

    # поиск: контроль пространства
    ap.add_argument("--search_space", choices=["light","wide"], default="light")
    ap.add_argument("--spw_mode", choices=["search","auto","sqrt","one"], default="search",
                    help="scale_pos_weight режим: искать среди auto/sqrt/one или зафиксировать")
    ap.add_argument("--booster_mode", choices=["both","gbtree","dart"], default="gbtree",
                    help="искать среди обоих бустеров или фиксировать один")
    ap.add_argument("--es_rounds", type=int, default=0,
                    help="early_stopping_rounds в поиске (0 = выключить)")

    # агрегатор
    ap.add_argument("--agg", choices=["mean","median","mean_minus_std"], default="mean_minus_std",
                    help="как агрегировать AP по всем фолдам/повторам")
    ap.add_argument("--agg_alpha", type=float, default=0.5,
                    help="штрафной коэффициент для mean_minus_std")

    # финальное обучение
    ap.add_argument("--es_rounds_final", type=int, default=0,
                    help="ES на финальном обучении (0 = выключить)")
    ap.add_argument("--es_split", type=float, default=0.1,
                    help="доля внутренней валидации при ES финала")

    args = ap.parse_args()

    # load
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)
    feat_names = load_feature_names(args.processed_dir)

    print(f"[INFO] Prevalence: train={prevalence(y_train):.4f} | val={prevalence(y_val):.4f} | test={prevalence(y_test):.4f}")

    rng = np.random.RandomState(args.seed)
    trials: List[Dict[str,Any]] = []
    best = {"ap_agg": -1.0}

    # random search
    for t in range(1, args.trials+1):
        cfg = sample_config(rng,
                            booster_mode=args.booster_mode,
                            spw_mode_flag=args.spw_mode,
                            es_rounds=args.es_rounds,
                            space=args.search_space)
        ap_agg, ap_mean, ap_std, mean_rounds = cv_ap_for_config(
            X_train, y_train, cfg,
            cv_folds=args.cv_folds, cv_repeats=args.cv_repeats, seed=args.seed+123,
            agg_mode=args.agg, agg_alpha=args.agg_alpha
        )
        row = dict(cfg); row.update({"cv_ap_agg": ap_agg, "cv_ap_mean": ap_mean, "cv_ap_std": ap_std, "mean_rounds": mean_rounds})
        trials.append(row)
        print(f"[{t:03d}/{args.trials}] CV(AP) agg={ap_agg:.4f} | mean={ap_mean:.4f} ± {ap_std:.4f} | cfg={cfg}")
        if ap_agg > best.get("ap_agg", -1.0):
            best = {"ap_agg": ap_agg, "ap_mean": ap_mean, "ap_std": ap_std, "cfg": cfg, "mean_rounds": mean_rounds}

    print(f"\n[INFO] BEST CV AP agg={best['ap_agg']:.4f} | mean={best['ap_mean']:.4f} ± {best['ap_std']:.4f}")
    print(f"CFG={best['cfg']}\n")

    # final train & eval
    result = train_final_and_eval(
        X_tr=X_train, y_tr=y_train, X_val=X_val, y_val=y_val, X_te=X_test, y_te=y_test,
        config=best["cfg"], seed=args.seed+999,
        train_on=args.train_on, es_rounds_final=args.es_rounds_final, es_split=args.es_split
    )

    # baselines
    base_val  = majority_metrics(y_val,  args.thr)
    base_test = majority_metrics(y_test, args.thr)

    print("=== Final XGB (best CV config) ===")
    print(f"[VAL ] PR-AUC={result['val']['PR_AUC']:.4f} | ROC-AUC={result['val']['ROC_AUC']:.4f} | F1={result['val']['F1']:.4f} | Recall@{args.thr:.2f}={result['val']['Recall@Thr']:.4f} | Brier={result['val']['Brier']:.4f}")
    print(f"[TEST] PR-AUC={result['test']['PR_AUC']:.4f} | ROC-AUC={result['test']['ROC_AUC']:.4f} | F1={result['test']['F1']:.4f} | Recall@{args.thr:.2f}={result['test']['Recall@Thr']:.4f} | Brier={result['test']['Brier']:.4f}")
    print("\n--- Majority baseline (always 0) ---")
    print(f"[VAL ] PR-AUC={base_val['PR_AUC']:.4f} | ROC-AUC={base_val['ROC_AUC']:.4f} | F1={base_val['F1']:.4f} | Recall@{args.thr:.2f}={base_val['Recall@Thr']:.4f} | Brier={base_val['Brier']:.4f}")
    print(f"[TEST] PR-AUC={base_test['PR_AUC']:.4f} | ROC-AUC={base_test['ROC_AUC']:.4f} | F1={base_test['F1']:.4f} | Recall@{args.thr:.2f}={base_test['Recall@Thr']:.4f} | Brier={base_test['Brier']:.4f}")

    # save artifacts
    art = pathlib.Path(args.artifacts_dir); (art/"metrics").mkdir(parents=True, exist_ok=True)

    # leaderboard
    lb = pd.DataFrame(trials)
    lb_path = art/"metrics"/"xgb_tuning_leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    # best config + metrics
    out = {
        "cv_folds": args.cv_folds, "cv_repeats": args.cv_repeats,
        "trials": args.trials,
        "best_cv_ap_agg": best["ap_agg"],
        "best_cv_ap_mean": best["ap_mean"],
        "best_cv_ap_std": best["ap_std"],
        "best_config": best["cfg"],
        "final_train_on": args.train_on,
        "val_metrics": result["val"],
        "test_metrics": result["test"],
        "baseline_val": base_val,
        "baseline_test": base_test,
        "xgboost_version": xgb.__version__
    }
    (art/"metrics"/"xgb_tuning_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # save probas
    if args.save_probas:
        (art/"probas").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y": y_val,  "proba": result["proba_val"]}).to_csv(art/"probas"/"xgb_best_val.csv", index=False)
        pd.DataFrame({"y": y_test, "proba": result["proba_test"]}).to_csv(art/"probas"/"xgb_best_test.csv", index=False)

    # save model
    if args.save_model:
        (art/"models").mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(result["model"], art/"models"/"xgb_best.joblib")

    # feature importances (gain)
    try:
        booster = result["model"].get_booster()
        score = booster.get_score(importance_type="gain")
        imp = pd.DataFrame(sorted(score.items(), key=lambda kv: kv[1], reverse=True), columns=["feature","gain"])
        feat_names = load_feature_names(args.processed_dir)
        if feat_names:
            def map_name(s):
                if s.startswith("f"):
                    idx = int(s[1:])
                    return feat_names[idx] if 0 <= idx < len(feat_names) else s
                return s
            imp["feature_name"] = imp["feature"].map(map_name)
            imp = imp[["feature_name","feature","gain"]]
        imp.to_csv(art/"metrics"/"xgb_feature_importance_gain.csv", index=False)
    except Exception:
        pass

    print(f"\n[OK] Лидерборд: {lb_path}")
    print(f"[OK] Сводка:    {art/'metrics'/'xgb_tuning_summary.json'}")
    if args.save_model:
        print(f"[OK] Модель:    {art/'models'/'xgb_best.joblib'}")
    if args.save_probas:
        print(f"[OK] Пробасы:   {art/'probas'}")


if __name__ == "__main__":
    main()
