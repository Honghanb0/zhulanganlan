"""Train and evaluate the leakage-aware hierarchical security-event model.

Examples:
    python train_security_model_v3.py --mode all
    python train_security_model_v3.py --mode evaluate --max-rows 300000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from security_features import SECURITY_PATTERNS, SecurityFeaturePipeline, make_group_ids
from security_metrics import evaluate_predictions, optimize_thresholds


LABEL_MAP = {"benign": 0, "suspicious": 1, "malicious": 2}
LABEL_NAMES = ["benign", "suspicious", "malicious"]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("security_model_config.json")


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    # Make every configured artifact path independent of the caller's cwd.
    for key, value in config.get("paths", {}).items():
        candidate = Path(value)
        if not candidate.is_absolute():
            config["paths"][key] = str((config_path.parent / candidate).resolve())
    return config


def set_reproducible(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_labeled_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "label_binary" not in df:
        raise ValueError(f"{path} has no label_binary column")
    df = df[df["label_binary"].isin(LABEL_MAP)].copy()
    df["label"] = df["label_binary"].map(LABEL_MAP).astype(np.int8)
    return df


def mixed_sample(df: pd.DataFrame, ratio: float, seed: int,
                 max_rows: int | None = None) -> pd.DataFrame:
    """Keep all threats and mix hard/random benign examples without synthetic rows."""
    threat = df[df.label.ne(0)]
    benign = df[df.label.eq(0)]
    target_benign = min(len(benign), int(len(threat) * ratio))
    if max_rows is not None and len(threat) + target_benign > max_rows:
        threat_target = min(len(threat), max(2, int(max_rows / (ratio + 1))))
        threat = threat.groupby("label", group_keys=False).sample(
            n=min(threat_target // 2, int(threat.label.value_counts().min())), random_state=seed
        ) if threat_target < len(threat) else threat
        target_benign = min(len(benign), max_rows - len(threat))
    msg = benign["message_sanitized"].fillna("").astype(str).str.lower()
    hard_pattern = "|".join(f"(?:{p})" for p in SECURITY_PATTERNS.values())
    hard_mask = msg.str.contains(hard_pattern, regex=True, na=False)
    hard = benign[hard_mask]
    hard_n = min(len(hard), target_benign // 2)
    hard_pick = hard.sample(n=hard_n, random_state=seed) if hard_n else hard.iloc[:0]
    remaining = benign.drop(index=hard_pick.index)
    random_n = min(len(remaining), target_benign - hard_n)
    random_pick = remaining.sample(n=random_n, random_state=seed + 1) if random_n else remaining.iloc[:0]
    sampled = pd.concat([threat, hard_pick, random_pick], ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def model_params(config: dict, objective: str, seed: int, class_weight: dict | str | None) -> dict:
    cfg = config["model"]
    params = {
        "objective": objective,
        "n_estimators": cfg["n_estimators"],
        "learning_rate": cfg["learning_rate"],
        "num_leaves": cfg["num_leaves"],
        "max_depth": cfg["max_depth"],
        "min_child_samples": cfg["min_child_samples"],
        "subsample": cfg["subsample"],
        "subsample_freq": 1,
        "colsample_bytree": cfg["colsample_bytree"],
        "reg_alpha": cfg["reg_alpha"],
        "reg_lambda": cfg["reg_lambda"],
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": cfg["n_jobs"],
        "verbosity": -1,
        "class_weight": class_weight,
    }
    if objective == "multiclass":
        params["num_class"] = 3
    return params


def fit_one(params: dict, x_train: pd.DataFrame, y_train: np.ndarray,
            categorical: list[str], x_val: pd.DataFrame | None = None,
            y_val: np.ndarray | None = None, early_stopping: int = 100) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    kwargs = {"categorical_feature": categorical, "callbacks": [lgb.log_evaluation(0)]}
    if x_val is not None:
        kwargs.update({
            "eval_set": [(x_val, y_val)],
            "callbacks": [lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
        })
    model.fit(x_train, y_train, **kwargs)
    return model


def cross_validate(df: pd.DataFrame, config: dict) -> tuple[dict, dict]:
    seed = config["seed"]
    folds = config["validation"]["folds"]
    groups = make_group_ids(df)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    y = df.label.to_numpy(dtype=np.int8)
    multi_oof = np.zeros((len(df), 3), dtype=np.float32)
    threat_oof = np.zeros(len(df), dtype=np.float32)
    subtype_oof = np.zeros(len(df), dtype=np.float32)
    feature_importance = []
    iterations = {"multiclass": [], "threat": [], "subtype": []}

    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, y, groups), start=1):
        started = time.time()
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        fe = SecurityFeaturePipeline(
            hash_buckets=config["features"]["hash_buckets"],
            rare_min_count=config["features"]["rare_min_count"],
            window_seconds=tuple(config["features"]["window_seconds"]),
        )
        x_train = fe.fit_transform(train_df)
        x_val = fe.transform(val_df)
        y_train, y_val = y[train_idx], y[val_idx]
        categorical = fe.categorical_features_
        stop = config["model"]["early_stopping_rounds"]

        multi = fit_one(model_params(config, "multiclass", seed + fold, {0: 1.0, 1: 2.5, 2: 2.0}),
                        x_train, y_train, categorical, x_val, y_val, stop)
        threat = fit_one(model_params(config, "binary", seed + fold + 10, {0: 1.0, 1: 2.5}),
                         x_train, (y_train != 0).astype(np.int8), categorical,
                         x_val, (y_val != 0).astype(np.int8), stop)
        threat_train = y_train != 0
        threat_val = y_val != 0
        subtype = fit_one(model_params(config, "binary", seed + fold + 20, {0: 1.2, 1: 1.0}),
                          x_train.loc[threat_train], (y_train[threat_train] == 2).astype(np.int8), categorical,
                          x_val.loc[threat_val], (y_val[threat_val] == 2).astype(np.int8), stop)

        multi_oof[val_idx] = multi.predict_proba(x_val)
        threat_oof[val_idx] = threat.predict_proba(x_val)[:, 1]
        subtype_oof[val_idx] = subtype.predict_proba(x_val)[:, 1]
        feature_importance.append(pd.Series(multi.feature_importances_, index=x_train.columns, name=f"fold_{fold}"))
        for name, model in (("multiclass", multi), ("threat", threat), ("subtype", subtype)):
            iterations[name].append(int(model.best_iteration_ or config["model"]["n_estimators"]))
        del x_train, x_val, multi, threat, subtype
        print(f"fold {fold}/{folds} complete: train={len(train_idx):,}, valid={len(val_idx):,}, "
              f"elapsed={time.time() - started:.1f}s", flush=True)

    baseline_pred = np.argmax(multi_oof, axis=1).astype(np.int8)
    baseline_metrics = evaluate_predictions(y, baseline_pred, multi_oof)
    selection, optimized_pred, optimized_proba = optimize_thresholds(
        y, multi_oof, threat_oof, subtype_oof, config["threshold_search"]
    )
    importance = pd.concat(feature_importance, axis=1).mean(axis=1).sort_values(ascending=False)
    importance = (importance / max(importance.sum(), 1)).head(40)
    result = {
        "validation": {
            "strategy": "StratifiedGroupKFold; exact product+message templates never cross folds",
            "folds": folds,
            "rows": len(df),
            "class_distribution": df.label_binary.value_counts().to_dict(),
        },
        "baseline_single_multiclass_argmax": baseline_metrics,
        "optimized_hierarchical_ensemble": selection["metrics"],
        "selected_policy": {k: v for k, v in selection.items() if k != "metrics"},
        "best_iterations": {k: int(np.median(v)) for k, v in iterations.items()},
        "feature_importance_gain_normalized": {k: float(v) for k, v in importance.items()},
    }
    oof = {
        "y": y,
        "baseline_pred": baseline_pred,
        "optimized_pred": optimized_pred,
        "optimized_proba": optimized_proba,
    }
    return result, oof


def train_final(df: pd.DataFrame, config: dict, evaluation: dict) -> Path:
    seed = config["seed"]
    fe = SecurityFeaturePipeline(
        hash_buckets=config["features"]["hash_buckets"],
        rare_min_count=config["features"]["rare_min_count"],
        window_seconds=tuple(config["features"]["window_seconds"]),
    )
    x = fe.fit_transform(df)
    y = df.label.to_numpy(dtype=np.int8)
    iterations = evaluation["best_iterations"]
    models = {}
    specifications = [
        ("multiclass", "multiclass", y, np.ones(len(y), dtype=bool), {0: 1.0, 1: 2.5, 2: 2.0}),
        ("threat", "binary", (y != 0).astype(np.int8), np.ones(len(y), dtype=bool), {0: 1.0, 1: 2.5}),
        ("subtype", "binary", (y == 2).astype(np.int8), y != 0, {0: 1.2, 1: 1.0}),
    ]
    for name, objective, target, mask, weights in specifications:
        params = model_params(config, objective, seed, weights)
        params["n_estimators"] = max(50, iterations[name])
        models[name] = fit_one(params, x.loc[mask], target[mask], fe.categorical_features_)
        print(f"trained final {name}: rows={int(mask.sum()):,}, trees={params['n_estimators']}", flush=True)

    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": config["version"],
        "config": config,
        "feature_pipeline": fe,
        "models": models,
        "policy": evaluation["selected_policy"],
        "label_names": LABEL_NAMES,
        "training_rows": len(df),
    }
    artifact_path = model_dir / "security_model_v3.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    return artifact_path


def save_report(evaluation: dict, config: dict, sampled: pd.DataFrame) -> Path:
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
    base = evaluation["baseline_single_multiclass_argmax"]
    new = evaluation["optimized_hierarchical_ensemble"]
    cm_base, cm_new = base["confusion_matrix"], new["confusion_matrix"]
    policy = evaluation["selected_policy"]
    lines = [
        "# 逐浪安澜恶意安全事件模型 v3 评测报告", "",
        "## 可复现设置", "",
        f"- 数据：`{config['paths']['train']}`，混合采样后 {len(sampled):,} 行，随机种子 {config['seed']}。",
        f"- 验证：{evaluation['validation']['strategy']}。",
        "- 基线：v3 特征上的单个多分类 LightGBM + argmax；完整版：威胁二分类/威胁子类/多分类三级集成。", "",
        "## 指标对比", "",
        "| 方案 | Precision(macro) | Recall(macro) | F1(macro) | AUC-OVR | Threat Recall | Malicious Recall | 综合得分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| v3 单模型消融 | {base['precision_macro']:.6f} | {base['recall_macro']:.6f} | {base['f1_macro']:.6f} | {base.get('auc_ovr_macro') or 0:.6f} | {base['threat_recall']:.6f} | {base['recall_malicious']:.6f} | {base['competition_score']:.6f} |",
        f"| v3 层次集成 | {new['precision_macro']:.6f} | {new['recall_macro']:.6f} | {new['f1_macro']:.6f} | {new.get('auc_ovr_macro') or 0:.6f} | {new['threat_recall']:.6f} | {new['recall_malicious']:.6f} | **{new['competition_score']:.6f}** |",
        "", "## 混淆矩阵", "", "行是真实类别，列是预测类别，顺序均为 benign / suspicious / malicious。", "",
        "### v3 单模型消融", "", "| | benign | suspicious | malicious |", "|---|---:|---:|---:|",
        *[f"| {LABEL_NAMES[i]} | {cm_base[i][0]} | {cm_base[i][1]} | {cm_base[i][2]} |" for i in range(3)],
        "", "### v3 层次集成", "", "| | benign | suspicious | malicious |", "|---|---:|---:|---:|",
        *[f"| {LABEL_NAMES[i]} | {cm_new[i][0]} | {cm_new[i][1]} | {cm_new[i][2]} |" for i in range(3)],
        "", "## 阈值依据", "",
        f"官方完整加权得分网格寻优：threat_threshold={policy['threat_threshold']}, malicious_threshold={policy['malicious_threshold']}，",
        f"同时约束 threat recall ≥ {config['threshold_search']['minimum_threat_recall']}、malicious recall ≥ {config['threshold_search']['minimum_malicious_recall']}。",
        "", "## 重要特征（gain 归一化 Top 20）", "",
        "| 特征 | 重要性 |", "|---|---:|",
        *[f"| {name} | {value:.6f} |" for name, value in list(evaluation["feature_importance_gain_normalized"].items())[:20]],
    ]
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=["evaluate", "train", "all"], default="all")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Override evaluation sample cap; final training still uses configured mixed sample")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_reproducible(config["seed"])
    df = load_labeled_data(config["paths"]["train"])
    configured_cap = config["validation"]["max_rows"]
    eval_cap = args.max_rows if args.max_rows is not None else configured_cap
    sampled_eval = mixed_sample(df, config["sampling"]["benign_to_threat_ratio"], config["seed"], eval_cap)
    evaluation = None
    metrics_path = Path(config["paths"]["report_dir"]) / "metrics.json"
    if args.mode in {"evaluate", "all"}:
        print(f"evaluation sample: {len(sampled_eval):,}; {sampled_eval.label_binary.value_counts().to_dict()}")
        evaluation, _ = cross_validate(sampled_eval, config)
        report = save_report(evaluation, config, sampled_eval)
        print(f"evaluation report: {report}")
        print(json.dumps(evaluation["optimized_hierarchical_ensemble"], ensure_ascii=False, indent=2))
    if args.mode in {"train", "all"}:
        if evaluation is None:
            if not metrics_path.exists():
                raise FileNotFoundError("Run --mode evaluate first; threshold policy and tree counts are required")
            evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
        final_sample = mixed_sample(df, config["sampling"]["benign_to_threat_ratio"], config["seed"])
        artifact = train_final(final_sample, config, evaluation)
        print(f"model artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
