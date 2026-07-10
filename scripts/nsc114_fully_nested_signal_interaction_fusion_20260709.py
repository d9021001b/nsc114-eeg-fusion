#!/usr/bin/env python3
"""Fully nested signal-interaction fusion audit for the first NSC114 paper.

This script tests whether second-paper-style signal-interaction features can
improve the first paper's 114-participant source-domain ROC/PRC claim.

Validation contract:
- patient/subject is the prediction unit;
- every outer fold trains the signal branch only on outer-train subjects;
- every outer fold chooses both the signal configuration and fusion weight
  using only an inner CV over outer-train subjects;
- outer-test subjects are used only once, after config/weight selection;
- the existing first-paper corrected bagged score is used as a fixed OOF base
  branch, while the new signal branch and fusion weight are fully nested here.

Claim boundary:
The new branch is fully nested. The base branch is a previously frozen OOF
prediction file from the first paper, not recomputed inside this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nsc114_source_signal_interaction_audit_20260709 import (  # noqa: E402
    collect_eeg_features,
    collect_physio_features,
    dump_json,
    fit_score,
    load_labels,
    matrix,
    merge_rows,
    metrics,
    natural_key,
    select_fold,
    write_csv,
)


FAMILIES = {
    "signal_main_plus_eeg": ("SIG/", "EEG_ALL/"),
    "signal_all": ("SIG/", "SIG_RATIO/", "SIG_DIFF/", "EEG_ALL/"),
}


def robust_unit_apply(train_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.nanpercentile(train_scores, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(train_scores))
        hi = float(np.nanmax(train_scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(train_scores, 0.5, dtype=float), np.full_like(test_scores, 0.5, dtype=float)
    return (
        np.clip((train_scores - lo) / (hi - lo), 0.0, 1.0),
        np.clip((test_scores - lo) / (hi - lo), 0.0, 1.0),
    )


def load_base_scores(path: Path, score_col: str, subjects: list[str], y: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    df["subject_id"] = df["subject_id"].astype(str)
    if score_col not in df.columns:
        raise ValueError(f"missing score column {score_col!r} in {path}")
    if "true_label" in df.columns:
        label_map = df.drop_duplicates("subject_id").set_index("subject_id")["true_label"].astype(int).to_dict()
        mismatches = [s for i, s in enumerate(subjects) if s in label_map and int(label_map[s]) != int(y[i])]
        if mismatches:
            raise ValueError(f"base score label mismatch for subjects: {mismatches[:10]}")
    table = df.drop_duplicates("subject_id").set_index("subject_id")
    missing = [s for s in subjects if s not in table.index]
    if missing:
        raise ValueError(f"missing base scores for subjects: {missing[:10]}")
    return table.loc[subjects, score_col].to_numpy(dtype=float)


def inner_oof_signal_scores(
    data,
    outer_train_idx: np.ndarray,
    model_name: str,
    top_k: int,
    seed: int,
    inner_splits: int,
) -> tuple[np.ndarray, Counter[str]]:
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    oof = np.full(len(outer_train_idx), np.nan, dtype=float)
    selected_counter: Counter[str] = Counter()
    X_outer = data.X[outer_train_idx]
    y_outer = data.y[outer_train_idx]
    for inner_fold, (tr_local, va_local) in enumerate(inner.split(X_outer, y_outer), start=1):
        tr_idx = outer_train_idx[tr_local]
        va_idx = outer_train_idx[va_local]
        xtr, xva, selected = select_fold(
            data.X[tr_idx],
            data.y[tr_idx],
            data.X[va_idx],
            data.feature_names,
            top_k,
        )
        oof[va_local] = fit_score(model_name, xtr, data.y[tr_idx], xva, seed + inner_fold)
        selected_counter.update(selected)
    if np.isnan(oof).any():
        raise ValueError("inner OOF signal scores contain NaN")
    return oof, selected_counter


def train_signal_outer(data, train_idx: np.ndarray, test_idx: np.ndarray, model_name: str, top_k: int, seed: int):
    xtr, xte, selected = select_fold(
        data.X[train_idx],
        data.y[train_idx],
        data.X[test_idx],
        data.feature_names,
        top_k,
    )
    test_score = fit_score(model_name, xtr, data.y[train_idx], xte, seed)
    train_score = fit_score(model_name, xtr, data.y[train_idx], xtr, seed + 7000)
    return train_score, test_score, selected


def choose_inner_config_and_weight(
    datasets: dict[str, object],
    base_scores: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    inner_splits: int,
    top_ks: list[int],
    models: list[str],
    weights: np.ndarray,
    objective: str,
):
    best = None
    inner_rows = []
    feature_rows = []
    for family, data in datasets.items():
        max_k = len(data.feature_names)
        if max_k == 0:
            continue
        for top_k in top_ks:
            if top_k > max_k:
                continue
            for model_name in models:
                signal_oof, selected_counter = inner_oof_signal_scores(
                    data, train_idx, model_name, top_k, seed + len(inner_rows) * 17, inner_splits
                )
                base_train = base_scores[train_idx]
                base_unit, _ = robust_unit_apply(base_train, base_train)
                signal_unit, _ = robust_unit_apply(signal_oof, signal_oof)
                for w_signal in weights:
                    fused = (1.0 - w_signal) * base_unit + w_signal * signal_unit
                    met = metrics(y[train_idx], fused)
                    if objective == "min_metric":
                        obj = float(min(met["AUROC"], met["AUPRC"]))
                    elif objective == "auprc_then_auroc":
                        obj = float(met["AUPRC"] + 0.05 * met["AUROC"])
                    else:
                        raise ValueError(f"unknown objective: {objective}")
                    row = {
                        "family": family,
                        "model": model_name,
                        "top_k": top_k,
                        "w_signal": float(w_signal),
                        "objective_value": obj,
                        **met,
                    }
                    inner_rows.append(row)
                    if best is None or obj > best["objective_value"]:
                        best = row
                for feature, count in selected_counter.most_common(50):
                    feature_rows.append(
                        {
                            "family": family,
                            "model": model_name,
                            "top_k": top_k,
                            "feature": feature,
                            "inner_selected_count": int(count),
                        }
                    )
    if best is None:
        raise ValueError("no inner config evaluated")
    return best, inner_rows, feature_rows


def run_one_seed(
    datasets: dict[str, object],
    base_scores: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    seed: int,
    n_splits: int,
    inner_splits: int,
    top_ks: list[int],
    models: list[str],
    weights: np.ndarray,
    objective: str,
):
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fused_scores = np.full(len(y), np.nan, dtype=float)
    signal_scores = np.full(len(y), np.nan, dtype=float)
    fold_details = []
    inner_rows_all = []
    feature_rows_all = []
    pred_rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
        best, inner_rows, feature_rows = choose_inner_config_and_weight(
            datasets,
            base_scores,
            y,
            train_idx,
            seed + fold * 100,
            inner_splits,
            top_ks,
            models,
            weights,
            objective,
        )
        data = datasets[best["family"]]
        train_signal, test_signal, selected = train_signal_outer(
            data,
            train_idx,
            test_idx,
            best["model"],
            int(best["top_k"]),
            seed + fold * 1000,
        )
        base_train_unit, base_test_unit = robust_unit_apply(base_scores[train_idx], base_scores[test_idx])
        signal_train_unit, signal_test_unit = robust_unit_apply(train_signal, test_signal)
        fused = (1.0 - float(best["w_signal"])) * base_test_unit + float(best["w_signal"]) * signal_test_unit
        fused_scores[test_idx] = fused
        signal_scores[test_idx] = signal_test_unit
        fold_details.append(
            {
                "seed": seed,
                "outer_fold": fold,
                "train_cases": int(len(train_idx)),
                "test_cases": int(len(test_idx)),
                "selected_family": best["family"],
                "selected_model": best["model"],
                "selected_top_k": int(best["top_k"]),
                "selected_w_signal": float(best["w_signal"]),
                "inner_objective_value": float(best["objective_value"]),
                "inner_AUROC": float(best["AUROC"]),
                "inner_AUPRC": float(best["AUPRC"]),
                "selected_feature_count": int(len(selected)),
            }
        )
        for row in inner_rows:
            inner_rows_all.append({"seed": seed, "outer_fold": fold, **row})
        for row in feature_rows:
            feature_rows_all.append({"seed": seed, "outer_fold": fold, **row})
        for local_i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "seed": seed,
                    "outer_fold": fold,
                    "subject_id": subjects[idx],
                    "true_label": int(y[idx]),
                    "base_score": float(base_scores[idx]),
                    "signal_score_unit": float(signal_test_unit[local_i]),
                    "fused_score": float(fused[local_i]),
                    "pred": int(fused[local_i] >= 0.5),
                    "selected_family": best["family"],
                    "selected_model": best["model"],
                    "selected_top_k": int(best["top_k"]),
                    "selected_w_signal": float(best["w_signal"]),
                }
            )
    if np.isnan(fused_scores).any():
        raise ValueError("outer fused scores contain NaN")
    seed_metric = metrics(y, fused_scores)
    seed_metric.update({"seed": seed, "method": "fully_nested_base_plus_signal_interaction"})
    return seed_metric, pred_rows, fold_details, inner_rows_all, feature_rows_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/nsc_dataset_images/manifest.csv")
    parser.add_argument("--physio-root", type=Path, default=REPO_ROOT / "data/physiology-csv")
    parser.add_argument("--eeg-root", type=Path, default=REPO_ROOT / "data/eeg-csv-data-by-class")
    parser.add_argument("--base-predictions", type=Path, default=REPO_ROOT / "analysis/nsc_restricted_subject_bagging_tail_stats_corrected_20260521/bagged_predictions.csv")
    parser.add_argument("--base-score-col", default="bagged_score")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis/nsc114_fully_nested_signal_interaction_fusion_20260709")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260709, 20260719)))
    parser.add_argument("--top-ks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64, 96, 128])
    parser.add_argument("--models", nargs="+", default=["logreg", "prototype"])
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--w-max", type=float, default=0.50)
    parser.add_argument("--w-step", type=float, default=0.05)
    parser.add_argument("--objective", choices=["min_metric", "auprc_then_auroc"], default="min_metric")
    parser.add_argument("--pass-auroc", type=float, default=0.797146)
    parser.add_argument("--pass-auprc", type=float, default=0.798864)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.manifest.resolve())
    subjects = sorted(labels, key=natural_key)
    y = np.asarray([labels[s] for s in subjects], dtype=int)

    physio_rows, physio_audit = collect_physio_features(set(subjects), args.physio_root.resolve())
    eeg_rows, eeg_audit = collect_eeg_features(set(subjects), args.eeg_root.resolve())
    rows = merge_rows(physio_rows, eeg_rows)
    datasets = {family: matrix(subjects, y, rows, prefixes) for family, prefixes in FAMILIES.items()}
    base_scores = load_base_scores(args.base_predictions.resolve(), args.base_score_col, subjects, y)
    weights = np.round(np.arange(0.0, args.w_max + 1e-9, args.w_step), 8)

    dataset_audit = []
    for family, data in datasets.items():
        dataset_audit.append(
            {
                "family": family,
                "case_count": int(len(data.subjects)),
                "positive_count": int(data.y.sum()),
                "negative_count": int(len(data.y) - data.y.sum()),
                "feature_count": int(len(data.feature_names)),
                "missing_entries": int(np.isnan(data.X).sum()),
                "total_entries": int(data.X.size),
                "missing_rate": float(np.isnan(data.X).mean()) if data.X.size else math.nan,
            }
        )

    seed_metrics = []
    pred_rows = []
    fold_rows = []
    inner_rows = []
    feature_rows = []
    for seed in args.seeds:
        seed_metric, seed_preds, seed_folds, seed_inner, seed_features = run_one_seed(
            datasets,
            base_scores,
            y,
            subjects,
            seed,
            args.n_splits,
            args.inner_splits,
            args.top_ks,
            args.models,
            weights,
            args.objective,
        )
        seed_metrics.append(seed_metric)
        pred_rows.extend(seed_preds)
        fold_rows.extend(seed_folds)
        inner_rows.extend(seed_inner)
        feature_rows.extend(seed_features)

    pred_df = pd.DataFrame(pred_rows)
    bagged = (
        pred_df.groupby(["subject_id", "true_label"], as_index=False)
        .agg(fused_score=("fused_score", "mean"), base_score=("base_score", "mean"))
        .sort_values("subject_id", key=lambda s: s.map(natural_key))
    )
    bagged_metrics = metrics(bagged["true_label"].to_numpy(dtype=int), bagged["fused_score"].to_numpy(dtype=float))
    bagged_metrics.update(
        {
            "method": "bagged_fully_nested_base_plus_signal_interaction",
            "seed_count": int(len(args.seeds)),
        }
    )
    base_metrics = metrics(y, base_scores)
    base_metrics.update({"method": "base_corrected_bagged_score"})
    seed_df = pd.DataFrame(seed_metrics)
    aggregate = {
        "method": "fully_nested_base_plus_signal_interaction",
        "seed_count": int(len(seed_metrics)),
    }
    for metric in ["AUROC", "AUPRC", "balanced_accuracy", "sensitivity", "specificity", "accuracy"]:
        values = seed_df[metric].to_numpy(dtype=float)
        aggregate[f"{metric}_mean"] = float(np.nanmean(values))
        aggregate[f"{metric}_std"] = float(np.nanstd(values))
        aggregate[f"{metric}_min"] = float(np.nanmin(values))
        aggregate[f"{metric}_max"] = float(np.nanmax(values))

    pass_gate = bool(bagged_metrics["AUROC"] >= args.pass_auroc and bagged_metrics["AUPRC"] >= args.pass_auprc)
    gate = {
        "passed": pass_gate,
        "criteria": {
            "bagged_AUROC_gte": args.pass_auroc,
            "bagged_AUPRC_gte": args.pass_auprc,
        },
        "bagged_AUROC": bagged_metrics["AUROC"],
        "bagged_AUPRC": bagged_metrics["AUPRC"],
        "base_AUROC": base_metrics["AUROC"],
        "base_AUPRC": base_metrics["AUPRC"],
        "decision": "eligible_to_update_first_paper" if pass_gate else "do_not_update_first_paper_headline",
    }

    write_csv(out_dir / "dataset_audit.csv", dataset_audit)
    write_csv(out_dir / "seed_metrics.csv", seed_metrics)
    write_csv(out_dir / "outer_fold_selected_params.csv", fold_rows)
    write_csv(out_dir / "inner_config_weight_search.csv", inner_rows)
    write_csv(out_dir / "selected_features_long.csv", feature_rows)
    write_csv(out_dir / "outer_predictions.csv", pred_rows)
    bagged.to_csv(out_dir / "bagged_predictions.csv", index=False, encoding="utf-8-sig")
    write_csv(out_dir / "metrics_summary.csv", [base_metrics, bagged_metrics, aggregate])
    dump_json(
        out_dir / "manifest.json",
        {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subjects": int(len(subjects)),
            "cases_by_label": dict(Counter(str(v) for v in y)),
            "base_predictions": str(args.base_predictions.resolve()),
            "base_score_col": args.base_score_col,
            "out_dir": str(out_dir),
            "validation": f"patient-aware stratified {args.n_splits}-fold outer CV with inner {args.inner_splits}-fold config and fusion-weight selection",
            "claim_boundary": "signal branch and fusion weight are fully nested; base branch is a previously frozen corrected OOF score file",
            "families": FAMILIES,
            "models": args.models,
            "top_ks": args.top_ks,
            "weights": [float(w) for w in weights],
            "objective": args.objective,
            "dataset_audit": dataset_audit,
            "physio_audit": {
                "files_by_signal": physio_audit["files_by_signal"],
            },
            "eeg_audit": {
                "subjects_with_eeg": eeg_audit["subjects_with_eeg"],
                "subjects_without_eeg": eeg_audit["subjects_without_eeg"],
            },
            "base_metrics": base_metrics,
            "bagged_metrics": bagged_metrics,
            "seed_metric_aggregate": aggregate,
            "gate": gate,
            "outputs": {
                "dataset_audit": str(out_dir / "dataset_audit.csv"),
                "seed_metrics": str(out_dir / "seed_metrics.csv"),
                "outer_fold_selected_params": str(out_dir / "outer_fold_selected_params.csv"),
                "inner_config_weight_search": str(out_dir / "inner_config_weight_search.csv"),
                "selected_features_long": str(out_dir / "selected_features_long.csv"),
                "outer_predictions": str(out_dir / "outer_predictions.csv"),
                "bagged_predictions": str(out_dir / "bagged_predictions.csv"),
                "metrics_summary": str(out_dir / "metrics_summary.csv"),
            },
        },
    )
    print(json.dumps({"gate": gate, "bagged_metrics": bagged_metrics, "seed_metric_aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
