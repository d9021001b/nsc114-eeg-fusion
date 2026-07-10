#!/usr/bin/env python3
"""Current nested-fusion NSC114 Paper-1 pipeline.

Removes the last remaining optimism: the published fusion reads a PRECOMPUTED base OOF
score file. Because the base folds differ from the fusion folds, the base scores of the
fusion's *training* subjects come from base models that had seen the fusion's *test*
subject. Standard stacking practice, but not airtight.

Inside every fusion outer fold, this script regenerates the EEG and signal branches using
ONLY that fold's training subjects:
  (a) base OOF for the training subjects  -> nested CV *within* the training subjects
  (b) base score for the test subjects    -> config chosen by inner CV on training
                                             subjects, model fit on all training subjects
The image score remains a fixed, separately cross-fitted OOF input. It is not regenerated
on the fusion outer folds, so this script must not be described as a fully nested image
pipeline. The fixed score is the raw/2D uncertain-band output stored in the ``mi_max``
column, not a pure image-only maximum score.

Signal branch, its config, and the base/signal fusion weight remain nested as published.
No performance gate: whatever comes out is reported.
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nsc114_fully_nested_signal_interaction_fusion_20260709 as fm  # noqa: E402
import nsc_restricted_subject_bagging_tail_stats as bm  # noqa: E402
from nsc_eeg_csv_fusion_ablation import (  # noqa: E402
    load_base_scores as load_mi,
    metrics,
    robust_unit_fit,
)
from nsc114_fully_nested_signal_interaction_fusion_20260709 import (  # noqa: E402
    FAMILIES,
    collect_eeg_features,
    collect_physio_features,
    load_labels,
    matrix,
    merge_rows,
    natural_key,
)

CS = [0.10, 0.25, 0.50]
TOPKS = [160, 320, 480]
PENALTY = "l1"


def _obj(met, objective):
    return min(met["AUROC"], met["AUPRC"]) if objective == "min_metric" else met["AUPRC"]


def select_base_cfg(X, y, mi, seed, inner_splits, w_eegs, objective):
    """Choose (C, top_k, w_eeg) using ONLY the rows given (an outer-training set)."""
    base_scaled, _ = robust_unit_fit(mi, mi)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    splits = list(inner.split(X, y))
    best_obj, best = -1.0, None
    for C in CS:
        for k in TOPKS:
            oof = np.zeros(len(y), dtype=float)
            for j, (a, b) in enumerate(splits, start=1):
                imp, sel = bm.select_topk(X[a], y[a], k)
                if len(sel) == 0:
                    oof[b] = 0.5
                    continue
                clf = bm.make_model(PENALTY, C, seed + j)
                clf.fit(imp.transform(X[a])[:, sel], y[a])
                oof[b] = clf.predict_proba(imp.transform(X[b])[:, sel])[:, 1]
            eeg_scaled, _ = robust_unit_fit(oof, oof)
            for w in w_eegs:
                o = _obj(metrics(y, (1.0 - w) * base_scaled + w * eeg_scaled), objective)
                if o > best_obj:
                    best_obj, best = o, (C, k, float(w))
    return best


def base_oof_and_test(X, y, mi, tr, te, seed, inner_splits, base_oof_splits, w_eegs, objective):
    """(a) OOF base scores for `tr` computed only from `tr`; (b) base score for `te`."""
    Xtr, ytr, mitr = X[tr], y[tr], mi[tr]

    oof = np.zeros(len(tr), dtype=float)
    sub = StratifiedKFold(n_splits=base_oof_splits, shuffle=True, random_state=seed)
    for f, (a, b) in enumerate(sub.split(Xtr, ytr), start=1):
        C, k, w = select_base_cfg(Xtr[a], ytr[a], mitr[a], seed + f * 7, inner_splits, w_eegs, objective)
        imp, sel = bm.select_topk(Xtr[a], ytr[a], k)
        clf = bm.make_model(PENALTY, C, seed + f * 11)
        Xa = imp.transform(Xtr[a])[:, sel]
        clf.fit(Xa, ytr[a])
        e_a = clf.predict_proba(Xa)[:, 1]
        e_b = clf.predict_proba(imp.transform(Xtr[b])[:, sel])[:, 1]
        _, mi_b = robust_unit_fit(mitr[a], mitr[b])
        _, e_bu = robust_unit_fit(e_a, e_b)
        oof[b] = (1.0 - w) * mi_b + w * e_bu

    C, k, w = select_base_cfg(Xtr, ytr, mitr, seed + 9973, inner_splits, w_eegs, objective)
    imp, sel = bm.select_topk(Xtr, ytr, k)
    clf = bm.make_model(PENALTY, C, seed + 555)
    Xa = imp.transform(Xtr)[:, sel]
    clf.fit(Xa, ytr)
    e_tr = clf.predict_proba(Xa)[:, 1]
    e_te = clf.predict_proba(imp.transform(X[te])[:, sel])[:, 1]
    _, mi_te = robust_unit_fit(mitr, mi[te])
    _, e_teu = robust_unit_fit(e_tr, e_te)
    base_te = (1.0 - w) * mi_te + w * e_teu
    return oof, base_te, {"C": C, "top_k": k, "w_eeg": w}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/nsc_dataset_images/manifest.csv")
    ap.add_argument("--physio-root", type=Path, default=ROOT / "data/physiology-csv")
    ap.add_argument("--eeg-root", type=Path, default=ROOT / "data/eeg-csv-data-by-class")
    ap.add_argument("--eeg-cache", type=Path,
                    default=ROOT / "analysis/nsc_eeg_csv_fusion_ablation_20260520/eeg_feature_cache")
    ap.add_argument("--mi-predictions", type=Path,
                    default=ROOT / "analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis/nsc114_airtight_fully_nested_20260710")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20260709, 20260719)))
    ap.add_argument("--top-ks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    ap.add_argument("--models", nargs="+", default=["logreg", "prototype"])
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--base-oof-splits", type=int, default=10)
    ap.add_argument("--w-max", type=float, default=0.35)
    ap.add_argument("--w-step", type=float, default=0.05)
    ap.add_argument("--w-eeg-max", type=float, default=0.25)
    ap.add_argument("--w-eeg-step", type=float, default=0.05)
    ap.add_argument("--objective", default="min_metric")
    args = ap.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.manifest.resolve())
    subjects = sorted(labels, key=natural_key)
    y = np.asarray([labels[s] for s in subjects], dtype=int)
    n = len(y)
    print(f"[data] subjects={n} class0={(y==0).sum()} class1={(y==1).sum()}", flush=True)

    t0 = time.time()
    physio_rows, physio_audit = collect_physio_features(set(subjects), args.physio_root.resolve())
    eeg_rows, eeg_audit_sig = collect_eeg_features(set(subjects), args.eeg_root.resolve())
    rows = merge_rows(physio_rows, eeg_rows)
    datasets = {fam: matrix(subjects, y, rows, pre) for fam, pre in FAMILIES.items()}
    print(f"[data] signal families built ({time.time()-t0:.0f}s)", flush=True)

    X_eeg, feat_names, eeg_audit = bm.build_subject_eeg_features_tail_stats(
        subjects, args.eeg_root.resolve(), args.eeg_cache.resolve(),
        include_counts=False, class_aware_features=False)
    mi = load_mi(args.mi_predictions.resolve(), subjects)["mi_max"]
    print(f"[data] X_eeg={X_eeg.shape}  mi_max AUROC={metrics(y, mi)['AUROC']:.4f}", flush=True)

    weights = np.round(np.arange(0.0, args.w_max + 1e-9, args.w_step), 8)
    w_eegs = np.arange(0.0, args.w_eeg_max + 1e-9, args.w_eeg_step)

    seed_fused, seed_base = {}, {}
    fold_log = []
    t0 = time.time()
    for seed in args.seeds:
        fused = np.full(n, np.nan)
        base_oof_all = np.full(n, np.nan)
        outer = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(np.zeros(n), y), start=1):
            oof_tr, base_te, bcfg = base_oof_and_test(
                X_eeg, y, mi, tr, te, seed + fold * 31,
                args.inner_splits, args.base_oof_splits, w_eegs, args.objective)
            base_local = np.full(n, np.nan)
            base_local[tr] = oof_tr
            base_local[te] = base_te
            base_oof_all[te] = base_te

            best, _ir, _fr = fm.choose_inner_config_and_weight(
                datasets, base_local, y, tr, seed + fold * 100,
                args.inner_splits, args.top_ks, args.models, weights, args.objective)
            data = datasets[best["family"]]
            tr_sig, te_sig, _sel = fm.train_signal_outer(
                data, tr, te, best["model"], int(best["top_k"]), seed + fold * 1000)
            _, base_te_u = fm.robust_unit_apply(base_local[tr], base_local[te])
            _, sig_te_u = fm.robust_unit_apply(tr_sig, te_sig)
            w = float(best["w_signal"])
            fused[te] = (1.0 - w) * base_te_u + w * sig_te_u
            fold_log.append({"seed": seed, "fold": fold, **{f"base_{k}": v for k, v in bcfg.items()},
                             "signal_family": best["family"], "signal_model": best["model"],
                             "signal_top_k": int(best["top_k"]), "w_signal": w})
        seed_fused[seed] = fused
        seed_base[seed] = base_oof_all
        print(f"  seed {seed}: fused AUROC={metrics(y,fused)['AUROC']:.4f} "
              f"base AUROC={metrics(y,base_oof_all)['AUROC']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    fused_bag = np.mean(list(seed_fused.values()), axis=0)
    base_bag = np.mean(list(seed_base.values()), axis=0)
    mf, mb, mm = metrics(y, fused_bag), metrics(y, base_bag), metrics(y, mi)

    print("\n=== CURRENT NESTED-FUSION RESULT ===", flush=True)
    print(f"  fixed raw/2D mi_max score  : AUROC={mm['AUROC']:.4f} AUPRC={mm['AUPRC']:.4f}", flush=True)
    print(f"  nested EEG base            : AUROC={mb['AUROC']:.4f} AUPRC={mb['AUPRC']:.4f} CM={mb['TN']}/{mb['FP']}/{mb['FN']}/{mb['TP']}", flush=True)
    print(f"  nested base + signal       : AUROC={mf['AUROC']:.4f} AUPRC={mf['AUPRC']:.4f} CM={mf['TN']}/{mf['FP']}/{mf['FN']}/{mf['TP']}", flush=True)

    with (out / "airtight_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["subject_id", "true_label", "fused_score", "base_score", "mi_max_score"])
        for s_, yy, fs, bs, ms in zip(subjects, y, fused_bag, base_bag, mi):
            wtr.writerow([s_, int(yy), float(fs), float(bs), float(ms)])

    # per-seed predictions + metrics (needed for the seed-stability figure)
    with (out / "seed_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["seed", "subject_id", "true_label", "fused_score", "base_score"])
        for s in args.seeds:
            for s_, yy, fs, bs in zip(subjects, y, seed_fused[s], seed_base[s]):
                wtr.writerow([s, s_, int(yy), float(fs), float(bs)])
    seed_rows = []
    for s in args.seeds:
        mfs, mbs = metrics(y, seed_fused[s]), metrics(y, seed_base[s])
        seed_rows.append({"seed": s,
                          "fused_AUROC": mfs["AUROC"], "fused_AUPRC": mfs["AUPRC"],
                          "base_AUROC": mbs["AUROC"], "base_AUPRC": mbs["AUPRC"]})
    with (out / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = _csv.DictWriter(f, fieldnames=list(seed_rows[0])); wtr.writeheader(); wtr.writerows(seed_rows)
    with (out / "fold_selections.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = _csv.DictWriter(f, fieldnames=list(fold_log[0]))
        wtr.writeheader(); wtr.writerows(fold_log)
    json.dump({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "claim_boundary": ("EEG features, EEG (C, top_k, w_eeg), signal family, signal model, "
                           "signal top_k, and base/signal fusion weight are selected inside each "
                           "fusion outer-training fold. The mi_max input is a fixed, separately "
                           "cross-fitted raw/2D uncertain-band score and is not regenerated on those "
                           "outer folds. No performance gate."),
        "seeds": args.seeds, "n_splits": args.n_splits, "inner_splits": args.inner_splits,
        "base_oof_splits": args.base_oof_splits,
        "base_grid": {"C": CS, "top_k": TOPKS, "w_eeg": [float(v) for v in w_eegs]},
        "signal_grid": {"top_k": args.top_ks, "models": args.models, "w_signal": [float(v) for v in weights]},
        "fixed_raw_2d_score": mm,
        "nested_eeg_base": mb,
        "nested_base_plus_signal_candidate": mf,
        "eeg_audit": eeg_audit,
    }, (out / "manifest.json").open("w"), indent=2, default=float)
    print(f"\n[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
