#!/usr/bin/env python3
"""Truly nested pipeline: the 2D image branch is rebuilt INSIDE every outer training fold.

Why this exists
---------------
The previous run (`nsc114_airtight_fully_nested_20260710.py`) read the image score from
`analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv`, column
`mi_max`.  Two things are wrong with that column, both verified against the source:

1.  It is NOT a pure image score.  `nsc_uncertain_band_patch_refinement.py` computes
        fused_test = apply_gate(raw_test, branch_test["mi_max"], best_gate)
    where `raw_test` is a raw physiological time-series branch (DTW + ExtraTrees) and the gate
    blends the two only inside an uncertainty band.  All ten folds selected mode
    `uncertain_band`, so for every participant whose raw score fell outside [low, high] the
    stored "image" score IS the raw score.  That is 75 of 114 participants (65.8%).

2.  Its out-of-fold partition (StratifiedGroupKFold, random_state=42) does not coincide with the
    fusion outer folds (StratifiedKFold, random_state=seed).  0 of the 100 outer folds match, and
    the model that produced a given participant's image score was, on average, trained on 9.42
    other participants who sit in that participant's own outer-test fold.

This script removes both problems:

  * The image branch is a pure 2D multi-instance ExtraTrees model with max pooling
    (`fit_2d_multi_instance(..., "ExtraTrees", "max")`), no raw branch, no gate.
  * For each outer fold it is fitted only on outer-training participants: an inner OOF pass gives
    the training-fold image scores that the base branch needs, and one fit on the whole outer
    training fold scores the held-out participants.

Everything else (base EEG sub-branch, image/EEG weight, signal family/model/top-k, fusion weight)
keeps the selection scheme of the previous run.  No performance gate.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "repro_packages/NSC114_EEG_repro_package_20260604/scripts"))

import nsc114_fully_nested_signal_interaction_fusion_20260709 as fm  # noqa: E402
import nsc_restricted_subject_bagging_tail_stats as bm  # noqa: E402
from nsc_eeg_csv_fusion_ablation import metrics, robust_unit_fit  # noqa: E402
from nsc_multi_instance_2d_raw_fusion import (  # noqa: E402
    collect_image_instances,
    fit_2d_multi_instance,
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
IMG_MODEL, IMG_AGG, IMG_TOPK, IMG_GRID = "ExtraTrees", "top3mean", 174, 8


def _obj(met, objective):
    return min(met["AUROC"], met["AUPRC"]) if objective == "min_metric" else met["AUPRC"]


# --------------------------------------------------------------------- image branch
def image_oof_and_test(Xi, yi, isub, subjects, y, tr, te, seed, splits):
    """Pure 2D image score. `mi_tr` is OOF within `tr`; `mi_te` is fitted on all of `tr`."""
    mi_tr = np.zeros(len(tr), dtype=float)
    sub = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    for f, (a, b) in enumerate(sub.split(np.zeros(len(tr)), y[tr]), start=1):
        _, sc, _ = fit_2d_multi_instance(Xi, yi, isub, subjects, y, tr[a], tr[b],
                                         seed + f * 17, IMG_TOPK, IMG_MODEL, IMG_AGG)
        mi_tr[b] = sc
    _, mi_te, _ = fit_2d_multi_instance(Xi, yi, isub, subjects, y, tr, te,
                                        seed + 991, IMG_TOPK, IMG_MODEL, IMG_AGG)
    return mi_tr, mi_te


# --------------------------------------------------------------------- base branch
def select_base_cfg(X, y, mi, seed, inner_splits, w_eegs, objective):
    """Choose (C, top_k, w_eeg) using ONLY the rows given (a subset of the outer-training fold)."""
    base_scaled, _ = robust_unit_fit(mi, mi)
    splits = list(StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                  random_state=seed).split(X, y))
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


def base_oof_and_test(X, y, mi_tr, mi_te, tr, te, seed, inner_splits, base_oof_splits,
                      w_eegs, objective):
    """mi_tr / mi_te are fold-local image scores; nothing here sees an outer-test participant."""
    Xtr, ytr = X[tr], y[tr]

    oof = np.zeros(len(tr), dtype=float)
    sub = StratifiedKFold(n_splits=base_oof_splits, shuffle=True, random_state=seed)
    for f, (a, b) in enumerate(sub.split(Xtr, ytr), start=1):
        C, k, w = select_base_cfg(Xtr[a], ytr[a], mi_tr[a], seed + f * 7,
                                  inner_splits, w_eegs, objective)
        imp, sel = bm.select_topk(Xtr[a], ytr[a], k)
        clf = bm.make_model(PENALTY, C, seed + f * 11)
        Xa = imp.transform(Xtr[a])[:, sel]
        clf.fit(Xa, ytr[a])
        e_a = clf.predict_proba(Xa)[:, 1]
        e_b = clf.predict_proba(imp.transform(Xtr[b])[:, sel])[:, 1]
        _, mi_b = robust_unit_fit(mi_tr[a], mi_tr[b])
        _, e_bu = robust_unit_fit(e_a, e_b)
        oof[b] = (1.0 - w) * mi_b + w * e_bu

    C, k, w = select_base_cfg(Xtr, ytr, mi_tr, seed + 9973, inner_splits, w_eegs, objective)
    imp, sel = bm.select_topk(Xtr, ytr, k)
    clf = bm.make_model(PENALTY, C, seed + 555)
    Xa = imp.transform(Xtr)[:, sel]
    clf.fit(Xa, ytr)
    e_tr = clf.predict_proba(Xa)[:, 1]
    e_te = clf.predict_proba(imp.transform(X[te])[:, sel])[:, 1]
    _, mi_te_u = robust_unit_fit(mi_tr, mi_te)
    _, e_teu = robust_unit_fit(e_tr, e_te)
    base_te = (1.0 - w) * mi_te_u + w * e_teu
    return oof, base_te, {"C": C, "top_k": k, "w_eeg": w}


def main() -> int:
    global IMG_MODEL, IMG_AGG, IMG_TOPK, IMG_GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "nsc_dataset_images/manifest.csv")
    ap.add_argument("--physio-root", type=Path, default=ROOT / "bio-addict/csv-data")
    ap.add_argument("--eeg-root", type=Path, default=ROOT / "eeg-csv-data-by-class")
    ap.add_argument("--images-root", type=Path, default=ROOT / "nsc_dataset_images")
    ap.add_argument("--eeg-cache", type=Path,
                    default=ROOT / "analysis/nsc_eeg_csv_fusion_ablation_20260520/eeg_feature_cache")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis/nsc114_true_nested_20260711")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20260709, 20260719)))
    ap.add_argument("--top-ks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    ap.add_argument("--models", nargs="+", default=["logreg", "prototype"])
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--base-oof-splits", type=int, default=10)
    ap.add_argument("--image-oof-splits", type=int, default=5)
    ap.add_argument("--image-model", choices=["ExtraTrees", "RandomForest", "HistGB"],
                    default=IMG_MODEL)
    ap.add_argument("--image-aggregation",
                    choices=["mean", "max", "p75", "top3mean", "mean_max"],
                    default=IMG_AGG)
    ap.add_argument("--image-top-k", type=int, default=IMG_TOPK)
    ap.add_argument("--image-grid", type=int, default=IMG_GRID)
    ap.add_argument("--w-max", type=float, default=0.35)
    ap.add_argument("--w-step", type=float, default=0.05)
    ap.add_argument("--w-eeg-max", type=float, default=0.25)
    ap.add_argument("--w-eeg-step", type=float, default=0.05)
    ap.add_argument("--objective", choices=["min_metric", "auprc_then_auroc"], default="min_metric")
    args = ap.parse_args()

    IMG_MODEL = args.image_model
    IMG_AGG = args.image_aggregation
    IMG_TOPK = args.image_top_k
    IMG_GRID = args.image_grid

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.manifest.resolve())
    subjects = sorted(labels, key=natural_key)
    y = np.asarray([labels[s] for s in subjects], dtype=int)
    n = len(y)
    print(f"[data] subjects={n} class0={(y==0).sum()} class1={(y==1).sum()}", flush=True)

    t0 = time.time()
    inst, names = collect_image_instances(args.images_root.resolve(), IMG_GRID)
    Xi = np.asarray([[d["features"].get(nm, np.nan) for nm in names] for d in inst], dtype=float)
    yi = np.asarray([d["label"] for d in inst], dtype=int)
    isub = np.asarray([str(d["subject_id"]) for d in inst])
    print(f"[data] image instances={Xi.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    physio_rows, _ = collect_physio_features(set(subjects), args.physio_root.resolve())
    eeg_rows, _ = collect_eeg_features(set(subjects), args.eeg_root.resolve())
    rows = merge_rows(physio_rows, eeg_rows)
    datasets = {fam: matrix(subjects, y, rows, pre) for fam, pre in FAMILIES.items()}
    X_eeg, _fn, _audit = bm.build_subject_eeg_features_tail_stats(
        subjects, args.eeg_root.resolve(), args.eeg_cache.resolve(),
        include_counts=False, class_aware_features=False)
    print(f"[data] X_eeg={X_eeg.shape}, signal families built ({time.time()-t0:.0f}s)", flush=True)

    weights = np.round(np.arange(0.0, args.w_max + 1e-9, args.w_step), 8)
    w_eegs = np.arange(0.0, args.w_eeg_max + 1e-9, args.w_eeg_step)

    seed_fused, seed_base, seed_img = {}, {}, {}
    fold_log = []
    t0 = time.time()
    for seed in args.seeds:
        fused = np.full(n, np.nan)
        base_all = np.full(n, np.nan)
        img_all = np.full(n, np.nan)
        outer = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(np.zeros(n), y), start=1):
            mi_tr, mi_te = image_oof_and_test(Xi, yi, isub, subjects, y, tr, te,
                                              seed + fold * 13, args.image_oof_splits)
            img_all[te] = mi_te

            oof_tr, base_te, bcfg = base_oof_and_test(
                X_eeg, y, mi_tr, mi_te, tr, te, seed + fold * 31,
                args.inner_splits, args.base_oof_splits, w_eegs, args.objective)
            base_local = np.full(n, np.nan)
            base_local[tr] = oof_tr
            base_local[te] = base_te
            base_all[te] = base_te

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
            fold_log.append({"seed": seed, "fold": fold,
                             **{f"base_{k}": v for k, v in bcfg.items()},
                             "signal_family": best["family"], "signal_model": best["model"],
                             "signal_top_k": int(best["top_k"]), "w_signal": w})
        seed_fused[seed], seed_base[seed], seed_img[seed] = fused, base_all, img_all
        print(f"  seed {seed}: fused={metrics(y,fused)['AUROC']:.4f} "
              f"base={metrics(y,base_all)['AUROC']:.4f} image={metrics(y,img_all)['AUROC']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    fused_bag = np.mean(list(seed_fused.values()), axis=0)
    base_bag = np.mean(list(seed_base.values()), axis=0)
    img_bag = np.mean(list(seed_img.values()), axis=0)
    mf, mb, mm = metrics(y, fused_bag), metrics(y, base_bag), metrics(y, img_bag)

    print("\n=== TRUE NESTED RESULT (pure image branch, fold-aligned) ===", flush=True)
    print(f"  image branch (2D {IMG_AGG}): AUROC={mm['AUROC']:.4f} AUPRC={mm['AUPRC']:.4f}", flush=True)
    print(f"  nested base               : AUROC={mb['AUROC']:.4f} AUPRC={mb['AUPRC']:.4f} "
          f"CM={mb['TN']}/{mb['FP']}/{mb['FN']}/{mb['TP']}", flush=True)
    print(f"  nested fused              : AUROC={mf['AUROC']:.4f} AUPRC={mf['AUPRC']:.4f} "
          f"CM={mf['TN']}/{mf['FP']}/{mf['FN']}/{mf['TP']}", flush=True)

    with (out / "true_nested_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["subject_id", "true_label", "fused_score", "base_score", "image_score"])
        for s_, yy, fs, bs, ms in zip(subjects, y, fused_bag, base_bag, img_bag):
            wtr.writerow([s_, int(yy), f"{fs:.10f}", f"{bs:.10f}", f"{ms:.10f}"])

    with (out / "seed_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["seed", "subject_id", "true_label", "fused_score", "base_score", "image_score"])
        for s in args.seeds:
            for s_, yy, fs, bs, ms in zip(subjects, y, seed_fused[s], seed_base[s], seed_img[s]):
                wtr.writerow([s, s_, int(yy), f"{fs:.10f}", f"{bs:.10f}", f"{ms:.10f}"])

    with (out / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, ["seed", "fused_AUROC", "fused_AUPRC", "base_AUROC", "base_AUPRC"])
        wtr.writeheader()
        for s in args.seeds:
            a, b = metrics(y, seed_fused[s]), metrics(y, seed_base[s])
            wtr.writerow({"seed": s, "fused_AUROC": a["AUROC"], "fused_AUPRC": a["AUPRC"],
                          "base_AUROC": b["AUROC"], "base_AUPRC": b["AUPRC"]})

    with (out / "fold_selections.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, list(fold_log[0]))
        wtr.writeheader()
        wtr.writerows(fold_log)

    json.dump({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "claim_boundary": ("FULLY nested: the 2D image branch, the base EEG sub-branch, "
                           "(C, top_k, w_eeg), the signal family/model/top_k and the base/signal "
                           "fusion weight are ALL rebuilt or selected inside the outer-training "
                           "fold. No raw time-series branch, no uncertainty-band gate, no "
                           "performance gate."),
        "image_branch": {"model": IMG_MODEL, "aggregation": IMG_AGG, "top_k": IMG_TOPK,
                         "patch_grid": IMG_GRID, "instances": int(Xi.shape[0])},
        "seeds": args.seeds, "n_splits": args.n_splits, "inner_splits": args.inner_splits,
        "base_oof_splits": args.base_oof_splits, "image_oof_splits": args.image_oof_splits,
        "base_grid": {"C": CS, "top_k": TOPKS, "w_eeg": [float(v) for v in w_eegs]},
        "signal_grid": {"top_k": args.top_ks, "models": args.models,
                        "w_signal": [float(v) for v in weights]},
        "image_branch_metrics": mm, "nested_base": mb, "nested_fused": mf,
    }, (out / "manifest.json").open("w"), indent=2, default=float)

    print(f"\n[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
