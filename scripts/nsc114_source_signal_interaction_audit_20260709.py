#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, re, sys, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'Features .* are constant')

REPO_ROOT = Path(__file__).resolve().parents[1]

SIGNAL_STATS = [
    'mean','std','min','max','range','median','iqr','mad','rms','abs_mean','center_abs_mean',
    'skew_proxy','kurtosis_proxy','diff_abs_mean','diff_std','diff_p95_abs','zero_cross_rate_centered',
    'trend_slope_sampled','fft_band1_frac','fft_band2_frac','fft_band3_frac','fft_band4_frac','fft_peak_frac',
    'sampled_turning_rate','last_minus_first_seg_mean','last_over_first_seg_std',
    'p01','p05','p10','p25','p50','p75','p90','p95','p99'
]

@dataclass
class Dataset:
    subjects: list[str]
    y: np.ndarray
    feature_names: list[str]
    X: np.ndarray


def natural_key(text):
    parts = re.split(r'(\d+)', str(text))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r}) if rows else []
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def safe_percentiles(values, qs):
    if values.size == 0:
        return {f'p{int(q):02d}': math.nan for q in qs}
    vals = np.nanpercentile(values, qs)
    return {f'p{int(q):02d}': float(v) for q, v in zip(qs, vals)}


def downsample_even(values, max_points=32768):
    values = values[np.isfinite(values)]
    if values.size <= max_points:
        return values.astype(float)
    idx = np.linspace(0, values.size - 1, max_points).astype(int)
    return values[idx].astype(float)


def vector_stats(values, fs=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'n': 0, 'finite_ratio': 0.0}
    full_n = int(values.size)
    duration_s = float(full_n / fs) if fs and fs > 0 else math.nan
    values = downsample_even(values)
    q = safe_percentiles(values, [1,5,10,25,50,75,90,95,99])
    med = q['p50']
    centered = values - med
    diffs = np.diff(values) if values.size > 1 else np.asarray([], dtype=float)
    abs_diffs = np.abs(diffs)
    feats = {
        'n': full_n, 'duration_s': duration_s, 'sample_rate_hz': float(fs) if fs and fs > 0 else math.nan,
        'finite_ratio': 1.0,
        'mean': float(np.nanmean(values)), 'std': float(np.nanstd(values)),
        'min': float(np.nanmin(values)), 'max': float(np.nanmax(values)),
        'range': float(np.nanmax(values)-np.nanmin(values)), 'median': float(med),
        'iqr': float(q['p75']-q['p25']), 'mad': float(np.nanmedian(np.abs(centered))),
        'rms': float(np.sqrt(np.nanmean(values**2))), 'abs_mean': float(np.nanmean(np.abs(values))),
        'center_abs_mean': float(np.nanmean(np.abs(centered))),
        'skew_proxy': float(np.nanmean(centered**3)/(np.nanstd(values)**3 + 1e-12)),
        'kurtosis_proxy': float(np.nanmean(centered**4)/(np.nanstd(values)**4 + 1e-12)),
        'diff_abs_mean': float(np.nanmean(abs_diffs)) if abs_diffs.size else 0.0,
        'diff_std': float(np.nanstd(diffs)) if diffs.size else 0.0,
        'diff_p95_abs': float(np.nanpercentile(abs_diffs, 95)) if abs_diffs.size else 0.0,
        'zero_cross_rate_centered': float(np.mean(np.diff(np.signbit(centered)) != 0)) if centered.size > 1 else 0.0,
    }
    feats.update(q)
    if values.size >= 16:
        x = np.linspace(0, 1, values.size)
        feats['trend_slope_sampled'] = float(np.polyfit(x, values, deg=1)[0])
        sampled_centered = values - float(np.nanmean(values))
        power = np.abs(np.fft.rfft(sampled_centered))**2
        total = float(np.sum(power) + 1e-12)
        for i, band in enumerate(np.array_split(power, 4), start=1):
            feats[f'fft_band{i}_frac'] = float(np.sum(band)/total)
        feats['fft_peak_frac'] = float(np.max(power)/total)
        feats['sampled_turning_rate'] = float(np.mean(np.diff(np.sign(np.diff(values))) != 0)) if values.size > 2 else 0.0
    else:
        for name in ['trend_slope_sampled','fft_band1_frac','fft_band2_frac','fft_band3_frac','fft_band4_frac','fft_peak_frac','sampled_turning_rate']:
            feats[name] = math.nan
    segs = np.array_split(values, 5)
    for i, seg in enumerate(segs, start=1):
        if seg.size:
            feats[f'seg{i}_mean'] = float(np.nanmean(seg))
            feats[f'seg{i}_std'] = float(np.nanstd(seg))
            feats[f'seg{i}_median'] = float(np.nanmedian(seg))
            feats[f'seg{i}_iqr'] = float(np.nanpercentile(seg,75)-np.nanpercentile(seg,25))
    if segs[0].size and segs[-1].size:
        feats['last_minus_first_seg_mean'] = float(np.nanmean(segs[-1])-np.nanmean(segs[0]))
        feats['last_over_first_seg_std'] = float((np.nanstd(segs[-1])+1e-12)/(np.nanstd(segs[0])+1e-12))
    return feats


def read_signal_csv(path: Path):
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    value_candidates = [c for c in cols if c not in {'time_sec','index','sample_index'}]
    val_col = value_candidates[-1] if value_candidates else cols[-1]
    usecols = []
    if 'time_sec' in cols:
        usecols.append('time_sec')
    usecols.append(val_col)
    df = pd.read_csv(path, usecols=usecols)
    vals = df[val_col].to_numpy(dtype=float)
    fs = None
    if 'time_sec' in df.columns:
        t = df['time_sec'].to_numpy(dtype=float)
        finite = np.isfinite(vals) & np.isfinite(t)
        vals, t = vals[finite], t[finite]
        if t.size > 1 and t[-1] > t[0]:
            fs = float((t.size-1)/(t[-1]-t[0]))
    else:
        vals = vals[np.isfinite(vals)]
    return vals, fs


def load_labels(manifest_path: Path):
    df = pd.read_csv(manifest_path)
    lab = df[['subject_id','label']].drop_duplicates().copy()
    lab['subject_id'] = lab['subject_id'].astype(str)
    return dict(zip(lab['subject_id'], lab['label'].astype(int)))


def collect_physio_features(subjects, physio_root: Path):
    pat = re.compile(r'^(?P<sid>\d+)_Sess(?P<sess>\d+)_(?P<sig>.+)\.csv$')
    grouped = defaultdict(lambda: defaultdict(list))
    fs_grouped = defaultdict(lambda: defaultdict(list))
    audit_files = Counter()
    for path in sorted(physio_root.glob('*.csv'), key=lambda p: natural_key(p.name)):
        m = pat.match(path.name)
        if not m: continue
        sid, sig = m.group('sid'), m.group('sig')
        if sid not in subjects: continue
        vals, fs = read_signal_csv(path)
        grouped[sid][sig].append(vals)
        if fs: fs_grouped[sid][sig].append(fs)
        audit_files[sig] += 1
    rows = {}
    source_stats = {}
    for sid in subjects:
        feats = {}
        stats_by_sig = {}
        for sig, parts in grouped.get(sid, {}).items():
            vals = np.concatenate([np.asarray(x, dtype=float) for x in parts if len(x)]) if parts else np.asarray([])
            fs_vals = fs_grouped.get(sid, {}).get(sig, [])
            fs = float(np.nanmedian(fs_vals)) if fs_vals else None
            stats = vector_stats(vals, fs=fs)
            stats_by_sig[sig] = stats
            for name, val in stats.items():
                if isinstance(val, (int,float)) and np.isfinite(val):
                    feats[f'SIG/{sig}/{name}'] = float(val)
        for stat in SIGNAL_STATS:
            available = [(sig, st[stat]) for sig, st in stats_by_sig.items() if stat in st and np.isfinite(st.get(stat, math.nan))]
            available = sorted(available, key=lambda x: x[0])
            for i, (a, va) in enumerate(available):
                for b, vb in available[i+1:]:
                    feats[f'SIG_RATIO/{a}_over_{b}/{stat}'] = float((va + 1e-12)/(vb + 1e-12))
                    feats[f'SIG_DIFF/{a}_minus_{b}/{stat}'] = float(va-vb)
        rows[sid] = feats
        source_stats[sid] = sorted(stats_by_sig)
    return rows, {'files_by_signal': dict(audit_files), 'signals_by_subject': {k:v for k,v in source_stats.items()}}


def collect_eeg_features(subjects, eeg_root: Path):
    # Intentionally label-agnostic aggregation: all trials for each subject are pooled.
    index = defaultdict(list)
    for cls_dir in ['0','1','2','3','4']:
        for p in sorted((eeg_root/cls_dir).glob('*.csv'), key=lambda x: natural_key(x.name)):
            m = re.match(r'^(\d+)_', p.name)
            if m and m.group(1) in subjects:
                index[m.group(1)].append(p)
    rows = {}
    audit = {'files_by_subject': {}, 'subjects_with_eeg': 0, 'subjects_without_eeg': 0}
    for sid in subjects:
        trial_stats = []
        for p in index.get(sid, []):
            vals, fs = read_signal_csv(p)
            trial_stats.append(vector_stats(vals, fs=fs))
        feats = {}
        numeric = sorted({k for st in trial_stats for k,v in st.items() if isinstance(v,(int,float))})
        for name in numeric:
            vals = np.asarray([float(st.get(name, math.nan)) for st in trial_stats], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                feats[f'EEG_ALL/{name}/mean'] = float(np.nanmean(vals))
                feats[f'EEG_ALL/{name}/std'] = float(np.nanstd(vals))
                feats[f'EEG_ALL/{name}/p25'] = float(np.nanpercentile(vals,25))
                feats[f'EEG_ALL/{name}/p50'] = float(np.nanpercentile(vals,50))
                feats[f'EEG_ALL/{name}/p75'] = float(np.nanpercentile(vals,75))
        feats['EEG_ALL/trial_count'] = float(len(trial_stats))
        feats['EEG_ALL/has_any'] = float(len(trial_stats)>0)
        rows[sid] = feats
        audit['files_by_subject'][sid] = len(trial_stats)
        if trial_stats: audit['subjects_with_eeg'] += 1
        else: audit['subjects_without_eeg'] += 1
    return rows, audit


def merge_rows(*feature_rows):
    out = defaultdict(dict)
    for rows in feature_rows:
        for sid, feats in rows.items():
            out[sid].update(feats)
    return dict(out)


def matrix(subjects, y, rows, prefixes=None):
    filtered = {}
    for sid in subjects:
        feats = rows.get(sid, {})
        if prefixes:
            feats = {k:v for k,v in feats.items() if k.startswith(prefixes)}
        filtered[sid] = feats
    names = sorted({k for sid in subjects for k in filtered[sid]})
    X = np.full((len(subjects), len(names)), np.nan)
    idx = {n:i for i,n in enumerate(names)}
    for r,sid in enumerate(subjects):
        for k,v in filtered[sid].items():
            if np.isfinite(v): X[r, idx[k]] = float(v)
    return Dataset(subjects, y, names, X)


def metrics(y, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, pred, labels=[0,1]).ravel()]
    sens = tp/(tp+fn) if tp+fn else math.nan
    spec = tn/(tn+fp) if tn+fp else math.nan
    return {
        'AUROC': float(roc_auc_score(y, score)), 'AUPRC': float(average_precision_score(y, score)),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'accuracy': float((tn+tp)/len(y)), 'sensitivity': float(sens), 'specificity': float(spec),
        'balanced_accuracy': float(np.nanmean([sens, spec])),
        'PPV': float(tp/(tp+fp)) if tp+fp else math.nan,
        'NPV': float(tn/(tn+fn)) if tn+fn else math.nan,
    }


def select_fold(Xtr, ytr, Xte, names, k):
    imp = SimpleImputer(strategy='median')
    Xtr_i = imp.fit_transform(Xtr)
    Xte_i = imp.transform(Xte)
    if Xtr_i.shape[1] == 0: raise ValueError('no features')
    if k <= 0 or k >= Xtr_i.shape[1]:
        sel = np.arange(Xtr_i.shape[1])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            score, _ = f_classif(Xtr_i, ytr)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        sel = np.argsort(score)[::-1][:k]
    return Xtr_i[:, sel], Xte_i[:, sel], [names[i] for i in sel]


def fit_score(model, Xtr, ytr, Xte, seed):
    if model == 'logreg':
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.25, class_weight='balanced', solver='liblinear'))
    elif model == 'prototype':
        sc = StandardScaler(); Ztr = sc.fit_transform(Xtr); Zte = sc.transform(Xte)
        c0 = Ztr[ytr==0].mean(axis=0); c1 = Ztr[ytr==1].mean(axis=0)
        raw = np.linalg.norm(Zte-c0, axis=1) - np.linalg.norm(Zte-c1, axis=1)
        return 1/(1+np.exp(-raw))
    elif model == 'extratrees':
        m = ExtraTreesClassifier(n_estimators=500, random_state=seed, class_weight='balanced', max_features='sqrt', min_samples_leaf=1, n_jobs=1)
    elif model == 'rf':
        m = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight='balanced', max_features='sqrt', min_samples_leaf=1, n_jobs=1)
    else:
        raise ValueError(model)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:,1]


def evaluate(data, family, model, top_k, seed, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score = np.full(len(data.y), np.nan)
    folds = np.full(len(data.y), -1)
    feat_counts = Counter()
    for fold, (tr, te) in enumerate(skf.split(data.X, data.y), start=1):
        Xtr, Xte, selected = select_fold(data.X[tr], data.y[tr], data.X[te], data.feature_names, top_k)
        score[te] = fit_score(model, Xtr, data.y[tr], Xte, seed+fold)
        folds[te] = fold
        feat_counts.update(selected)
    row = metrics(data.y, score)
    row.update({'family': family, 'model': model, 'top_k': top_k, 'seed': seed, 'feature_count': len(data.feature_names), 'case_count': len(data.y)})
    preds = [{'subject_id': s, 'true_label': int(data.y[i]), 'score': float(score[i]), 'pred': int(score[i]>=0.5), 'fold': int(folds[i]), 'family': family, 'model': model, 'top_k': top_k, 'seed': seed} for i,s in enumerate(data.subjects)]
    feats = [{'family': family, 'model': model, 'top_k': top_k, 'seed': seed, 'feature': f, 'selected_count': int(c)} for f,c in feat_counts.most_common(80)]
    return row, preds, feats


def load_baseline_predictions(path: Path, subjects):
    if not path.exists(): return {}, []
    df = pd.read_csv(path)
    df['subject_id'] = df['subject_id'].astype(str)
    rows = []
    scores = {}
    for method, g in df.groupby('method'):
        gg = g.drop_duplicates('subject_id').set_index('subject_id')
        if all(s in gg.index for s in subjects):
            scores[f'BASE/{method}'] = gg.loc[subjects, 'score'].to_numpy(float)
            met = metrics(gg.loc[subjects, 'true_label'].to_numpy(int), gg.loc[subjects, 'score'].to_numpy(float))
            met.update({'family': 'baseline_oof', 'model': '', 'top_k': '', 'seed': '', 'feature_count': 1, 'case_count': len(subjects), 'method': method})
            rows.append(met)
    return scores, rows


def append_baseline_features(rows, baseline_scores, subjects):
    out = {sid: dict(rows.get(sid, {})) for sid in subjects}
    for name, arr in baseline_scores.items():
        for sid, val in zip(subjects, arr):
            out[sid][name] = float(val)
    return out


def aggregate(rows):
    grp = defaultdict(list)
    for r in rows:
        if 'error' not in r or not r.get('error'):
            grp[(r['family'], r['model'], r['top_k'])].append(r)
    out=[]
    for (fam, model, k), vals in grp.items():
        a={'family':fam,'model':model,'top_k':k,'seed_count':len(vals)}
        for m in ['AUROC','AUPRC','balanced_accuracy','sensitivity','specificity','accuracy']:
            x=np.asarray([float(v[m]) for v in vals],float)
            a[f'{m}_mean']=float(np.nanmean(x)); a[f'{m}_std']=float(np.nanstd(x)); a[f'{m}_min']=float(np.nanmin(x)); a[f'{m}_max']=float(np.nanmax(x))
        out.append(a)
    out.sort(key=lambda r:(r['AUPRC_mean'], r['AUROC_mean'], r['balanced_accuracy_mean']), reverse=True)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=REPO_ROOT / 'data/nsc_dataset_images/manifest.csv')
    ap.add_argument('--physio-root', type=Path, default=REPO_ROOT / 'data/physiology-csv')
    ap.add_argument('--eeg-root', type=Path, default=REPO_ROOT / 'data/eeg-csv-data-by-class')
    ap.add_argument('--baseline-predictions', type=Path, default=REPO_ROOT / 'analysis/nsc_eeg_csv_fusion_ablation_noleak_lr_refine_20260520/eeg_fusion_predictions.csv')
    ap.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'analysis/nsc114_source_signal_interaction_audit_20260709')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(20260709,20260719)))
    ap.add_argument('--top-ks', type=int, nargs='+', default=[8,16,24,32,48,64,96,128,192,256])
    ap.add_argument('--models', nargs='+', default=['logreg','prototype','extratrees'])
    ap.add_argument('--n-splits', type=int, default=10)
    args=ap.parse_args()
    out=args.out_dir; out.mkdir(parents=True, exist_ok=True)
    labels=load_labels(args.manifest)
    subjects=sorted(labels, key=natural_key)
    y=np.asarray([labels[s] for s in subjects], int)
    phys, phys_audit = collect_physio_features(set(subjects), args.physio_root)
    eeg, eeg_audit = collect_eeg_features(set(subjects), args.eeg_root)
    merged = merge_rows(phys, eeg)
    baseline_scores, baseline_rows = load_baseline_predictions(args.baseline_predictions, subjects)
    merged_with_base = append_baseline_features(merged, baseline_scores, subjects)
    families = {
        'physio_main': ('SIG/',),
        'physio_interactions': ('SIG_RATIO/','SIG_DIFF/'),
        'eeg_all_stats': ('EEG_ALL/',),
        'signal_main_plus_eeg': ('SIG/','EEG_ALL/'),
        'signal_all': ('SIG/','SIG_RATIO/','SIG_DIFF/','EEG_ALL/'),
        'baseline_plus_signal_all': ('BASE/','SIG/','SIG_RATIO/','SIG_DIFF/','EEG_ALL/'),
    }
    dataset_audit=[]; rows=[]; preds=[]; feat_rows=[]
    # include baselines as rows, not as candidate aggregate.
    for br in baseline_rows:
        rows.append({**br, 'is_baseline': 1})
    for fam, prefixes in families.items():
        data=matrix(subjects, y, merged_with_base if fam.startswith('baseline') else merged, prefixes)
        dataset_audit.append({'family':fam,'case_count':len(data.subjects),'positive_count':int(y.sum()),'negative_count':int(len(y)-y.sum()),'feature_count':len(data.feature_names),'missing_rate':float(np.isnan(data.X).mean()) if data.X.size else math.nan})
        if len(data.feature_names)==0: continue
        for seed in args.seeds:
            for k in args.top_ks:
                if k > len(data.feature_names): continue
                for model in args.models:
                    try:
                        row, pr, fr = evaluate(data, fam, model, k, seed, args.n_splits)
                        row['is_baseline']=0
                        rows.append(row)
                        cid=f'{fam}|{model}|k{k}|seed{seed}'
                        for r in pr: r['candidate_id']=cid
                        preds.extend(pr)
                        feat_rows.extend(fr)
                    except Exception as e:
                        rows.append({'family':fam,'model':model,'top_k':k,'seed':seed,'error':str(e),'is_baseline':0})
    cand_rows=[r for r in rows if not r.get('is_baseline') and not r.get('error')]
    cand_rows.sort(key=lambda r:(float(r['AUPRC']), float(r['AUROC']), float(r['balanced_accuracy'])), reverse=True)
    agg=aggregate(cand_rows)
    write_csv(out/'dataset_audit.csv', dataset_audit)
    write_csv(out/'candidate_metrics_by_seed.csv', rows)
    write_csv(out/'candidate_metrics_aggregate.csv', agg)
    write_csv(out/'candidate_predictions.csv', preds)
    write_csv(out/'selected_features_long.csv', feat_rows)
    feat_rank=Counter()
    for r in feat_rows:
        feat_rank[(r['family'], r['feature'])]+=int(r['selected_count'])
    write_csv(out/'selected_features_ranked.csv', [{'family':fam,'feature':feat,'selection_count':c} for (fam,feat),c in feat_rank.most_common(400)], ['family','feature','selection_count'])
    best=cand_rows[0] if cand_rows else {}
    best_id=f"{best.get('family')}|{best.get('model')}|k{best.get('top_k')}|seed{best.get('seed')}" if best else ''
    best_preds=[r for r in preds if r.get('candidate_id')==best_id]
    if best_preds: write_csv(out/'best_candidate_predictions.csv', best_preds)
    manifest={'created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'out_dir':str(out),'subjects':len(subjects),'cases_by_label':dict(Counter(map(str,y))),'physio_audit':phys_audit,'eeg_audit':eeg_audit,'dataset_audit':dataset_audit,'models':args.models,'top_ks':args.top_ks,'seeds':args.seeds,'n_splits':args.n_splits,'baseline_predictions':str(args.baseline_predictions),'baseline_rows':baseline_rows,'best_single_seed_candidate':best,'best_aggregate_candidate':agg[0] if agg else {},'claim_level':'exploratory source-domain patient-aware 10-fold audit; feature selection/imputation/scaling/modeling fold-local; baseline_plus_signal uses precomputed OOF baseline scores as a stacking feature'}
    dump_json(out/'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:12000])

if __name__=='__main__':
    main()
