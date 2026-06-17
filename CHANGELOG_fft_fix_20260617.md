# FFT 頻率軸修正（2026-06-17）

## 問題
`scripts/nsc_eeg_csv_fusion_ablation.py::extract_one_eeg_csv` 計算頻帶特徵時，
以「全解析度 sample_rate」配「已 downsample_even 到 8192 點的陣列」呼叫 np.fft.rfftfreq，
導致樣本數 > 8192 的 trial 頻率軸被錯置。

## 影響範圍
- 6075 trials 中僅 4 個 > 8192（subj 45, 112, 164, 165；皆 class 0）。
- 其中只有 subj 45 在 analyzable 114 內（112/164/165 已被排除，不進 headline）。
- 故 headline 僅受 subj 45 之單一 trial 影響。

## 修正
改用 decimate 後的等效取樣率 eff_rate = (sampled.size - 1) / duration。

## A/B 驗證（同 venv sklearn 1.8.0，baseline 完全重現 0.7971464）
| 指標 | 修正前 | 修正後 |
|---|---|---|
| AUROC | 0.7971464 | 0.7962159 (Δ -0.0009) |
| AUPRC | 0.7988639 | 0.7970294 (Δ -0.0018) |
| CM (TN/FP/FN/TP) | 34/28/10/42 | 33/29/10/42（subj45 由 TN→FP）|

結論：變動極小（ΔAUROC ≈ -0.0009，遠在 95% CI 0.71–0.87 之內），
不改變任何實質結論；但**正確的 headline 應為 AUROC 0.7962**（bug-free）。
團隊可選擇 (a) 以 0.7962 為定稿 headline 並引用本修正，或 (b) 重新 freeze 完整 package。
原始檔備份：/tmp 之 ablation_ORIGINAL.py.bak（開發機）。
