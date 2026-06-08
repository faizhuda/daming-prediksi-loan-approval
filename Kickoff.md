# KOM1338 Data Mining — Prediksi Loan Approval

Kaggle competition: https://www.kaggle.com/competitions/kom-1338-data-mining-prediksi-loan-approval/

## Task

Binary classification. Predict `loan_status` (0 atau 1) untuk setiap baris di `test.csv`.
Submission harus berupa **probabilitas** (float 0–1), bukan label biner — dievaluasi dengan **AUC-ROC**.

## Dataset

| File | Shape | Keterangan |
|---|---|---|
| `train.csv` | 43983 × 13 | Fitur + target `loan_status` |
| `test.csv` | 14662 × 12 | Fitur saja, tanpa target |
| `sample_submission.csv` | 14662 × 2 | Format: `sample_id`, `loan_status` |

Tidak ada missing value. Tidak ada file pendukung tambahan.

## Columns

```
sample_id                  — ID baris (string, format TR.../TS...)
person_age                 — usia peminjam (int)
person_income              — pendapatan tahunan (int)
person_home_ownership      — RENT | MORTGAGE | OWN | OTHER
person_emp_length          — lama bekerja dalam tahun (float)
loan_intent                — PERSONAL | VENTURE | EDUCATION | HOMEIMPROVEMENT | DEBTCONSOLIDATION | MEDICAL
loan_grade                 — A | B | C | D | E | F | G
loan_amnt                  — jumlah pinjaman (int)
loan_int_rate              — suku bunga (float)
loan_percent_income        — rasio pinjaman/pendapatan (float)
cb_person_default_on_file  — Y | N (riwayat default)
cb_person_cred_hist_length — panjang riwayat kredit (int)
loan_status                — TARGET: 0 (tidak default) | 1 (default)
```

## Class Distribution

- Class 0: 37721 (85.8%)
- Class 1: 6262 (14.2%)

Data imbalanced — gunakan `class_weight='balanced'` atau `scale_pos_weight` saat training.

## Pipeline

1. **EDA** — distribusi fitur, korelasi dengan target, outlier
2. **Preprocessing**
   - Label encode: `loan_grade` (ordinal A→G)
   - One-hot encode: `person_home_ownership`, `loan_intent`, `cb_person_default_on_file`
   - Drop: `sample_id`
   - Opsional: `StandardScaler` untuk numerik (tidak wajib untuk tree-based)
3. **Feature engineering (opsional)**
   - `loan_income_ratio` = `loan_amnt / person_income`
   - `int_rate_x_grade` = interaksi suku bunga × grade numerik
4. **Modelling** — urutan prioritas:
   - LightGBM (`lgb.LGBMClassifier`) — baseline utama
   - XGBoost — alternatif
   - Logistic Regression — baseline sederhana
5. **Validation** — `StratifiedKFold(n_splits=5)`, metrik `roc_auc`
6. **Hyperparameter tuning** — `Optuna` atau `GridSearchCV` pada LightGBM
7. **Submission** — `predict_proba()[:, 1]`, simpan ke CSV dengan kolom `sample_id`, `loan_status`

## Submission Format

```csv
sample_id,loan_status
TS35462746,0.82
TS48835513,0.13
...
```

Gunakan `sample_id` dari `test.csv`, bukan index integer.

## Target Score

AUC-ROC ≥ 0.92 realistis dengan LightGBM yang di-tune.

## Notes

- Data tidak perlu time-based split (bukan time series)
- Tidak ada missing value, tidak perlu imputation
- Metrik AUC tidak terpengaruh threshold — submit probabilitas mentah, bukan 0/1
- Batas submission Kaggle: **10 Juni 2026 23:59 WIB**
- Batas laporan ke LMS: **14 Juni 2026 23:59 WIB** (maks 2 halaman + kode + model)
