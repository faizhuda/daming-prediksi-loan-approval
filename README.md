# Prediksi Loan Approval — KOM1338 Data Mining

Klasifikasi biner untuk memprediksi `loan_status` (0 = lunas, 1 = default) pada
[kompetisi Kaggle KOM1338](https://www.kaggle.com/competitions/kom-1338-data-mining-prediksi-loan-approval/).
Keluaran berupa **probabilitas**, dievaluasi dengan **AUC-ROC**.

## Hasil

Validasi **StratifiedKFold(5)**, out-of-fold (OOF) AUC — semua dengan `random_state=42`:

| Model | OOF AUC |
|---|---|
| LightGBM | 0.93223 |
| XGBoost | 0.93226 |
| CatBoost | 0.93030 |
| **Ensemble rank-blend (3·LGB + 3·XGB + 1·Cat)** | **0.93247** |

Submission akhir memakai ensemble rank-blend. Seluruh hasil **reproducible hanya dari
`train.csv`/`test.csv`** — tanpa data eksternal.

## Struktur

```
.
├── src/loan_pipeline.py   # Core logic (DRY): load, FE, CV, tuning, blending
├── train_model.py         # Orchestrator: tuning + ensemble + simpan artifact
├── build_notebook.py      # Generator notebook.ipynb
├── notebook.ipynb         # Analisis akademik lengkap (EDA + narasi + modeling)
├── requirements.txt       # Dependensi ter-pin (Python 3.11)
├── models/                # Artifact: best_params.json, metrics.json, model .pkl, OOF
├── reports/figures/       # Grafik EDA & hasil
├── train.csv / test.csv / sample_submission.csv
└── submission.csv         # Output akhir (sample_id, loan_status=probabilitas)
```

## Cara menjalankan

```bash
pip install -r requirements.txt

# Opsi A — finalisasi cepat dengan hyperparameter tervalidasi (~6 menit)
python train_model.py --fast

# Opsi B — full tuning Optuna (LGB 60 + XGB 50 + Cat 25 trial, ~30-50 menit)
python train_model.py

# Regenerasi & eksekusi notebook akademik
python build_notebook.py
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace
```

Keduanya menulis `submission.csv` dan artifact ke `models/`.

## Ringkasan metodologi

- **Preprocessing:** ordinal-encode `loan_grade` (A→1 … G→7, karena default rate naik
  monoton terhadap grade); one-hot untuk `person_home_ownership`, `loan_intent`,
  `cb_person_default_on_file` (LightGBM/XGBoost) atau categorical native (CatBoost).
- **Feature engineering:** sengaja minimal — hanya interaksi `int_rate_x_grade`.
  Penambahan banyak fitur turunan justru **menurunkan** AUC (overfitting noise), dan
  rasio loan/income dibuang karena redundan (korelasi 0,9995 dengan `loan_percent_income`).
- **Tanpa class weighting / resampling:** AUC adalah metrik ranking; pembobotan
  mendistorsi probabilitas dan terbukti menurunkan OOF AUC (0,9314 → 0,9319 tanpa bobot).
- **Model:** tiga gradient boosting (LightGBM, XGBoost, CatBoost) di-tune dengan Optuna,
  digabung via rank-blend dengan bobot hasil grid-search pada OOF AUC.

Detail lengkap beserta justifikasi dan grafik ada di [`notebook.ipynb`](notebook.ipynb).

## Keterbatasan

Dengan data kompetisi saja, ceiling AUC OOF berada di kisaran ~0,93. Menembus papan
peringkat teratas (~0,94+) umumnya memerlukan augmentasi dataset publik asal
(`credit_risk_dataset`); pendekatan itu **tidak** dipakai di sini demi reproducibility
murni dan integritas akademik.
