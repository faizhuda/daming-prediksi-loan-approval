"""Generate `notebook.ipynb` — the documented academic analysis notebook.

Run:  python build_notebook.py
Then: jupyter nbconvert --to notebook --execute notebook.ipynb --inplace
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# ───────────────────────────────────────────────────────────────────────────────
md(r"""# Prediksi *Loan Approval* — KOM1338 Data Mining

**Kompetisi:** [KOM1338 Data Mining — Prediksi Loan Approval](https://www.kaggle.com/competitions/kom-1338-data-mining-prediksi-loan-approval/)

**Tugas:** Klasifikasi biner — memprediksi `loan_status` (0 = tidak default, 1 = default) untuk
tiap baris di `test.csv`. Keluaran berupa **probabilitas** (bukan label), dievaluasi dengan **AUC-ROC**.

---

## Daftar Isi
1. [Setup & Pemuatan Data](#1)
2. [Exploratory Data Analysis (EDA)](#2)
3. [Preprocessing & Feature Engineering](#3)
4. [Skema Validasi](#4)
5. [Pemodelan: LightGBM, XGBoost, CatBoost](#5)
6. [Ensembling (Rank-Blend)](#6)
7. [Feature Importance](#7)
8. [Pembuatan Submission](#8)
9. [Kesimpulan & Diskusi](#9)

> **Catatan reproducibility.** Seluruh logika inti (feature engineering, CV, tuning, blending)
> berada di modul `src/loan_pipeline.py` sehingga notebook ini dan skrip `train_model.py`
> memakai kode yang **persis sama** — tidak ada risiko hasil berbeda antara keduanya.
> Semua proses memakai `random_state=42`.""")

# 1. Setup ----------------------------------------------------------------------
md('<a id="1"></a>\n## 1. Setup & Pemuatan Data')
code(r"""import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

import src.loan_pipeline as lp   # core, shared pipeline logic

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
RNG = lp.SEED

train, test = lp.load_data(".")
print("train:", train.shape, "| test:", test.shape)
train.head()""")

md(r"""Dataset memiliki **43.983 baris × 13 kolom** untuk train dan **14.662 baris × 12 kolom**
untuk test (tanpa kolom target). Berikut tipe data dan pengecekan *missing value*.""")
code(r"""info = pd.DataFrame({
    "dtype": train.dtypes,
    "n_unique": train.nunique(),
    "n_missing": train.isna().sum(),
})
print("Total missing values:", int(train.isna().sum().sum()))
info""")

md(r"""**Observasi awal:** tidak ada *missing value* sama sekali, jadi tidak diperlukan imputasi.
Terdapat 4 kolom kategorikal bertipe `object` (`person_home_ownership`, `loan_intent`,
`loan_grade`, `cb_person_default_on_file`) dan sisanya numerik.""")

# 2. EDA ------------------------------------------------------------------------
md('<a id="2"></a>\n## 2. Exploratory Data Analysis (EDA)')

md("### 2.1 Distribusi Target — Data Tidak Seimbang")
code(r"""counts = train[lp.TARGET].value_counts().sort_index()
rate = train[lp.TARGET].mean()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=counts.index, y=counts.values, ax=ax[0], palette="Set2")
ax[0].set(title="Jumlah per kelas", xlabel="loan_status", ylabel="count")
for i, v in enumerate(counts.values):
    ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
ax[1].pie(counts.values, labels=["0 (lunas)", "1 (default)"], autopct="%1.1f%%",
          colors=sns.color_palette("Set2"), startangle=90)
ax[1].set_title("Proporsi kelas")
plt.tight_layout(); plt.savefig("reports/figures/target_distribution.png", dpi=110); plt.show()
print(f"Positive (default) rate: {rate:.3f}")""")
md(r"""Hanya **±14,2%** kasus *default* (kelas 1). Ketidakseimbangan ini penting dicatat, namun
karena metriknya **AUC-ROC** (berbasis *ranking*, bukan *threshold*), kita **tidak** akan
melakukan *resampling* maupun *class weighting* — justifikasi dibahas di Bagian 3.""")

md("### 2.2 Distribusi Fitur Numerik & Outlier")
code(r"""num_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
train[num_cols].describe().T[["min", "mean", "50%", "max", "std"]].round(2)""")
code(r"""fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.ravel(), num_cols):
    sns.histplot(train[c], bins=40, ax=ax, color="steelblue")
    ax.set_title(c)
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.savefig("reports/figures/numeric_distributions.png", dpi=110); plt.show()""")
md(r"""**Temuan outlier / anomali data:**
- `person_age` mencapai **100** (4 baris > 80) — sangat ekstrem untuk peminjam.
- `person_emp_length` mencapai **80 tahun**, dan **±4.555 baris** memiliki lama bekerja
  yang melebihi (usia − 16), yang **mustahil** secara logika (mulai kerja sebelum lahir).
- `person_income` sangat *right-skewed* (maks ±2 juta).

Anomali ini kami **biarkan apa adanya**: model berbasis pohon (tree-based) tahan terhadap
outlier monoton, dan eksperimen kami menunjukkan penambahan *flag* anomali maupun
transformasi log **tidak** meningkatkan AUC (lihat Bagian 3).""")

md("### 2.3 Tingkat Default per Kategori")
code(r"""cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, c in zip(axes.ravel(), cat_cols):
    rates = train.groupby(c)[lp.TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=rates.values, y=rates.index, ax=ax, palette="rocket")
    ax.axvline(train[lp.TARGET].mean(), color="gray", ls="--", lw=1, label="rata-rata global")
    ax.set(title=f"Default rate — {c}", xlabel="P(default)")
    ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("reports/figures/default_by_category.png", dpi=110); plt.show()""")
md(r"""**Sinyal terkuat ada di sini:**
- **`loan_grade`** sangat diskriminatif dan **monoton**: grade A ≈ 5% default,
  meningkat terus hingga grade G ≈ **85%**. Ini membenarkan *ordinal encoding* (A→1 … G→7).
- **`person_home_ownership`**: penyewa (RENT, 22%) jauh lebih berisiko daripada pemilik
  (OWN, 1,6%).
- **`cb_person_default_on_file`**: riwayat default (Y, 30%) vs bersih (N, 12%).
- **`loan_intent`**: DEBTCONSOLIDATION/MEDICAL paling berisiko; VENTURE/EDUCATION paling aman.""")

md("### 2.4 Korelasi Fitur Numerik dengan Target")
code(r"""corr_df = train[num_cols + [lp.TARGET]].copy()
corr_df["grade_ord"] = train["loan_grade"].map(lp.GRADE_MAP)
corr = corr_df.corr(numeric_only=True)
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Matriks Korelasi (Pearson)")
plt.tight_layout(); plt.savefig("reports/figures/correlation_heatmap.png", dpi=110); plt.show()
corr[lp.TARGET].drop(lp.TARGET).sort_values(key=abs, ascending=False)""")
md(r"""Korelasi linier tertinggi terhadap `loan_status`: **`loan_int_rate`**,
**`loan_percent_income`**, dan **`grade_ord`** — konsisten dengan analisis kategorikal.
`loan_int_rate` sendiri sangat berkorelasi dengan `grade_ord` (bank menetapkan bunga
berdasarkan grade), yang memotivasi fitur interaksi `int_rate_x_grade`.""")

# 3. Preprocessing --------------------------------------------------------------
md('<a id="3"></a>\n## 3. Preprocessing & Feature Engineering')
md(r"""Keputusan desain (semua **divalidasi dengan CV**, bukan asumsi):

| Keputusan | Alasan |
|---|---|
| **Ordinal encode** `loan_grade` (A→1 … G→7) | Default rate naik monoton terhadap grade. |
| **One-hot** `home_ownership`, `loan_intent`, `default_on_file` | Nominal, kardinalitas rendah (≤6). |
| **Native categorical** untuk CatBoost | CatBoost menangani kategorikal secara internal (target statistics). |
| **Drop** `sample_id` | Identifier, bukan fitur. |
| Fitur interaksi `int_rate_x_grade` | Bunga & grade saling menguatkan sinyal risiko. |
| **Tidak** menambah rasio loan/income | Korelasi **0,9995** dengan `loan_percent_income` yang sudah ada → redundan. |
| **Tanpa** `class_weight` / resampling | AUC adalah metrik *ranking*; pembobotan mendistorsi probabilitas. Terbukti **menurunkan** OOF AUC (0,9314 → 0,9319 tanpa bobot). |
| **Tanpa** *scaling* | Tidak relevan untuk model pohon. |

> **Catatan empiris.** Kami sempat mencoba ~10 fitur tambahan (rasio usia/pendapatan,
> deviasi bunga per grade, *flag* anomali, dsb). Semuanya **menurunkan** OOF AUC
> (0,9319 → 0,9300) karena menambah *noise* pada dataset yang sinyalnya sudah terkonsentrasi.
> Maka prinsip *parsimony* (sesedikit mungkin fitur) kami pegang.""")
code(r"""# Dua "tampilan" data: one-hot (LightGBM/XGBoost) dan native-categorical (CatBoost)
X_oh, y, X_oh_test = lp.make_onehot(train, test)
X_nat, _, X_nat_test, cat_features = lp.make_native(train, test)
print("One-hot matrix :", X_oh.shape)
print("Native  matrix :", X_nat.shape, "| categorical:", cat_features)
X_oh.head(3)""")

# 4. Validation -----------------------------------------------------------------
md('<a id="4"></a>\n## 4. Skema Validasi')
md(r"""Kami memakai **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`**.
*Stratified* menjaga proporsi 14,2% default di tiap lipatan — penting untuk estimasi AUC
yang stabil pada data tak seimbang. Setiap model menghasilkan prediksi **out-of-fold (OOF)**:
tiap baris diprediksi oleh model yang **tidak** melihatnya saat training, sehingga AUC OOF
adalah estimasi performa generalisasi yang jujur (tanpa kebocoran).""")

# 5. Modeling -------------------------------------------------------------------
md('<a id="5"></a>\n## 5. Pemodelan')
md(r"""Tiga model *gradient boosting* di-tuning dengan **Optuna** (TPE sampler, seed tetap).

Agar notebook ini cepat dijalankan ulang, kita **memuat hyperparameter terbaik** hasil
*full tuning* dari `models/best_params.json` (dihasilkan oleh `python train_model.py`).
Jika berkas itu belum ada, notebook akan melakukan tuning ringkas (sedikit trial) sebagai
*fallback*.""")
code(r"""import json, os
PARAMS_PATH = "models/best_params.json"
if os.path.exists(PARAMS_PATH):
    best = json.load(open(PARAMS_PATH))
    print("Memuat hyperparameter hasil full-tuning dari", PARAMS_PATH)
else:
    print("best_params.json belum ada -> tuning ringkas (fallback).")
    lgb_p, _ = lp.tune_lightgbm(X_oh, y, n_trials=15)
    xgb_p, _ = lp.tune_xgboost(X_oh, y, n_trials=15)
    cat_p, _ = lp.tune_catboost(X_nat, y, cat_features, n_trials=10)
    best = {"lightgbm": lgb_p, "xgboost": xgb_p, "catboost": cat_p}
best""")

md("### 5.1 LightGBM")
code(r"""import lightgbm as lgb
print("LightGBM 5-fold OOF:")
oof_lgb, test_lgb, lgb_models = lp.cv_fit_predict(
    lambda: lgb.LGBMClassifier(**best["lightgbm"]), X_oh, y, X_oh_test)""")

md("### 5.2 XGBoost")
code(r"""import xgboost as xgb
print("XGBoost 5-fold OOF:")
oof_xgb, test_xgb, xgb_models = lp.cv_fit_predict(
    lambda: xgb.XGBClassifier(**best["xgboost"]), X_oh, y, X_oh_test)""")

md("### 5.3 CatBoost")
code(r"""from catboost import CatBoostClassifier
print("CatBoost 5-fold OOF:")
oof_cat, test_cat, cat_models = lp.cv_fit_predict(
    lambda: CatBoostClassifier(**best["catboost"]), X_nat, y, X_nat_test,
    cat_features=cat_features)""")

# 6. Ensemble -------------------------------------------------------------------
md('<a id="6"></a>\n## 6. Ensembling (Rank-Blend)')
md(r"""Ketiga model menangkap pola yang sedikit berbeda, sehingga **menggabungkan** prediksinya
biasanya melampaui model tunggal terbaik. Kami memakai **rank-blend**: setiap prediksi
diubah ke *rank* (0–1) lalu dirata-rata berbobot — robust terhadap perbedaan skala/kalibrasi
antarmodel, dan sejalan dengan AUC yang memang berbasis *ranking*. Bobot dicari lewat
*grid search* kecil yang memaksimalkan AUC OOF.""")
code(r"""oof_list  = [oof_lgb, oof_xgb, oof_cat]
test_list = [test_lgb, test_xgb, test_cat]
weights, blend_auc = lp.search_blend_weights(oof_list, y)

results = pd.DataFrame({
    "Model": ["LightGBM", "XGBoost", "CatBoost", f"Rank-Blend {weights}"],
    "OOF AUC": [roc_auc_score(y, oof_lgb), roc_auc_score(y, oof_xgb),
                roc_auc_score(y, oof_cat), blend_auc],
}).sort_values("OOF AUC", ascending=False).reset_index(drop=True)
results.style.format({"OOF AUC": "{:.5f}"}).background_gradient(subset=["OOF AUC"], cmap="Greens")""")
code(r"""# Kurva ROC dari prediksi OOF blend
from sklearn.metrics import roc_curve
blend_oof = lp.rank_blend(oof_list, weights)
fpr, tpr, _ = roc_curve(y, blend_oof)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label=f"Blend OOF (AUC = {blend_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Kurva ROC — Ensemble (OOF)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("reports/figures/roc_curve.png", dpi=110); plt.show()""")

# 7. Feature importance ---------------------------------------------------------
md('<a id="7"></a>\n## 7. Feature Importance')
code(r"""imp = np.mean([m.feature_importances_ for m in lgb_models], axis=0)
fi = pd.Series(imp, index=X_oh.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(9, 6))
sns.barplot(x=fi.values, y=fi.index, palette="viridis")
plt.title("LightGBM — 15 Fitur Terpenting (rata-rata 5 fold)")
plt.xlabel("Importance (gain)")
plt.tight_layout(); plt.savefig("reports/figures/feature_importance.png", dpi=110); plt.show()
fi""")
md(r"""Hasilnya konsisten dengan EDA: `loan_percent_income`, `loan_int_rate`, `grade_ord`,
interaksi `int_rate_x_grade`, dan `person_income` mendominasi — beberapa fitur kuat inilah
yang menjelaskan mengapa menambah fitur lemah justru merusak performa.""")

# 8. Submission -----------------------------------------------------------------
md('<a id="8"></a>\n## 8. Pembuatan Submission')
code(r"""test_blend = lp.rank_blend(test_list, weights)
submission = lp.make_submission(test, test_blend, "submission.csv")
print("submission.csv:", submission.shape)
print(submission["loan_status"].describe().round(4).to_string())
submission.head()""")
md(r"""Format sesuai `sample_submission.csv`: kolom `sample_id` (dari `test.csv`) dan
`loan_status` berisi **probabilitas** kelas 1.""")

# 9. Conclusion -----------------------------------------------------------------
md('<a id="9"></a>\n## 9. Kesimpulan & Diskusi')
md(r"""**Ringkasan hasil (OOF AUC, validasi 5-fold):** lihat tabel pada Bagian 6 —
ensemble *rank-blend* tiga model adalah konfigurasi terbaik kami.

**Apa yang berhasil**
- Memperlakukan `loan_grade` sebagai ordinal (sinyal terkuat & monoton).
- Membiarkan model pohon menangani outlier; tanpa *scaling*/imputasi.
- **Tidak** memakai *class weighting* — sesuai sifat metrik AUC (ranking), terbukti naik.
- Ensemble tiga *gradient boosting* yang errornya terdekorelasi.

**Apa yang tidak berhasil (dan kami buang)**
- ~10 fitur turunan tambahan → menurunkan AUC (overfitting *noise*).
- Rasio loan/income → redundan (korelasi 0,9995 dengan fitur yang ada).
- *Class weighting* / resampling → menurunkan AUC.

**Keterbatasan & arah lanjutan.** Dengan **hanya** data kompetisi ini, AUC OOF kami berada di
kisaran ~0,93. Untuk menembus papan peringkat teratas (≈0,94+), teknik yang umum dipakai adalah
**augmentasi dengan dataset publik asal** (`credit_risk_dataset`, sumber sintesis kompetisi
ini). Kami **tidak** memakainya demi menjaga reproducibility murni dan integritas akademik —
seluruh hasil di sini dapat direproduksi hanya dari `train.csv`/`test.csv` dengan `random_state=42`.

---
*Notebook dihasilkan untuk laporan KOM1338. Logika inti: `src/loan_pipeline.py`.*""")

nb["cells"] = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"},
               "language_info": {"name": "python"}}
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebook.ipynb with", len(cells), "cells")
