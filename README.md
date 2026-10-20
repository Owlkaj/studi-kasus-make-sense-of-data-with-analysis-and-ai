# 📉 Prediksi Keterlambatan Pembayaran Invoice > 30 Hari — PT Maju Bersama

Studi kasus *end-to-end* **Make Sense of Data with Analysis and AI**, berdasarkan Kuliah Praktisi Informatika
(Narasumber: **Andria Arisal, BRIN**). Proyek ini memprediksi invoice yang berpotensi **dibayar terlambat lebih dari 30 hari**
agar tim Finance dapat mengambil tindakan sebelum kas tertahan.

> ⚠️ **Dataset 100% sintetis/fiktif** (dibuat oleh `src/generate_dataset.py`) dan hanya untuk pembelajaran.
> Angka hasil di bawah **tidak mewakili kondisi perusahaan nyata**.

---

## 1. Latar Belakang Bisnis

**PT Maju Bersama** (perusahaan distributor) menghadapi kondisi *"untung di atas kertas, kas menipis"*:

| Indikator | Kondisi |
|---|---|
| Sales Revenue | Naik |
| Net Profit | Turun **14%** |
| Piutang (Receivables) | Melonjak **300%** |
| Cash Flow | Menipis (**-Rp51 juta**) |

**Masalah analitis:** invoice mana yang berpotensi terlambat dibayar > 30 hari?
**Target:** `is_late_30` = 1 jika `days_late > 30`, selain itu 0.

## 2. Poin Penting dari Pemateri & Penerapannya

| Poin | Penerapan di proyek |
|---|---|
| Hindari **data leakage** | `actual_payment_date`, `days_late`, `days_overdue` dibuang dari fitur (ada demo AUC = 1.0 yang tidak realistis jika dipakai) |
| Tangani **imbalanced data** | `SMOTE` hanya pada data latih; ada perbandingan dengan baseline tanpa SMOTE |
| **4 Levels of Analytics** | Descriptive → Diagnostic → Predictive → Prescriptive (lihat bagian 3) |
| **Machine Learning klasifikasi** | Random Forest Classifier |

## 3. 4 Levels of Analytics

| Level | Pertanyaan | Yang dilakukan di proyek | Output |
|---|---|---|---|
| **Descriptive** | *Apa yang terjadi?* | KPI piutang, sebaran `days_late`, proporsi kelas, tren nilai invoice bulanan | `01_`, `02_` |
| **Diagnostic** | *Mengapa terjadi?* | % telat per segmen, metode bayar, band skor kredit, termin; korelasi fitur dengan target | `03_`, `04_` |
| **Predictive** | *Apa yang akan terjadi?* | Random Forest + SMOTE memprediksi probabilitas telat > 30 hari | `05_`–`07_` |
| **Prescriptive** | *Apa yang harus dilakukan?* | `predict_invoice()` → level risiko + tindakan; daftar prioritas penagihan berdasar *expected exposure* | `reports/priority_invoices.csv` |

## 4. Struktur Repository

```
pt-maju-bersama-late-payment/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── invoices_pt_maju_bersama.csv      # dataset sintetis (200 baris)
├── notebooks/
│   └── main_analysis.ipynb               # analisis lengkap (CRISP-DM) + output
├── src/
│   ├── generate_dataset.py               # pembangkit dataset sintetis
│   └── main.py                           # versi skrip dari notebook
├── models/
│   └── .gitkeep                          # rf_late_payment.joblib dibuat otomatis
└── reports/
    ├── priority_invoices.csv             # daftar prioritas penagihan
    └── figures/                          # grafik hasil analisis (*.png)
```

### Kamus Data (`data/invoices_pt_maju_bersama.csv`)

| Kolom | Keterangan | Dipakai sebagai fitur? |
|---|---|---|
| `invoice_id`, `customer_id` | Identitas | ❌ |
| `invoice_date`, `due_date` | Tanggal terbit & jatuh tempo | ❌ |
| `customer_segment` | Retail / Wholesale / Corporate / Government | ✅ |
| `credit_score` | Skor kredit pelanggan (300–850), ada sedikit missing | ✅ |
| `invoice_amount` | Nilai invoice (Rp) | ✅ |
| `payment_terms_days` | Termin pembayaran (14/30/45/60 hari) | ✅ |
| `payment_method` | Transfer Bank, Virtual Account, Giro, Kredit/Tempo, Tunai (ada sedikit missing) | ✅ |
| `prev_late_count` | Jumlah keterlambatan sebelumnya | ✅ |
| `actual_payment_date` | Tanggal bayar aktual | 🚫 **leakage** |
| `days_late` | Selisih hari bayar vs jatuh tempo | 🚫 **leakage** |
| `days_overdue` | `max(days_late, 0)` | 🚫 **leakage** |
| `is_late_30` | **Target** (1 = telat > 30 hari) | — |

## 5. Alur CRISP-DM di Notebook

1. Business Understanding
2. Loading Data & Data Understanding
3. Descriptive Analytics
4. Diagnostic Analytics
5. Data Cleaning & **Anti-Leakage**
6. Train-Test Split (75:25, *stratified*)
7. Preprocessing & Encoding (imputasi median/modus, `StandardScaler`, `OneHotEncoder`)
8. **SMOTE** (hanya data latih)
9. Modeling: Random Forest (baseline vs + SMOTE)
10. Evaluasi: Classification Report, ROC-AUC, Confusion Matrix, kurva ROC
11. Threshold tuning
12. Feature importance & Stratified 5-Fold Cross-Validation
13. Prescriptive: fungsi `predict_invoice()` + rekomendasi tindakan
14. Prioritas penagihan (batch)
15. Simpan model (`joblib`)

## 6. Cara Menjalankan

**Prasyarat:** Python 3.9+.

```bash
# 1) Clone & masuk ke folder
git clone https://github.com/<username>/pt-maju-bersama-late-payment.git
cd pt-maju-bersama-late-payment

# 2) (Opsional) buat virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3) Pasang dependensi
pip install -r requirements.txt

# 4) (Opsional) bangkitkan ulang dataset sintetis
python src/generate_dataset.py

# 5a) Jalankan notebook
jupyter notebook notebooks/main_analysis.ipynb
# 5b) ...atau jalankan versi skrip (grafik disimpan ke reports/figures/)
python src/main.py
```

**Dependensi utama:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`, `jupyter`.

### Contoh Penggunaan Fungsi Prediksi

```python
predict_invoice({
    "customer_segment": "Government",
    "credit_score": 520,
    "invoice_amount": 250_000_000,
    "payment_terms_days": 60,
    "payment_method": "Giro",
    "prev_late_count": 3,
})
# -> {'probabilitas_telat': 0.871, 'level_risiko': 'TINGGI', 'nilai_berisiko': 'Rp217.752.993',
#     'tindakan_disarankan': [...]}
```

Level risiko: **TINGGI** (≥ 0,60), **SEDANG** (≥ 0,35), **RENDAH** (< 0,35).

## 7. Hasil

### Deskriptif & Diagnostik (data sintetis)

- 200 invoice, **41 (20,5%) telat > 30 hari**; nilainya **Rp4,32 miliar** atau **24,3%** dari total nilai invoice Rp17,81 miliar → telat pada invoice besar lebih "mahal" bagi kas.
- Tingkat keterlambatan tertinggi: segmen **Government (40%)** dan metode **Giro (37%)**; terendah: **Tunai (9%)** dan **Virtual Account (10%)**.
- Korelasi dengan target: `prev_late_count` (+0,41) dan `credit_score` (−0,29) paling menonjol.

### Evaluasi Model (data uji 50 invoice, 10 di antaranya telat; ambang 0,5)

| Model | Precision (telat) | Recall (telat) | F1 (telat) | ROC-AUC |
|---|---|---|---|---|
| Random Forest tanpa SMOTE | 0,80 | 0,40 | 0,53 | 0,74 |
| **Random Forest + SMOTE** | 0,56 | **0,50** | 0,53 | **0,75** |

**Stratified 5-Fold CV (RF + SMOTE):** ROC-AUC **0,78 ± 0,12**, Recall 0,42 ± 0,14, Precision 0,53 ± 0,09.

Threshold tuning (data uji): ambang **0,2 → Recall 0,70**, **0,4 → Recall 0,60 & Precision 0,60**, **0,6 → Precision 0,67 tetapi Recall 0,40**.

**Fitur terpenting:** `prev_late_count` (0,33), `credit_score` (0,15), `invoice_amount` (0,12), metode bayar Giro & Virtual Account.

> 📝 **Catatan jujur:** data uji hanya berisi 10 invoice telat, sehingga selisih antar model **tidak signifikan secara statistik**
> dan angka bisa bergeser jika `random_state` atau dataset diubah. Pada data ini SMOTE menaikkan Recall tetapi menurunkan Precision
> (*trade-off*), bukan "memperbaiki segalanya". Gunakan hasil CV dan threshold tuning sebagai acuan, dan jalankan ulang untuk melihat angka Anda sendiri.

## 8. Rekomendasi Bisnis

1. **Skor setiap invoice saat diterbitkan**, bukan setelah jatuh tempo, agar masih ada waktu bertindak.
2. **Risiko TINGGI:** eskalasi ke manajer, minta uang muka/pembayaran sebagian, telepon H-7 & H-1, tinjau limit kredit.
3. **Risiko SEDANG:** reminder otomatis H-7 & H-3, konfirmasi jadwal bayar, tawarkan diskon pembayaran dini.
4. **Prioritaskan penagihan berdasarkan *expected exposure*** (probabilitas × nilai invoice), bukan hanya nilai invoice.
5. **Dorong Virtual Account/transfer** dan kurangi Giro serta Kredit/Tempo untuk pelanggan berisiko.
6. **Perketat kebijakan kredit** untuk pelanggan dengan riwayat telat berulang dan skor kredit rendah; evaluasi termin segmen Government.
7. Pilih **threshold sesuai biaya kesalahan**: jika melewatkan invoice berisiko lebih mahal, turunkan ambang (mis. 0,3–0,4) agar Recall naik.

## 9. Keterbatasan & Pengembangan Lanjut

- Data sintetis dan kecil (200 baris) → hasil sangat sensitif; gunakan data historis nyata untuk validasi.
- `SMOTE` pada fitur hasil One-Hot menghasilkan nilai pecahan; coba `SMOTENC` atau `class_weight="balanced"`.
- Lakukan *hyperparameter tuning* (`GridSearchCV`), bandingkan dengan Decision Tree / Gradient Boosting.
- Pantau *model drift* dan kalibrasi probabilitas sebelum dipakai untuk keputusan nyata.
- Pertimbangkan validasi berbasis waktu (*time-based split*) pada data nyata.

## 10. Referensi

- Kuliah Praktisi Informatika: *Make Sense of Data with Analysis and AI* — Andria Arisal (BRIN).
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research, 16*, 321–357.
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Chapman, P., et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
