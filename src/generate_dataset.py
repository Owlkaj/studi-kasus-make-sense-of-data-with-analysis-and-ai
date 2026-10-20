"""
generate_dataset.py
===================
Membangkitkan dataset SINTETIS invoice PT Maju Bersama (200 baris).

Dataset ini FIKTIF dan hanya untuk keperluan pembelajaran. Pola keterlambatan
sengaja dibuat realistis (skor kredit rendah, nilai invoice besar, riwayat
telat, metode bayar tertentu -> risiko telat > 30 hari lebih tinggi) dengan
proporsi kelas minoritas (telat > 30 hari) sekitar 18-20% => data IMBALANCED.

Cara pakai (dari root repository):
    python src/generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_INVOICES = 200          # jumlah baris invoice
N_CUSTOMERS = 60          # jumlah pelanggan unik
TARGET_LATE_RATE = 0.18   # proporsi invoice telat > 30 hari yang diinginkan
SIGNAL_STRENGTH = 1.8     # makin besar -> pola risiko makin jelas (data kecil butuh sinyal cukup kuat)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------
# 1) Tabel pelanggan: segmen + skor kredit (tetap per pelanggan)
# ---------------------------------------------------------------
segments = ["Retail", "Wholesale", "Corporate", "Government"]
seg_prob = [0.35, 0.30, 0.25, 0.10]
seg_credit_mean = {"Retail": 600, "Wholesale": 650, "Corporate": 700, "Government": 680}
seg_credit_std = {"Retail": 70, "Wholesale": 60, "Corporate": 50, "Government": 50}

cust_segment = rng.choice(segments, size=N_CUSTOMERS, p=seg_prob)
cust_credit = np.array([
    rng.normal(seg_credit_mean[s], seg_credit_std[s]) for s in cust_segment
]).clip(300, 850).round().astype(int)
customers = pd.DataFrame({
    "customer_id": [f"CUST-{i:03d}" for i in range(1, N_CUSTOMERS + 1)],
    "customer_segment": cust_segment,
    "credit_score": cust_credit,
})

# ---------------------------------------------------------------
# 2) Tabel invoice: sampling pelanggan + atribut transaksi
# ---------------------------------------------------------------
inv = customers.sample(n=N_INVOICES, replace=True, random_state=SEED).reset_index(drop=True)

# Nilai invoice (Rupiah) ~ lognormal, median berbeda per segmen
seg_median = {"Retail": 15e6, "Wholesale": 60e6, "Corporate": 120e6, "Government": 200e6}
inv["invoice_amount"] = [
    round(rng.lognormal(np.log(seg_median[s]), 0.6), -3) for s in inv["customer_segment"]
]
inv["invoice_amount"] = inv["invoice_amount"].astype(int)

# Termin pembayaran (hari) tergantung segmen
terms_by_seg = {
    "Retail": ([14, 30], [0.6, 0.4]),
    "Wholesale": ([30, 45], [0.6, 0.4]),
    "Corporate": ([30, 45, 60], [0.3, 0.4, 0.3]),
    "Government": ([45, 60], [0.4, 0.6]),
}
inv["payment_terms_days"] = [
    int(rng.choice(terms_by_seg[s][0], p=terms_by_seg[s][1])) for s in inv["customer_segment"]
]

# Metode pembayaran
methods = ["Transfer Bank", "Virtual Account", "Giro", "Kredit/Tempo", "Tunai"]
inv["payment_method"] = rng.choice(methods, size=N_INVOICES, p=[0.35, 0.25, 0.15, 0.15, 0.10])

# Jumlah keterlambatan sebelumnya (lebih banyak jika skor kredit rendah)
lam = ((700 - inv["credit_score"]) / 120).clip(lower=0.1)
inv["prev_late_count"] = np.minimum(rng.poisson(lam), 6)

# Tanggal invoice sepanjang 2025
start = pd.Timestamp("2025-01-01")
inv["invoice_date"] = start + pd.to_timedelta(rng.integers(0, 365, N_INVOICES), unit="D")
inv["due_date"] = inv["invoice_date"] + pd.to_timedelta(inv["payment_terms_days"], unit="D")

# ---------------------------------------------------------------
# 3) Mekanisme keterlambatan (logit) -> days_late -> label target
# ---------------------------------------------------------------
seg_eff = {"Retail": 0.2, "Wholesale": 0.0, "Corporate": -0.1, "Government": 0.8}
method_eff = {"Transfer Bank": 0.0, "Virtual Account": -0.5, "Giro": 0.6,
              "Kredit/Tempo": 0.5, "Tunai": -0.8}

z = (
    -0.9 * (inv["credit_score"] - 650) / 80
    + 0.5 * np.log(inv["invoice_amount"] / 50e6)
    + 0.5 * inv["prev_late_count"]
    + inv["customer_segment"].map(seg_eff)
    + inv["payment_method"].map(method_eff)
    + 0.4 * (inv["payment_terms_days"] >= 60)
) * SIGNAL_STRENGTH

# Cari intercept (bisection) agar rata-rata probabilitas ~ TARGET_LATE_RATE
lo, hi = -10.0, 10.0
for _ in range(60):
    mid = (lo + hi) / 2
    if (1 / (1 + np.exp(-(z + mid)))).mean() > TARGET_LATE_RATE:
        hi = mid
    else:
        lo = mid
p_late = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
is_late = rng.random(N_INVOICES) < p_late

# days_late = selisih hari antara pembayaran aktual & jatuh tempo (negatif = lebih awal)
inv["days_late"] = np.where(
    is_late,
    np.minimum(31 + rng.gamma(2.0, 10.0, N_INVOICES).astype(int), 120),
    rng.integers(-5, 31, N_INVOICES),
)
inv["actual_payment_date"] = inv["due_date"] + pd.to_timedelta(inv["days_late"], unit="D")
inv["days_overdue"] = inv["days_late"].clip(lower=0)
inv["is_late_30"] = (inv["days_late"] > 30).astype(int)   # TARGET

# ---------------------------------------------------------------
# 4) Sisipkan sedikit missing value (untuk latihan data cleaning)
# ---------------------------------------------------------------
inv.loc[rng.choice(N_INVOICES, 4, replace=False), "credit_score"] = np.nan
inv.loc[rng.choice(N_INVOICES, 3, replace=False), "payment_method"] = np.nan

# ---------------------------------------------------------------
# 5) Rapikan kolom, beri ID invoice, simpan ke CSV
# ---------------------------------------------------------------
inv = inv.sort_values("invoice_date").reset_index(drop=True)
inv.insert(0, "invoice_id", [f"INV-2025-{i:04d}" for i in range(1, N_INVOICES + 1)])
cols = ["invoice_id", "customer_id", "invoice_date", "due_date", "customer_segment",
        "credit_score", "invoice_amount", "payment_terms_days", "payment_method",
        "prev_late_count", "actual_payment_date", "days_late", "days_overdue", "is_late_30"]
inv = inv[cols]

out_path = Path(__file__).resolve().parents[1] / "data" / "invoices_pt_maju_bersama.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
inv.to_csv(out_path, index=False)

print(f"Dataset disimpan ke: {out_path}")
print(f"Jumlah baris      : {len(inv)}")
print(f"Invoice telat >30 : {inv['is_late_30'].sum()} ({inv['is_late_30'].mean():.1%})")
