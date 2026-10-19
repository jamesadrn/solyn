"""Pengaturan bersama untuk lima notebook: daftar saham, fitur, target, warna grafik, dan kuartal 2025."""
from pathlib import Path

import pandas as pd

# urutan saham harus sama saat fitur, prediksi, covariance, dan bobot dipasangkan
tickers = [
    "BBCA",
    "BBNI",
    "BBRI",
    "BBTN",
    "BDMN",
    "BJBR",
    "BMRI",
    "BNGA",
    "BNLI",
    "BTPS",
    "PNBN"
]

# urutan fitur dipertahankan saat model dilatih dan dibaca kembali
feature_columns = [
    "SMA10",
    "SMA20",
    "EMA10",
    "RSI14",
    "MACD",
    "Signal",
    "BB_Upper",
    "BB_Lower",
    "Stochastic_%K",
    "ATR14"
]

# semua model memakai return penutupan ke hari bursa berikutnya
target = 'target_next_return'

# nama lengkap bank buat judul tabel dan grafik
ticker_names = {
    "BBCA": "Bank Central Asia Tbk",
    "BBNI": "Bank Negara Indonesia Tbk",
    "BBRI": "Bank Rakyat Indonesia Tbk",
    "BBTN": "Bank Tabungan Negara Tbk",
    "BDMN": "Bank Danamon Indonesia Tbk",
    "BJBR": "Bank Pembangunan Daerah Jawa Barat dan Banten Tbk",
    "BMRI": "Bank Mandiri Tbk",
    "BNGA": "Bank CIMB Niaga Tbk",
    "BNLI": "Bank Permata Tbk",
    "BTPS": "Bank BTPN Syariah Tbk",
    "PNBN": "Bank Pan Indonesia Tbk"
}

# tiap ticker memakai warna yang sama pada grafik lintas notebook
palette = {
    "BBCA": "#1F4E78",
    "BBNI": "#70AD47",
    "BBRI": "#2E75B6",
    "BBTN": "#FFC000",
    "BDMN": "#C00000",
    "BJBR": "#8C564B",
    "BMRI": "#ED7D31",
    "BNGA": "#7030A0",
    "BNLI": "#17A2A4",
    "BTPS": "#D65F8D",
    "PNBN": "#5B6C8F"
}

# seed yang sama menjaga pengacakan dapat diulang
random_seed = 42

# batas kuartal dipakai bersama oleh pemodelan dan interpretasi
quarters_2025 = [
    {
        "name": "Q1 2025",
        "code": "Q1_2025",
        "start": "2025-01-01",
        "end": "2025-03-31"
    },
    {
        "name": "Q2 2025",
        "code": "Q2_2025",
        "start": "2025-04-01",
        "end": "2025-06-30"
    },
    {
        "name": "Q3 2025",
        "code": "Q3_2025",
        "start": "2025-07-01",
        "end": "2025-09-30"
    },
    {
        "name": "Q4 2025",
        "code": "Q4_2025",
        "start": "2025-10-01",
        "end": "2025-12-31"
    }
]

# warna dasar grafik
blue = '#1F4E78'

orange = '#ED7D31'

green = '#2E75B6'

light = '#BDD7EE'

red = '#C00000'

quarter_codes = {quarter["name"]: quarter["code"] for quarter in quarters_2025}


def configure_notebook():
    """Mengatur format tabel dan menentukan folder proyek dari lokasi modul."""
    pd.set_option("display.max_rows", 15)
    pd.set_option("display.max_columns", 16)
    pd.set_option("display.float_format", "{:,.6g}".format)
    root_dir = Path(__file__).resolve().parent
    if not (root_dir / "data").is_dir():
        raise FileNotFoundError("Folder data penelitian belum tersedia")
    return root_dir, root_dir / "output"
