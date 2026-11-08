"""Pengaturan bersama: daftar saham, fitur, dan target (dipakai notebook dan aplikasi), plus warna grafik dan kuartal 2025 untuk notebook."""
from pathlib import Path

import pandas as pd

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

target = 'target_next_return'

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

random_seed = 42

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

blue = '#1F4E78'

orange = '#ED7D31'

quarter_codes = {quarter["name"]: quarter["code"] for quarter in quarters_2025}


def configure_notebook():
    """Mengatur format tabel dan menentukan folder proyek dari lokasi modul."""
    pd.set_option("display.max_rows", 15)
    pd.set_option("display.max_columns", 16)
    pd.set_option("display.float_format", "{:,.6g}".format)
    root_dir = Path(__file__).resolve().parent
    return root_dir, root_dir / "output"
