"""Mengecek file Ringkasan Saham IDX yang di-upload sebelum dihitung indikatornya atau dipakai training."""
from datetime import datetime
from io import BytesIO
import re
from zoneinfo import ZoneInfo

import importlib.util

import pandas as pd

# python-calamine membaca Excel sekitar 6 kali lebih cepat dari openpyxl dengan hasil yang sama
# (sudah dicek pada 242 file IDX 2020 sampai 2025). Kalau belum terpasang, pandas memakai openpyxl.
EXCEL_ENGINE = "calamine" if importlib.util.find_spec("python_calamine") else None

# tambahan " (1)", " (2)" dari browser saat file terunduh dua kali tetap diterima
FILENAME_PATTERN = re.compile(r"Ringkasan\s+Saham-(\d{4})(\d{2})(\d{2})(?: \(\d+\))?\.xlsx", re.IGNORECASE)


class UploadValidationError(ValueError):
    """Pesan error upload yang aman ditampilkan langsung ke pengguna."""


# nama bulan Indonesia atau Inggris, singkat atau lengkap, diubah jadi nomor bulan
# file IDX menyingkat Agustus menjadi "Agt", misalnya "31 Agt 2026"
IDX_MONTHS = {
    "jan": "01", "januari": "01", "feb": "02", "februari": "02",
    "mar": "03", "maret": "03", "apr": "04", "april": "04",
    "mei": "05", "may": "05", "jun": "06", "juni": "06",
    "jul": "07", "juli": "07", "agu": "08", "agt": "08", "agustus": "08",
    "aug": "08", "sep": "09", "september": "09", "okt": "10",
    "oktober": "10", "oct": "10", "nov": "11", "november": "11",
    "des": "12", "desember": "12", "dec": "12",
}


def parse_idx_trade_dates(values):
    """Membaca tanggal bahasa Indonesia dari file IDX, misalnya "02 Jan 2026" atau "02 Januari 2026"."""
    month_pattern = re.compile(r"\b(" + "|".join(sorted(IDX_MONTHS, key=len, reverse=True)) + r")\b", re.IGNORECASE)

    def normalize(value):
        if not isinstance(value, str):
            return value
        return month_pattern.sub(lambda match: IDX_MONTHS[match.group().lower()], value.strip())

    return pd.to_datetime(values.map(normalize), errors="coerce", dayfirst=True, format="mixed")


def filename_date(filename, today=None):
    """Tanggal dari nama file IDX, misalnya Ringkasan Saham-20260102.xlsx menjadi 2 Januari 2026.

    Dicek sebelum isi Excel dibuka, jadi nama file yang salah langsung ketahuan tanpa membaca isinya.
    """
    date_match = FILENAME_PATTERN.fullmatch(filename)
    if not date_match:
        raise UploadValidationError(
            "Nama file harus sama dengan nama asli dari IDX, misalnya Ringkasan Saham-20260102.xlsx."
        )
    try:
        snapshot = pd.Timestamp(*map(int, date_match.groups()))
    except ValueError as error:
        raise UploadValidationError("Tanggal pada nama file tidak valid. Format yang benar adalah YYYYMMDD.") from error
    limit = pd.Timestamp(today or datetime.now(ZoneInfo("Asia/Jakarta")).date()).normalize()
    if snapshot > limit:
        raise UploadValidationError("Tanggal pada nama file melewati hari ini (WIB). Periksa kembali nama file.")
    return snapshot


def validate_market_upload(filename, content, required_columns, requested_tickers, today=None):
    """Mengecek satu file Ringkasan Saham IDX, hasilnya (tanggal file, tabel saham bank).

    Yang dicek: file Excel bisa dibaca, jumlah baris wajar, kolom wajib lengkap, ada minimal satu
    dari 11 saham, kode saham tidak dobel, nama file sesuai format IDX (Ringkasan Saham-YYYYMMDD.xlsx),
    dan tanggalnya tidak lewat hari ini (WIB). Kalau gagal, muncul UploadValidationError dengan pesan yang jelas.
    """
    try:
        frame = pd.read_excel(BytesIO(content), engine=EXCEL_ENGINE)
    except Exception as error:
        raise UploadValidationError(
            "File Excel tidak dapat dibaca. Pastikan file tidak rusak, tidak dikunci kata sandi, "
            "dan merupakan Ringkasan Saham asli dari IDX."
        ) from error
    if len(frame) > 10_000:
        raise UploadValidationError("File memuat lebih dari 10.000 baris. Gunakan file Ringkasan Saham harian dari IDX.")
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise UploadValidationError(
            "Kolom wajib belum lengkap: " + ", ".join(missing)
            + ". Gunakan file Ringkasan Saham asli dari IDX tanpa mengubah nama kolom."
        )
    if frame.columns.duplicated().any():
        raise UploadValidationError("Terdapat nama kolom yang sama lebih dari satu kali. Gunakan file asli dari IDX.")

    frame["Kode Saham"] = frame["Kode Saham"].astype(str).str.strip().str.upper()
    matched = frame["Kode Saham"].isin(requested_tickers)
    if not matched.any():
        raise UploadValidationError(
            "File tidak memuat satu pun saham yang didukung aplikasi (" + ", ".join(requested_tickers) + ")."
        )

    # file IDX isinya semua saham di bursa, yang diproses cuma 11 saham bank penelitian
    frame = frame.loc[matched].copy()
    duplicated_codes = sorted(frame.loc[frame["Kode Saham"].duplicated(), "Kode Saham"].unique())
    if duplicated_codes:
        raise UploadValidationError(
            "Kode saham muncul lebih dari satu kali dalam file: " + ", ".join(duplicated_codes) + "."
        )
    snapshot = filename_date(filename, today)
    limit = pd.Timestamp(today or datetime.now(ZoneInfo("Asia/Jakarta")).date()).normalize()
    trades = parse_idx_trade_dates(frame["Tanggal Perdagangan Terakhir"])
    if trades.isna().any():
        raise UploadValidationError(
            "Kolom Tanggal Perdagangan Terakhir kosong atau tidak terbaca untuk saham yang didukung."
        )
    if (trades.dt.normalize() > limit).any():
        raise UploadValidationError("Tanggal perdagangan terakhir tidak boleh melewati hari ini (WIB).")
    if (trades.dt.normalize() > snapshot).any():
        raise UploadValidationError(
            "Tanggal Perdagangan Terakhir melewati tanggal pada nama file. Pastikan nama file sesuai tanggal datanya."
        )
    frame["Tanggal Perdagangan Terakhir"] = trades
    return snapshot, frame
