# Library
import base64
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
from io import BytesIO, StringIO
import json
import logging
import os
import re
from pathlib import Path
import sys
import threading
from zipfile import ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Path Direktori
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Function
from application.modeling import (
    build_portfolio_allocation,
    calculate_features,
    read_baseline_bundle,
    train_baseline_models,
)
from application.access_sheet import (
    APPROVAL_HEADERS,
    TRIAL_COLUMN,
    add_user,
    claim_trial,
    clean_records,
    ensure_headers,
    find_row,
    request_access,
    role_of,
    set_role,
    trial_window,
)
from application.baseline_builder import append_idx_rows, build_baseline_bundle
from application.github_store import GitHubStoreError, github_is_configured, github_settings, publish_bundle
from notebook_config import feature_columns as FEATURES, tickers as PORTFOLIO_TICKERS
from pipeline_utils import optimize_max_sharpe
from upload_validation import UploadValidationError, filename_date, validate_market_upload

# Konfigurasi Path out dan untuk ambil data hasil preprocessing
OUTPUT_DIR = BASE_DIR / "output" / "research"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# File hasil notebook
RESEARCH_SOURCES = {
    "predictions_2025.parquet": PROCESSED_DIR / "predictions_2025.parquet",
    "panel_calendar.parquet": PROCESSED_DIR / "panel_calendar.parquet",
}


def research_path(filename):
    """Lokasi file hasil notebook untuk ditampilin di Hasil Pengujian."""
    return RESEARCH_SOURCES.get(filename, OUTPUT_DIR / filename)


DEFAULT_BASELINE_PATH = "runtime/baseline_bundle.zip"
MAX_UPLOAD_FILES = 30
MAX_UPLOAD_TOTAL_BYTES = 30 * 1024 * 1024
MAX_UPLOAD_UNCOMPRESSED_BYTES = 30 * 1024 * 1024
# batas upload admin untuk update model bawaan
MAX_ADMIN_UPLOAD_FILES = 260
MAX_ADMIN_UPLOAD_TOTAL_BYTES = 300 * 1024 * 1024
IDX_REQUIRED_COLUMNS = [
    "Kode Saham", "Remarks", "Open Price", "Tanggal Perdagangan Terakhir",
    "Tertinggi", "Terendah", "Penutupan", "Volume", "Listed Shares",
]
# kode Remarks IDX
CORPORATE_ACTION_CODES = {
    "--": "Tidak ada aksi korporasi",
    "CD": "Cum Dividend",
    "XD": "Ex Dividend",
    "CR": "Cum Rights",
    "XR": "Ex Rights",
    "SS": "Stock Split",
    "RS": "Reverse Stock Split",
    "CA": "Cum Combined Corporate Action",
    "XA": "Ex Combined Corporate Action",
    "CB": "Cum Bonus Shares",
    "XB": "Ex Bonus Shares",
}
EXPLICIT_SPLIT_CODES = {"SS", "RS"}
INFERRED_SPLIT_CODES = {"CA", "XA", "CB", "XB"}
# keterangan indikator di aplikasi
FEATURE_HELP = {
    "SMA10": "Rata-rata harga penutupan 10 hari terakhir. Dipakai untuk melihat arah tren jangka pendek.",
    "SMA20": "Rata-rata harga penutupan 20 hari terakhir. Dipakai untuk melihat arah tren yang lebih panjang.",
    "EMA10": "Mirip SMA10, tapi harga hari-hari terakhir memiliki bobot lebih tinggi. Jadi lebih cepat menangkap perubahan arah harga.",
    "RSI14": "Ukuran 0 sampai 100 untuk kekuatan kenaikan harga 14 hari terakhir. Di atas 70 berarti harga sudah naik banyak dalam waktu singkat. Di bawah 30 berarti harga sudah turun banyak.",
    "MACD": "Selisih rata-rata harga 12 dan 26 hari. Positif berarti harga jangka pendek sedang di atas jangka panjang, tanda harga sedang naik.",
    "Signal": "Rata-rata MACD 9 hari terakhir. Kalau MACD di atas Signal, kenaikan harga sedang menguat. Kalau di bawah, sedang melemah.",
    "BB_Upper": "Batas atas kisaran harga yang biasa terjadi. Dihitung dari rata-rata harga 20 hari ditambah dua kali besar naik turunnya harga. Harga di dekat batas ini berarti sedang tinggi dibanding biasanya.",
    "BB_Lower": "Batas bawah kisaran harga yang biasa terjadi. Dihitung dari rata-rata harga 20 hari dikurangi dua kali besar naik turunnya harga. Harga di dekat batas ini berarti sedang rendah dibanding biasanya.",
    "Stochastic_%K": "Posisi harga penutupan hari ini di antara harga terendah dan tertinggi 14 hari terakhir. 100 berarti tutup di harga tertinggi, 0 berarti di harga terendah.",
    "ATR14": "Rata-rata besar gerak harga harian dalam 14 hari terakhir. Makin besar, harganya makin bergejolak.",
}

FEATURE_LABELS = {
    "SMA10": "SMA10",
    "SMA20": "SMA20",
    "EMA10": "EMA10",
    "RSI14": "RSI14",
    "MACD": "MACD",
    "Signal": "Signal",
    "BB_Upper": "BB Upper",
    "BB_Lower": "BB Lower",
    "Stochastic_%K": "Stochastic %K",
    "ATR14": "ATR14",
}

INDICATOR_HELP = ("Indikator teknikal adalah alat bantu untuk menganalisis pergerakan "
                  "harga saham berdasarkan data historis perdagangan. Indikator ini "
                  "dihitung menggunakan rumus tertentu yang dapat melibatkan harga "
                  "pembukaan, tertinggi, terendah, penutupan, maupun volume perdagangan. "
                  )

# Secrets, login Google, dan izin akses lewat Google Sheet
def get_secret_section(name):
    """Mengambil satu bagian dari Streamlit Secrets, misalnya [auth]. Kosong kalau belum diisi."""
    try:
        return dict(st.secrets.get(name, {}))
    except FileNotFoundError:
        return {}


def auth_is_configured():
    """Cek pengaturan login Google [auth] sudah lengkap atau belum."""
    auth = get_secret_section("auth")
    required = ["redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url"]
    return all(auth.get(key) for key in required)


def development_settings():
    """Pengaturan login lokal buat mode DEV."""
    if os.environ.get("SOLYN_ALLOW_LOCAL_LOGIN") != "1":
        return {"enabled": False}
    return get_secret_section("development")


def approval_settings():
    """Pengaturan [approval]: email admin, Google Sheet daftar pengguna, dan service account."""
    settings = get_secret_section("approval")
    settings["service_account"] = dict(settings.get("service_account", {}))
    return settings


def approval_is_configured():
    """Cek permission Google Sheet"""
    settings = approval_settings()
    required = ["admin_email", "spreadsheet_id", "worksheet"]
    credentials = settings.get("service_account", {})
    credential_fields = ["project_id", "private_key", "client_email", "token_uri"]
    return all(settings.get(key) for key in required) and all(credentials.get(key) for key in credential_fields)


@st.cache_resource(show_spinner=False)
def get_approval_worksheet():
    """Membuka database pengguna"""
    import gspread

    settings = approval_settings()
    client = gspread.service_account_from_dict(settings["service_account"])
    spreadsheet = client.open_by_key(str(settings["spreadsheet_id"]))
    worksheet_name = str(settings.get("worksheet", "Users"))
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=1000,
            cols=len(APPROVAL_HEADERS),
        )
    # menambah kolom yang belum ada
    ensure_headers(worksheet)
    return worksheet


@st.cache_resource(show_spinner=False)
def training_slot():
    """Membatasi biar cuma satu training model sementara yang jalan dalam satu server."""
    return threading.BoundedSemaphore(1)


@st.cache_resource(show_spinner=False)
def approval_request_lock():
    """Mengunci biar dua permintaan akses tidak menulis ke Google Sheet barengan."""
    return threading.Lock()


@st.cache_data(ttl=15, show_spinner=False)
def load_approval_records():
    """Membaca semua baris daftar pengguna, di-cache 15 detik."""
    return clean_records(get_approval_worksheet())


def find_approval_record(email):
    """Mencari data izin untuk satu email, None kalau belum terdaftar."""
    return find_row(load_approval_records(), email)[1]


def create_approval_request(identity):
    """Mengajukan akses ke admin (status Pending) di Google Sheet."""
    actor = current_identity()
    if not actor or actor.get("local") or actor["email"] != identity.get("email"):
        raise PermissionError("Permintaan akses memerlukan akun Google yang aktif.")
    with approval_request_lock():
        request_access(get_approval_worksheet(), actor["email"], actor["name"], wib_now())
    load_approval_records.clear()


def wib_now():
    """Waktu sekarang dalam WIB, dipakai untuk semua catatan di Google Sheet."""
    return datetime.now(timezone(timedelta(hours=7)))


def review_approval_request(email, status):
    """Admin approve atau tolak permintaan akses."""
    actor = current_identity()
    if not actor or actor.get("local") or not actor.get("is_admin"):
        raise PermissionError("Tindakan ini hanya tersedia untuk administrator Google.")
    worksheet = get_approval_worksheet()
    row_index, record = find_row(clean_records(worksheet), email)
    if record is None:
        raise ValueError("Akun tidak ditemukan.")
    target = record["Email"].lower()
    if status == "Rejected" and (target in owner_emails() or target == actor["email"]):
        raise ValueError("Akses pemilik aplikasi dan akun Anda sendiri tidak dapat dicabut.")
    changed_at = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    worksheet.update(
        [[status, changed_at]],
        range_name=f"D{row_index + 2}:E{row_index + 2}",
        value_input_option="RAW",
    )
    if status == "Rejected" and role_of(record) == "Admin":
        # Update status akun yang di reject ke role user
        worksheet.update([["User"]], range_name=f"G{row_index + 2}", value_input_option="RAW")
    load_approval_records.clear()


def change_user_role(email, role):
    """Admin mengubah role akun lain menjadi Admin atau User."""
    actor = current_identity()
    if not actor or actor.get("local") or not actor.get("is_admin"):
        raise PermissionError("Tindakan ini hanya tersedia untuk administrator Google.")
    target = str(email).strip().lower()
    if role == "User" and (target in owner_emails() or target == actor["email"]):
        raise ValueError("Role admin pemilik aplikasi dan akun Anda sendiri tidak dapat dicabut.")
    set_role(get_approval_worksheet(), target, role, wib_now())
    load_approval_records.clear()


def register_user(email, name, role):
    """Admin mendaftarkan email baru yang langsung disetujui, dengan role User atau Admin."""
    actor = current_identity()
    if not actor or actor.get("local") or not actor.get("is_admin"):
        raise PermissionError("Tindakan ini hanya tersedia untuk administrator Google.")
    email = str(email).strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("Format email salah, contoh: budi@gmail.com.")
    name = str(name).strip() or email.split("@")[0]
    add_user(get_approval_worksheet(), email, name, role, wib_now())
    load_approval_records.clear()


# Membaca dan Cek file Ringkasan Saham IDX yang di-upload
def parse_market_file(filename, content, repair_prices=True):
    """Membaca satu file Ringkasan Saham IDX jadi tabel harga 11 saham bank.

    Args:
        filename: Nama file asli dari IDX, untuk membaca tanggal data.
        content: Isi file .xlsx dalam bytes.
        repair_prices: Jika True, Open, High, dan Low yang bernilai 0 diisi dari
            Close. Upload admin memakai False, dan baris suspensinya diperbaiki
            oleh repair_suspension_rows saat training.

    Raises:
        UploadValidationError: Jika struktur, isi, atau nilai harga file tidak valid.
    """
    try:
        with ZipFile(BytesIO(content)) as workbook:
            members = workbook.infolist()
            if len(members) > 500 or sum(item.file_size for item in members) > MAX_UPLOAD_UNCOMPRESSED_BYTES:
                raise UploadValidationError("Ukuran file Excel terlalu besar.")
    except UploadValidationError:
        raise
    except Exception as error:
        raise UploadValidationError("File tidak valid.") from error
    snapshot_date, frame = validate_market_upload(filename, content, IDX_REQUIRED_COLUMNS, PORTFOLIO_TICKERS)

    frame = frame.rename(columns={
        "Kode Saham": "Ticker",
        "Open Price": "Open",
        "Tertinggi": "High",
        "Terendah": "Low",
        "Penutupan": "Close",
    })
    frame["Date"] = snapshot_date
    frame["Remarks"] = frame["Remarks"].fillna("--").astype(str).str.strip()
    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Listed Shares"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    invalid_required = frame[["Ticker", "Close", "Volume", "Listed Shares"]].isna().any(axis=1)
    invalid_required |= (frame["Close"] <= 0) | (frame["Volume"] < 0) | (frame["Listed Shares"] <= 0)
    if invalid_required.any():
        codes = ", ".join(sorted(frame.loc[invalid_required, "Ticker"].astype(str).unique()))
        raise UploadValidationError(
            f"Penutupan, Volume, atau Listed Shares kosong atau tidak valid untuk {codes}. "
            "Penutupan dan Listed Shares harus lebih dari nol, Volume tidak boleh negatif."
        )

    if repair_prices:
        # Open/High/Low nol diisi dari Close, khusus upload pengguna
        frame["Open"] = frame["Open"].where(frame["Open"] > 0, frame["Close"])
        frame["High"] = frame["High"].where(
            frame["High"] > 0, frame[["Open", "Close"]].max(axis=1)
        )
        frame["Low"] = frame["Low"].where(
            frame["Low"] > 0, frame[["Open", "Close"]].min(axis=1)
        )
        frame["High"] = frame[["Open", "High", "Low", "Close"]].max(axis=1)
        frame["Low"] = frame[["Open", "High", "Low", "Close"]].min(axis=1)
    frame["Source File"] = filename
    return frame[[
        "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
        "Listed Shares", "Remarks", "Source File",
    ]]


@st.cache_data(show_spinner=False, max_entries=600)
def parse_market_file_cached(filename, content, repair_prices=True):
    """Menyimpan hasil baca per file agar file tidak dibaca ulang"""
    return parse_market_file(filename, content, repair_prices)


def parse_market_uploads(uploaded_files, max_files=MAX_UPLOAD_FILES, max_total_bytes=MAX_UPLOAD_TOTAL_BYTES,
                         repair_prices=True, on_progress=None, check_dates=None):
    """Membaca banyak file sekaligus lalu menggabungkan jadi satu tabel.

    Untuk tanggal yang sama hanya satu file yang dipakai, dan nama tanpa "(1)"
    dicoba lebih dulu. Pemeriksaan berhenti di tanggal pertama yang tidak punya
    file valid.

    Args:
        uploaded_files: File dari st.file_uploader.
        max_total_bytes: Batas total ukuran semua file dalam bytes.
        repair_prices: Diteruskan ke parse_market_file.
        on_progress: Fungsi opsional on_progress(selesai, total, nama_file) untuk progress bar.
        check_dates: Fungsi opsional yang menerima daftar tanggal file dan
            mengembalikan daftar pesan error, dipanggil sebelum isi file dibaca.

    Returns:
        Tuple (data, errors). Jika ada error, data bernilai None. Nama file yang
        dilewati karena tanggalnya ganda disimpan di data.attrs["skipped_files"].
    """
    if len(uploaded_files) > max_files:
        return None, [f"Anda memilih {len(uploaded_files)} file. Unggah maksimal {max_files} file sekaligus."]
    if sum(uploaded_file.size for uploaded_file in uploaded_files) > max_total_bytes:
        return None, [f"Ukuran seluruh file melebihi {max_total_bytes // (1024 * 1024)} MB. Kurangi jumlah file."]
    by_date, errors = {}, []
    for uploaded_file in uploaded_files:
        try:
            by_date.setdefault(filename_date(uploaded_file.name), []).append(uploaded_file)
        except UploadValidationError as error:
            errors.append(f"{uploaded_file.name}: {error}")
    if errors:
        return None, errors
    if check_dates:
        errors = check_dates(sorted(by_date))
        if errors:
            return None, errors
    frames, skipped, checked = [], [], 0
    total = len(uploaded_files)
    for date in sorted(by_date):
        # file tanpa "(1)" dicoba lebih dulu
        candidates = sorted(by_date[date], key=lambda item: ("(" in item.name, item.name))
        frame, first_error = None, None
        for uploaded_file in candidates:
            if frame is not None:
                skipped.append(uploaded_file.name)
                continue
            if on_progress:
                on_progress(checked, total, uploaded_file.name)
            checked += 1
            try:
                frame = parse_market_file_cached(uploaded_file.name, uploaded_file.getvalue(), repair_prices)
            except UploadValidationError as error:
                first_error = first_error or f"{uploaded_file.name}: {error}"
            except Exception:
                LOGGER.exception("Upload parsing failed")
                first_error = first_error or (
                    f"{uploaded_file.name}: File tidak dapat dibaca"
                )
        if frame is None:
            remaining = sum(len(files) for later, files in by_date.items() if later > date)
            note = [f"Pemeriksaan berhenti di file ini, jadi {remaining} file sesudahnya belum diperiksa."] if remaining else []
            return None, [first_error] + note
        frames.append(frame)
    if on_progress:
        on_progress(total, total, "")

    data = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
    data.attrs["skipped_files"] = sorted(skipped)
    return data, []


def render_upload_problems(errors, clear_key=None):
    """Masalah file ditampilkan dalam satu kotak merah dengan nama file lengkap.

    Args:
        errors: Pesan error dari parse_market_uploads dengan format "nama.xlsx: alasan".
            File dengan alasan yang sama digabung dalam satu baris.
        clear_key: Kunci session state kotak unggah. Jika diisi, muncul tombol
            untuk mengosongkan pilihan file.
    """
    grouped, general = {}, []
    for error in errors:
        name, separator, reason = str(error).partition(".xlsx: ")
        if separator:
            grouped.setdefault(reason.strip(), []).append(name + ".xlsx")
        else:
            general.append(str(error))
    lines = []
    for reason, names in grouped.items():
        labels = []
        for name in names:
            try:
                labels.append(f"{name} ({date_label(filename_date(name))})")
            except UploadValidationError:
                labels.append(name)
        lines.append(f"{', '.join(labels)}: {reason[:1].lower()}{reason[1:]}")
    lines.extend(general)
    if grouped:
        lines.append("Ganti file tersebut dengan file asli dari situs IDX. Tekan Kosongkan pilihan file, lalu pilih ulang file Anda.")
    st.error("\n\n".join(lines))
    if clear_key:
        st.button("Kosongkan pilihan file", key=f"{clear_key}_clear", on_click=clear_uploads, args=(clear_key,))


def clear_uploads(round_key):
    """Mengosongkan kotak unggah"""
    st.session_state[round_key] = st.session_state.get(round_key, 0) + 1


def render_skipped_files(data):
    """Keterangan singkat kalau ada file ganda untuk tanggal yang sama yang dilewati."""
    skipped = data.attrs.get("skipped_files", [])
    if skipped:
        st.info(f"{len(skipped)} file ganda untuk tanggal yang sama tidak dipakai: " + ", ".join(skipped) + ".")


class CorporateActionSafetyError(ValueError):
    """Error kalau data upload punya aksi korporasi atau perubahan harga yang tidak bisa disesuaikan dengan aman."""


def prepare_market_data(data):
    """Menyesuaikan harga data gabungan (bawaan dan upload) kalau ada stock split."""
    prepared_groups = []
    source = data.copy()

    for ticker, ticker_data in source.groupby("Ticker"):
        frame = ticker_data.sort_values("Date").copy()
        frame["Remarks"] = frame["Remarks"].fillna("--").astype(str).str.strip()
        frame["ca_code"] = frame["Remarks"].str[:2].str.upper()
        unknown_codes = sorted(set(frame["ca_code"]) - set(CORPORATE_ACTION_CODES))
        if unknown_codes:
            unknown_rows = frame[frame["ca_code"].isin(unknown_codes)]
            details = []
            for _, row in unknown_rows.iterrows():
                details.append(f"{ticker} {pd.Timestamp(row['Date']):%Y-%m-%d} kode {row['ca_code']}")
            raise CorporateActionSafetyError(
                "Kode Remarks belum dikenali: " + ", ".join(details) + ". Retraining dihentikan untuk mencegah penyesuaian harga yang salah."
            )

        frame["previous_close"] = frame["Close"].shift(1)
        frame["previous_volume"] = frame["Volume"].shift(1)
        frame["previous_listed_shares"] = frame["Listed Shares"].shift(1)
        frame["shares_ratio"] = frame["Listed Shares"] / frame["previous_listed_shares"]
        frame["raw_return"] = frame["Close"] / frame["previous_close"] - 1
        frame["volume_ratio"] = frame["Volume"] / frame["previous_volume"].replace(0, np.nan)
        frame["basis_neutral_return"] = (
            frame["Close"] * frame["shares_ratio"] / frame["previous_close"] - 1
        )
        material_change = (frame["shares_ratio"] - 1).abs() > 0.01
        price_anomaly = frame["raw_return"].abs() > 0.35
        volume_price_anomaly = (frame["volume_ratio"] > 10) & (frame["raw_return"].abs() > 0.20)

        explicit_split_ok = (
            ((frame["ca_code"] == "SS") & (frame["shares_ratio"] > 1.01))
            | ((frame["ca_code"] == "RS") & (frame["shares_ratio"] < 0.99))
        )
        explicit_split = frame["ca_code"].isin(EXPLICIT_SPLIT_CODES) & material_change
        invalid_explicit = explicit_split & ~explicit_split_ok
        inferred_candidate = frame["ca_code"].isin(INFERRED_SPLIT_CODES) & material_change
        inferred_split = inferred_candidate & (frame["basis_neutral_return"].abs() <= 0.20)
        # anomali hanya dicek di baris upload
        uploaded_row = frame["Source File"].notna()
        unexplained_change = (
            (frame["ca_code"] == "--")
            & (material_change | price_anomaly | volume_price_anomaly)
            & uploaded_row
        )
        unsafe_rows = frame[invalid_explicit | (inferred_candidate & ~inferred_split) | unexplained_change]
        if not unsafe_rows.empty:
            details = []
            for _, row in unsafe_rows.iterrows():
                price_change = row["raw_return"] * 100 if pd.notna(row["raw_return"]) else np.nan
                volume_change = row["volume_ratio"] if pd.notna(row["volume_ratio"]) else np.nan
                details.append(
                    f"{ticker} {pd.Timestamp(row['Date']):%Y-%m-%d}, kode {row['ca_code']}, "
                    f"perubahan harga {price_change:+.1f}%, rasio volume {volume_change:.1f}x"
                )
            raise CorporateActionSafetyError(
                "Terjadi perubahan yang belum dapat diklasifikasikan dengan aman: "
                + ", ".join(details)
                + ". Retraining dihentikan agar model tidak mempelajari data yang salah."
            )

        auto_adjust = (explicit_split & explicit_split_ok) | inferred_split
        event_factor = np.where(auto_adjust, 1 / frame["shares_ratio"], 1.0)
        adjustment_factor = np.ones(len(frame))
        accumulated_factor = 1.0
        for index in range(len(frame) - 1, -1, -1):
            adjustment_factor[index] = accumulated_factor
            accumulated_factor *= event_factor[index]

        frame["price_adjustment_factor"] = adjustment_factor
        frame["Open"] = frame["Open"] * adjustment_factor
        frame["High"] = frame["High"] * adjustment_factor
        frame["Low"] = frame["Low"] * adjustment_factor
        frame["Close"] = frame["Close"] * adjustment_factor
        frame["Volume"] = frame["Volume"] / adjustment_factor
        prepared_groups.append(frame)

    prepared = pd.concat(prepared_groups, ignore_index=True).sort_values(["Ticker", "Date"])
    return prepared.reset_index(drop=True)


# Format tanggal dan grafik halaman Hasil Pengujian
def format_date_id(value):
    """Format tanggal ke bahasa Indonesia, contohnya 30 Desember 2025."""
    date = pd.Timestamp(value)
    month_names = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember",
    ]
    return f"{date.day} {month_names[date.month - 1]} {date.year}"


# warna grafik, disamakan dengan warna di solyn.css
CHART_CANVAS = "#F7F9FC"
CHART_INK = "#162235"
CHART_TEXT = "#475569"
CHART_GRID = "#E2E8F0"
CHART_ZEROLINE = "#CBD5E1"
CHART_REFERENCE = "#94A3B8"
CHART_GREEN = "#176342"
CHART_RED = "#A4262C"
CHART_MODEL = "#0072B2"
CHART_MODEL_FILL = "rgba(0,114,178,.10)"
CHART_BENCHMARK = "#E69F00"
CHART_ACTUAL = "#0072B2"
CHART_ACTUAL_BELOW = "#B45F06"
CHART_FORECAST = "#A7372E"
CHART_FONT = "Segoe UI, Arial, sans-serif"
CHART_CONFIG = {"displayModeBar": False}


def oos_chart_layout(height, yaxis_title):
    """Layout bersama untuk grafik hasil pengujian 2025."""
    return {
        "height": height,
        "margin": {"l": 12, "r": 12, "t": 28, "b": 12},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.14, "x": 0},
        "xaxis": {"showgrid": False},
        "yaxis": {"showgrid": True, "title": yaxis_title},
    }


def build_oos_charts(daily, quarterly):
    """Membuat 3 grafik halaman Hasil Pengujian: return kumulatif, drawdown, dan return per kuartal."""
    model_quarterly = quarterly["Realized Return XGBoost-Markowitz (%)"]
    benchmark_quarterly = quarterly["Realized Return Historical Mean (%)"]
    cumulative = go.Figure()
    cumulative.add_trace(go.Scatter(
        x=daily["realized_date"],
        y=daily["cum_xgb"] * 100,
        name="Portofolio Model",
        line={"color": CHART_MODEL, "width": 2.4},
        hovertemplate="%{y:+.2f}%<extra></extra>",
    ))
    cumulative.add_trace(go.Scatter(
        x=daily["realized_date"],
        y=daily["cum_hist"] * 100,
        name="Portofolio Pembanding",
        line={"color": CHART_BENCHMARK, "width": 2, "dash": "dash"},
        hovertemplate="%{y:+.2f}%<extra></extra>",
    ))
    cumulative.add_hline(y=0, line_dash="dot", line_color=CHART_REFERENCE)
    cumulative.update_layout(**oos_chart_layout(360, "Return kumulatif (%)"))

    model_wealth = (1 + daily["return_xgb"]).cumprod()
    benchmark_wealth = (1 + daily["return_hist"]).cumprod()
    model_drawdown = (model_wealth / model_wealth.cummax().clip(lower=1) - 1) * 100
    benchmark_drawdown = (benchmark_wealth / benchmark_wealth.cummax().clip(lower=1) - 1) * 100
    drawdown = go.Figure()
    drawdown.add_trace(go.Scatter(
        x=daily["realized_date"],
        y=model_drawdown,
        name="Portofolio Model",
        fill="tozeroy",
        fillcolor=CHART_MODEL_FILL,
        line={"color": CHART_MODEL, "width": 1.5},
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    drawdown.add_trace(go.Scatter(
        x=daily["realized_date"],
        y=benchmark_drawdown,
        name="Portofolio Pembanding",
        line={"color": CHART_BENCHMARK, "width": 1.7, "dash": "dash"},
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    drawdown.update_layout(**oos_chart_layout(310, "Drawdown (%)"))

    quarterly_chart = go.Figure()
    quarterly_chart.add_trace(go.Bar(
        x=quarterly["Kuartal"],
        y=model_quarterly,
        name="Portofolio Model",
        marker_color=CHART_MODEL,
        text=[f"{value:+.2f}%" for value in model_quarterly],
        texttemplate="%{text}",
        textposition="outside",
        textfont={"size": 16},
        cliponaxis=False,
        hovertemplate="%{x}<br>%{y:+.2f}%<extra></extra>",
    ))
    quarterly_chart.add_trace(go.Bar(
        x=quarterly["Kuartal"],
        y=benchmark_quarterly,
        name="Portofolio Pembanding",
        marker_color=CHART_BENCHMARK,
        text=[f"{value:+.2f}%" for value in benchmark_quarterly],
        texttemplate="%{text}",
        textposition="outside",
        textfont={"size": 16},
        cliponaxis=False,
        hovertemplate="%{x}<br>%{y:+.2f}%<extra></extra>",
    ))
    quarterly_chart.update_layout(**{
        **oos_chart_layout(400, "Return per kuartal (%)"),
        "barmode": "group",
        "margin": {"l": 18, "r": 18, "t": 48, "b": 20},
        "legend": {"orientation": "h", "y": 1.16, "x": 0, "font": {"size": 14}},
    })
    quarterly_chart.update_xaxes(tickfont={"size": 14})
    quarterly_chart.update_yaxes(tickfont={"size": 13}, title_font={"size": 15})
    return cumulative, drawdown, quarterly_chart


# Tampilan: judul tab, logo, CSS, nama bank, daftar halaman
LOGGER = logging.getLogger(__name__)
LOGO_PATH = BASE_DIR / "assets" / "logo_transparent.png"
ICON_PATH = BASE_DIR / "assets" / "logo_only.png"
st.set_page_config(page_title="Solyn",
                   page_icon=str(ICON_PATH),
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>" + (BASE_DIR / "assets" / "solyn.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

BANK_NAMES = {
    "BBCA":"Bank Central Asia", "BBNI":"Bank Negara Indonesia",
    "BBRI":"Bank Rakyat Indonesia", "BBTN":"Bank Tabungan Negara",
    "BDMN":"Bank Danamon Indonesia", "BJBR":"Bank Pembangunan Daerah Jawa Barat dan Banten",
    "BMRI":"Bank Mandiri", "BNGA":"Bank CIMB Niaga", "BNLI":"Bank Permata",
    "BTPS":"Bank BTPN Syariah", "PNBN":"Bank Pan Indonesia",
}
AUTHOR = {
    "nama": "James Anderson",
    "nim": "535230065",
    "fakultas": "Fakultas Teknologi Informasi",
    "universitas": "Universitas Tarumanagara",
}
SUPERVISORS = [
    "Tony, S.Kom., M.Kom., Ph. D. (1080005)",
    "Irvan Lewenusa, S.Kom., M.Kom. (10823004)",
]
# kode di URL (?view=...) dan judul halaman
VIEWS = {"home":"Home", "analysis":"Prediksi Saham", "portfolio":"Simulasi Portofolio",
         "performance":"Hasil Pengujian", "upload":"Unggah Data", "about":"Tentang",
         "admin":"Panel Admin"}


# Login, coba gratis 5 menit, dan status akses
# tambahan di atas umur token Google 1 jam, total sesi 12 jam
SESSION_EXTRA_SECONDS = 11 * 60 * 60

TRIAL_SESSION_KEY = "v3_trial"
TRIAL_NOTICE_KEY = "v3_trial_notice"
APPROVED_NOTICE_KEY = "v3_approved_notice"


def trial_settings():
    """Pengaturan coba gratis. Defaultnya aktif 5 menit"""
    settings = get_secret_section("trial")
    try:
        minutes = float(settings.get("minutes", 5))
    except (TypeError, ValueError):
        minutes = 5.0
    return {"enabled": bool(settings.get("enabled", True)), "minutes": max(1.0, minutes)}


def trial_available():
    """Coba gratis butuh Google Sheet, karena pemakaian tiap email dicatat di sana."""
    return trial_settings()["enabled"] and approval_is_configured()


def trial_record_window(email):
    """Waktu mulai dan selesai coba gratis milik satu email, dibaca dari Google Sheet."""
    if not trial_available():
        return None
    try:
        return trial_window(find_approval_record(email), trial_settings()["minutes"])
    except Exception:
        LOGGER.exception("Trial status unavailable")
        return None


def owner_emails():
    """Email pemilik aplikasi dari Secrets [approval]. Role admin pemilik tidak bisa dicabut lewat aplikasi."""
    email = str(get_secret_section("approval").get("admin_email", "")).strip().lower()
    return {email} if email else set()


def admin_in_sheet(email):
    """Akun dengan Role Admin dan status Approved di Google Sheet juga dianggap admin."""
    if not approval_is_configured():
        return False
    try:
        record = find_approval_record(email)
    except Exception:
        LOGGER.exception("Role status unavailable")
        return False
    return bool(record) and role_of(record) == "Admin" and record.get("Status", "").strip().lower() == "approved"


def approved_in_sheet(email):
    """Cek apakah admin sudah menyetujui email ini di Google Sheet."""
    try:
        record = find_approval_record(email)
    except Exception:
        LOGGER.exception("Approval status unavailable")
        return False
    return bool(record) and record.get("Status", "").strip().lower() == "approved"


def trial_state():
    """Data coba gratis yang sedang jalan untuk pengguna ini, atau None."""
    identity = current_identity()
    if not identity or not identity.get("trial"):
        return None
    return {"expires_at": identity["trial_expires_at"]}


def trial_seconds_left(trial):
    """Sisa waktu coba gratis dalam detik."""
    return max(0, int(trial["expires_at"] - datetime.now(timezone.utc).timestamp()))


def start_trial(identity, destination="home"):
    """Memulai coba gratis untuk akun Google ini. Satu email cuma bisa sekali.

    Args:
        identity: Akun yang sedang login, berisi email dan name.
        destination: Halaman tujuan setelah coba gratis dimulai. Halaman admin
            atau nama yang tidak dikenal diarahkan ke home.

    Returns:
        "started" jika berhasil, "used" jika email ini sudah pernah memakai coba
        gratis, "rejected" jika aksesnya pernah ditolak, atau "unavailable" jika
        gagal.
    """
    try:
        with approval_request_lock():
            worksheet = get_approval_worksheet()
            result = claim_trial(worksheet, identity["email"], identity["name"], wib_now())
            if result == "started":
                request_access(worksheet, identity["email"], identity["name"], wib_now())
    except Exception:
        LOGGER.exception("Trial claim failed")
        return "unavailable"
    load_approval_records.clear()
    if result == "started":
        st.session_state.pop(TRIAL_NOTICE_KEY, None)
        st.query_params["view"] = destination if destination in VIEWS and destination != "admin" else "home"
    return result


def end_trial(message):
    """Menghapus semua data sesi coba gratis, termasuk upload dan model sementara."""
    for key in list(st.session_state):
        del st.session_state[key]
    st.query_params.clear()
    st.session_state[TRIAL_NOTICE_KEY] = message


def enforce_trial_expiry():
    """Dicek tiap rerun. Kalau waktu coba gratis habis, data sesi dihapus dan pengguna diberi tahu."""
    identity = current_identity()
    if identity and identity.get("trial"):
        st.session_state[TRIAL_SESSION_KEY] = identity["trial_expires_at"]
        return False
    if st.session_state.get(TRIAL_SESSION_KEY) is not None:
        if identity and access_status(identity) == "approved":
            # disetujui admin di tengah coba gratis
            st.session_state.pop(TRIAL_SESSION_KEY, None)
            st.session_state[APPROVED_NOTICE_KEY] = True
            return False
        minutes = trial_settings()["minutes"]
        end_trial(f"Masa coba gratis {minutes:g} menit sudah habis.")
        return True
    return False


def current_identity():
    """Siapa pengguna sekarang: akun Google, coba gratis, login lokal, atau None kalau tamu."""
    dev = development_settings()
    if getattr(st.user, "is_logged_in", False):
        expiry = st.user.get("exp")
        if expiry is not None:
            try:
                if datetime.now(timezone.utc).timestamp() >= float(expiry) + SESSION_EXTRA_SECONDS:
                    st.logout()
                    return None
            except (TypeError, ValueError):
                LOGGER.warning("Unusable Google identity expiration claim")
                return None
        email = str(st.user.get("email", "")).strip().lower()
        if not email:
            return None
        identity = {"email":email, "name":str(st.user.get("name", email)), "local":False}
    else:
        identity = st.session_state.get("v3_development_identity") if dev.get("enabled", False) else None
        if not identity:
            return None
        identity = dict(identity)
    identity["is_owner"] = identity["email"].lower() in owner_emails()
    identity["is_admin"] = identity["is_owner"] or admin_in_sheet(identity["email"])
    if not identity["is_admin"]:
        # coba gratis aktif selama belum lewat batas waktu dan belum disetujui
        window = trial_record_window(identity["email"])
        if window and datetime.now(timezone.utc) < window[1] and not approved_in_sheet(identity["email"]):
            identity["trial"] = True
            identity["trial_expires_at"] = window[1].timestamp()
    return identity


def set_view(view):
    """Pindah halaman lewat parameter view di URL."""
    st.query_params["view"] = view


def local_sign_in():
    """Login sebagai developer lokal, cuma kalau mode lokal diaktifkan."""
    settings = development_settings()
    if not settings.get("enabled", False):
        return
    st.session_state["v3_development_identity"] = {
        "email":str(settings.get("email", "developer@localhost")).lower(),
        "name":str(settings.get("name", "Pengembang lokal")), "local":True}


def sign_out_v3():
    """Logout. Semua data sesi dihapus, lalu logout dari Google kalau tadi login pakai Google."""
    was_google = bool(getattr(st.user, "is_logged_in", False))
    for key in list(st.session_state):
        del st.session_state[key]
    st.query_params.clear()
    if was_google:
        st.logout()


def hide_sidebar():
    """Sembunyikan menu samping di halaman login dan halaman persetujuan."""
    st.markdown('<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
                '[data-testid="stExpandSidebarButton"] {display:none!important}</style>', unsafe_allow_html=True)


APP_DESCRIPTION = ("Solyn adalah aplikasi berbasis web untuk memprediksi return harian 11 saham perbankan di "
                   "Bursa Efek Indonesia menggunakan model XGBoost berbasis indikator teknikal, serta membantu "
                   "menyusun bobot portofolio dengan metode Markowitz.")
GOOGLE_ICON_PATH = BASE_DIR / "assets" / "google_g.png"


def image_base64(path):
    """Isi gambar dalam base64 supaya bisa ditaruh langsung di HTML atau CSS."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def render_auth_logo(css_class):
    """Logo Solyn lengkap dengan tulisannya."""
    encoded = image_base64(LOGO_PATH)
    st.markdown(f'<img class="{css_class}" src="data:image/png;base64,{encoded}" alt="Solyn">',
                unsafe_allow_html=True)


def render_auth_layout(render_body):
    """Susunan halaman login dan persetujuan: isi di kiri, logo Solyn di kanan."""
    hide_sidebar()
    with st.container(key="solyn_auth"):
        left, right = st.columns(2, gap="large", vertical_alignment="center")
        with left:
            with st.container(key="solyn_auth_body"):
                render_auth_logo("solyn-auth-logo-mobile")
                render_body()
        with right:
            render_auth_logo("solyn-auth-logo")
    st.markdown('<div class="solyn-auth-foot">© 2026 JA Production. All rights reserved.</div>',
                unsafe_allow_html=True)


def render_login_body():
    """Isi halaman login: judul, deskripsi aplikasi, dan tombol masuk."""
    notice = st.session_state.pop(TRIAL_NOTICE_KEY, None)
    if notice:
        st.info(notice)
    st.markdown('<h1 class="solyn-auth-heading">Selamat datang di Solyn</h1>'
                f'<p class="solyn-auth-lead">{escape(APP_DESCRIPTION)}</p>', unsafe_allow_html=True)
    if auth_is_configured():
        icon = image_base64(GOOGLE_ICON_PATH)
        # ikon Google lewat CSS karena st.button tidak menerima gambar
        st.markdown('<style>.st-key-solyn_login_google button p::before {content:"";display:inline-block;'
                    'width:22px;height:22px;margin-right:14px;vertical-align:-5px;'
                    f'background:url(data:image/png;base64,{icon}) center/contain no-repeat}}</style>',
                    unsafe_allow_html=True)
        if st.button("Masuk dengan Google", key="solyn_login_google", use_container_width=True):
            st.login()
    else:
        st.info("Login Google belum diaktifkan oleh pengelola.")
    if development_settings().get("enabled", False):
        if st.button("Masuk lokal untuk pengujian", key="solyn_login_local", use_container_width=True):
            local_sign_in()
            st.rerun()
    if trial_available():
        note = f"Akun baru bisa mencoba semua fitur gratis selama {trial_settings()['minutes']:g} menit."
    else:
        note = "Akun baru perlu disetujui pengelola sebelum fitur dapat dipakai."
    st.markdown(f'<p class="solyn-auth-note">{escape(note)}</p>', unsafe_allow_html=True)


def render_login_page():
    """Halaman pertama untuk pengguna yang belum masuk. Isi aplikasi tidak ditampilkan sebelum login."""
    render_auth_layout(render_login_body)


def render_access_page(identity):
    """Halaman untuk akun yang sudah masuk tapi belum disetujui admin."""
    st.session_state.pop(TRIAL_NOTICE_KEY, None)
    render_auth_layout(lambda: render_approval_gate(identity))


def local_approval_bypass(identity):
    """Melewati izin admin khusus untuk login lokal yang memang diizinkan."""
    dev = development_settings()
    return bool(identity and identity.get("local") is True
                and dev.get("enabled") is True and dev.get("require_approval", True) is False)


def access_status(identity):
    """Status akses pengguna: approved, pending, new, rejected, atau unavailable."""
    if identity.get("trial"):
        return "approved"
    if local_approval_bypass(identity):
        return "approved"
    if identity.get("is_admin"):
        return "approved"
    if not approval_is_configured():
        return "unavailable"
    try:
        record = find_approval_record(identity["email"])
        status = record.get("Status", "").strip().lower() if record else ""
        return status or "new"
    except Exception:
        LOGGER.exception("Approval status unavailable")
        return "unavailable"


def render_trial_offer(identity, status):
    """Info coba gratis di halaman persetujuan."""
    if status == "unavailable":
        return
    if not trial_available():
        return
    minutes = trial_settings()["minutes"]
    if trial_record_window(identity["email"]):
        return
    with st.container(border=True):
        st.write(f"**Coba gratis {minutes:g} menit**")
        st.caption("Semua fitur pengguna dapat dicoba tanpa menunggu persetujuan. Setiap akun hanya bisa memakai "
                   "coba gratis satu kali. Setelah waktu habis, data unggahan dan model sementara dihapus.")
        if st.button(f"Mulai coba gratis {minutes:g} menit", key="v3_trial_start"):
            result = start_trial(identity, st.query_params.get("view", "home"))
            if result == "started":
                st.rerun()
            elif result == "used":
                st.toast("Coba gratis untuk akun ini sudah pernah dipakai")
            else:
                st.error("Coba gratis belum dapat dimulai. Coba lagi beberapa saat lagi.")


def render_approval_gate(identity):
    """Halaman yang muncul kalau akun Google belum di-approve admin."""
    status = access_status(identity)
    if status == "rejected":
        st.markdown('<h1 class="solyn-auth-heading">Akses ditolak</h1>', unsafe_allow_html=True)
        st.caption(f"Akun: {identity['email']}")
        st.error("**Tidak dapat melanjutkan**  \nAkses Anda ke aplikasi Solyn ditolak oleh pengelola.")
        st.button("Keluar", key="v3_approval_logout", on_click=sign_out_v3)
        return
    st.markdown('<h1 class="solyn-auth-heading">Persetujuan akses</h1>', unsafe_allow_html=True)
    st.caption(f"Akun: {identity['email']}")
    render_trial_offer(identity, status)
    trial_used = trial_available() and bool(trial_record_window(identity["email"]))
    if status == "pending":
        prefix = "Masa coba gratis sudah habis. " if trial_used else ""
        st.info("**Menunggu persetujuan**  \n" + prefix + "Permintaan akses Anda sedang menunggu persetujuan pengelola.")
    elif status == "unavailable":
        # Google Sheet sementara gagal dibaca
        st.info("**Memeriksa status akses**  \nMohon tunggu sebentar, halaman akan diperbarui otomatis.")
    else:
        st.write("Kirim permintaan menggunakan akun Google ini. Persetujuan diperlukan untuk membuat prediksi, simulasi, unggahan, dan membaca hasil pengujian.")
        if st.button("Ajukan akses", type="primary", key="v3_request_access"):
            try:
                create_approval_request(identity)
                st.rerun()
            except Exception:
                LOGGER.exception("Approval request failed")
                st.error("Permintaan belum dapat dikirim. Coba kembali beberapa saat lagi.")
    if status in {"pending", "unavailable"}:
        if status == "pending":
            st.caption("Halaman ini diperbarui otomatis setelah pengelola memberi keputusan.")
        watch_access_status(identity, status)
    st.button("Keluar", key="v3_approval_logout", on_click=sign_out_v3)


@st.fragment(run_every=10)
def watch_access_status(identity, shown_status):
    """Memeriksa status akses tiap 10 detik tanpa tombol.

    Args:
        identity: Akun yang sedang login.
        shown_status: Status yang sedang tampil, misalnya "pending". Halaman
            dimuat ulang jika status di sheet sudah berbeda.
    """
    if access_status(identity) != shown_status:
        st.rerun(scope="app")


# Load model bawaan (runtime/baseline_bundle.zip) dan fungsi bantu tampilan
def bundle_revision():
    """Penanda versi file model bawaan (waktu ubah dan ukuran), dipakai buat cache."""
    path = BASE_DIR / DEFAULT_BASELINE_PATH
    return (path.stat().st_mtime_ns, path.stat().st_size)


@st.cache_resource(show_spinner=False, max_entries=2)
def _baseline_for_revision(revision):
    """Load model bawaan dari runtime/baseline_bundle.zip, di-cache per versi."""
    return read_baseline_bundle((BASE_DIR / DEFAULT_BASELINE_PATH).read_bytes())


def load_local_baseline():
    """Model, contoh input, dan manifest dari model bawaan terbaru."""
    return _baseline_for_revision(bundle_revision())


@st.cache_data(show_spinner=False, max_entries=2)
def _public_manifest(revision):
    """Info ringan buat Home (tanggal data dan daftar saham) tanpa load model."""
    with ZipFile(BASE_DIR / DEFAULT_BASELINE_PATH) as archive:
        raw = json.loads(archive.read("manifest.json"))
    return {k:raw.get(k) for k in ["market_data_date", "published_at", "training_completed_at", "tickers"]}


@st.cache_data(show_spinner=False, max_entries=2)
def _market_history(revision):
    """Riwayat harga yang sudah disesuaikan split, disimpan di dalam bundle."""
    with ZipFile(BASE_DIR / DEFAULT_BASELINE_PATH) as archive:
        history = pd.read_csv(BytesIO(archive.read("market_history.csv")))
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    return history.dropna(subset=["Date", "Ticker", "Close"])


def load_local_market_history():
    """Riwayat harga dari model bawaan terbaru."""
    return _market_history(bundle_revision())


def date_label(value):
    """Tanggal format Indonesia, atau "Belum tersedia"."""
    try:
        return format_date_id(value) if value is not None and not pd.isna(value) else "Belum tersedia"
    except (ValueError, TypeError):
        return "Belum tersedia"


def rupiah(value):
    """Format angka jadi rupiah, contohnya Rp1.000.000."""
    return "Rp" + f"{value:,.0f}".replace(",", ".")


def bank_label(ticker):
    """Teks pilihan saham, contohnya "BBCA - Bank Central Asia"."""
    return f"{ticker} - {BANK_NAMES.get(ticker, ticker)}"


def prediction_target(manifest):
    """Perkiraan tanggal yang diprediksi, yaitu hari kerja setelah data harga terakhir."""
    value = manifest.get("market_data_date")
    try:
        stamp = pd.Timestamp(value)
        if not value or pd.isna(stamp):
            return "Belum tersedia"
        target = stamp + pd.offsets.BDay(1)
        weekday = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][target.weekday()]
        return f"{weekday}, {date_label(target)}"
    except (ValueError, TypeError):
        return "Belum tersedia"


def show_data_status(manifest):
    """Memberi tahu kalau data harga sudah lama, supaya pengguna tahu hasilnya berdasarkan data kapan."""
    value = manifest.get("market_data_date")
    today = pd.Timestamp.now(tz="Asia/Jakarta").date()
    if (today - pd.Timestamp(value).date()).days > 7:
        st.warning(f"Data harga terakhir {date_label(value)}. Hasil bisa kurang sesuai dengan kondisi pasar sekarang.")


def current_session_model():
    """Model sementara hasil upload pengguna di sesi ini, kalau ada."""
    return st.session_state.get("session_model_result")


def source_context():
    """Menentukan model yang dipakai, model bawaan atau model sementara dari upload."""
    models, samples, manifest = load_local_baseline()
    session = current_session_model()
    options = ["baseline"]
    if session:
        options.append("session")
    labels = {"baseline":"Data aplikasi", "session":"Data unggahan saya"}
    current = st.session_state.get("v3_source", options[0])
    if len(options) > 1:
        chosen = st.selectbox("Data yang digunakan", options, index=options.index(current),
            format_func=labels.get, key="v3_source_widget")
    else:
        chosen = options[0]
    st.session_state["v3_source"] = chosen
    if chosen == "session":
        return session["models"], session["latest_features"], session, session["version"]
    return models, samples, manifest, str(bundle_revision())


def ticker_choice(options, key="v3_ticker_widget"):
    """Pilihan saham yang ingat pilihan terakhir pengguna.

    Args:
        options: Kode saham yang bisa dipilih.
        key: Kunci session state widget. Pilihan dibagi antarhalaman lewat
            v3_ticker, kecuali "v3_evaluation_ticker" di Hasil Pengujian yang
            selalu mulai dari saham pertama.
    """
    if key not in st.session_state or st.session_state[key] not in options:
        previous = st.session_state.get("v3_ticker", options[0]) if key != "v3_evaluation_ticker" else options[0]
        st.session_state[key] = previous
    chosen = st.selectbox("Pilih saham", options, format_func=bank_label, key=key)
    if key != "v3_evaluation_ticker":
        st.session_state["v3_ticker"] = chosen
    return chosen


def timestamp_label(value):
    """Menampilkan waktu training atau publikasi dalam WIB."""
    if not value:
        return "Waktu belum tersedia"
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return "Waktu belum tersedia"
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
        stamp = stamp.tz_convert("Asia/Jakarta")
        return f"{date_label(stamp)} ({stamp:%H:%M} WIB)"
    except (ValueError, TypeError):
        return "Waktu belum tersedia"


def render_data_sidebar():
    """Sidebar: tanggal data harga terakhir dan sumber datanya."""
    manifest = _public_manifest(bundle_revision())
    source = "Ringkasan Saham IDX"
    session = current_session_model()
    if st.session_state.get("v3_source_widget", st.session_state.get("v3_source")) == "session":
        manifest = session
        source = "Unggahan Anda dan Ringkasan Saham IDX"
    st.markdown(f'''<div class="solyn-data">
<div class="solyn-data-label">Data harga hingga</div>
<div class="solyn-data-time">{escape(date_label(manifest.get('market_data_date')))}</div>
<div class="solyn-schedule">Sumber data<br><strong>{escape(source)}</strong></div>
<div class="solyn-data-note">Data diperbarui oleh pengelola aplikasi.</div>
</div>''', unsafe_allow_html=True)


def render_nfa():
    """Catatan bukan saran investasi di bawah tiap halaman."""
    st.markdown('''<div class="solyn-nfa"><span><b>Not Financial Advice. Bukan nasihat investasi.</b> Seluruh informasi di aplikasi ini bukan rekomendasi untuk membeli atau menjual saham. Keputusan investasi sepenuhnya tanggung jawab Anda, dan Solyn tidak bertanggung jawab atas kerugian yang timbul dari penggunaan aplikasi ini. Lakukan riset sendiri (DYOR) sebelum berinvestasi.</span></div>''', unsafe_allow_html=True)


def page_heading(title, description):
    """Judul halaman plus penjelasan singkat."""
    st.title(title)
    st.markdown(f'<p class="v3-intro">{escape(description)}</p>', unsafe_allow_html=True)


def render_chart(fig):
    """Menyamakan gaya grafik dengan tema aplikasi, lalu menampilkannya."""
    fig.update_layout(
        paper_bgcolor=CHART_CANVAS, plot_bgcolor=CHART_CANVAS,
        font={"family": CHART_FONT, "size": 16, "color": CHART_TEXT},
        colorway=[CHART_MODEL, CHART_BENCHMARK, CHART_GREEN, CHART_RED],
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 14, "color": CHART_TEXT}},
        hoverlabel={"bgcolor": "#FFFFFF", "font_color": CHART_INK, "bordercolor": CHART_ZEROLINE},
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def build_home_price_chart(history, ticker):
    """Grafik harga penutupan satu saham di Home."""
    selected = history.loc[history["Ticker"].eq(ticker), ["Date", "Close"]].copy()
    selected["Date"] = pd.to_datetime(selected["Date"], errors="coerce")
    selected["Close"] = pd.to_numeric(selected["Close"], errors="coerce")
    selected = selected.dropna(subset=["Date", "Close"]).sort_values("Date")
    if selected.empty:
        return None

    figure = go.Figure(go.Scatter(
        x=selected["Date"],
        y=selected["Close"],
        mode="lines",
        name=ticker,
        line={"color": CHART_GREEN, "width": 2},
        hovertemplate="%{x|%d %b %Y}<br>Harga %{y:,.0f}<extra></extra>",
    ))
    figure.update_layout(
        title=f"Harga penutupan {ticker}",
        height=440,
        xaxis_title="Tanggal",
        yaxis_title="Harga adjusted (Rp)",
        hovermode="x unified",
        margin={"l": 12, "r": 12, "t": 52, "b": 12},
    )
    return figure


def render_market_table(history, tickers, forecasts, target_label):
    """Tabel harga terakhir semua saham, plus prediksinya kalau ada.

    Args:
        history: Riwayat harga dengan kolom Ticker, Date, dan Close.
        forecasts: Prediksi return per saham dalam persen. Saham yang tidak punya
            model ditampilkan "Model belum tersedia".
        target_label: Tanggal target prediksi untuk judul kolom.
    """
    records = []
    for ticker in sorted(tickers):
        subset = history[history['Ticker'] == ticker].sort_values('Date')
        prices = pd.to_numeric(subset['Close'], errors='coerce').dropna()
        close = f"{prices.iloc[-1]:,.0f}".replace(',', '.') if len(prices) else 'Tidak tersedia'
        change = (prices.iloc[-1]/prices.iloc[-2]-1)*100 if len(prices)>1 else None
        change_text = f"{change:+.2f}%" if change is not None else 'Tidak tersedia'
        change_class = 'solyn-up' if change is not None and change>=0 else 'solyn-down'
        prediction = forecasts.get(ticker)
        pred_cell = (f'<td class="num {"solyn-up" if prediction>=0 else "solyn-down"}">{prediction:+.3f}%</td>' if prediction is not None else '<td class="num solyn-pending">Model belum tersedia</td>')
        records.append(f'<tr><td><strong>{escape(ticker)}</strong><small>{escape(BANK_NAMES.get(ticker,ticker))}</small></td><td class="num">{close}</td><td class="num {change_class}">{change_text}</td>{pred_cell}</tr>')
    dates = sorted(pd.to_datetime(history['Date']).dropna().unique())
    last_day = date_label(dates[-1])
    previous_day = date_label(dates[-2])
    forecast_header = f'<th class="num">Prediksi return<br>{escape(target_label)}</th>'
    st.markdown('<div class="solyn-table-wrap"><table class="solyn-table"><thead><tr><th>Saham</th>'
                f'<th class="num">Harga penutupan (Rp)<br>{escape(last_day)}</th>'
                f'<th class="num">Perubahan harga<br>dari {escape(previous_day)}</th>'
                + forecast_header + '</tr></thead><tbody>' + ''.join(records) + '</tbody></table></div>', unsafe_allow_html=True)


# Halaman Home
def render_home():
    """Halaman Home: ringkasan pasar, grafik harga, dan jalan pintas ke fitur lain."""
    tickers = PORTFOLIO_TICKERS
    # Menampilkan judul dan deskripsi halaman Home
    page_heading("Prediksi saham perbankan",
                 "Lihat prediksi return sebelas saham perbankan dan simulasikan pembagian dana investasi Anda.")
    models, samples, metadata, version = source_context()
    # Button shortcut
    with st.container(key="solyn_home_actions"):
        a, b = st.columns(2, gap="small")
        a.button("Prediksi Saham", type="primary", use_container_width=True,
                 key="v3_home_predict", on_click=set_view, args=("analysis",))
        b.button("Simulasi Portofolio", use_container_width=True,
                 key="v3_home_portfolio", on_click=set_view, args=("portfolio",))
    # Subjudul dan deskripsi sebelum tabel
    target = prediction_target(metadata)
    st.markdown(f'<div class="solyn-section-head solyn-home-head"><h2>Harga dan Prediksi Saham untuk {escape(target)}</h2></div>'
                '<p class="solyn-section-note">Kalau tanggal hari ini bursa libur, prediksi berlaku untuk hari bursa setelahnya.</p>',
                unsafe_allow_html=True)
    # Menghitung prediksi return tiap saham menggunakan model XGBoost
    forecasts = {code: float(model.predict(pd.DataFrame([samples[code]], columns=FEATURES))[0]) * 100
                 for code, model in models.items()}
    history = metadata.get("session_history")
    if history is None:
        history = load_local_market_history()

    # Menampilkan Tabel
    render_market_table(history, tickers, forecasts, target_label=prediction_target(metadata))

    # Subjudul grafik harga
    st.markdown('<div class="solyn-section-head"><h2>Grafik harga</h2></div>', unsafe_allow_html=True)
    # Kolom dropdown
    chart_ticker = st.selectbox("Pilih saham", sorted(tickers), key="v3_home_chart_ticker")
    price_history = load_local_market_history()
    figure = build_home_price_chart(price_history, chart_ticker)
    # Menampilkan grafik harga penutupan saham yang dipilih, atau pesan jika belum tersedia
    if figure is None:
        st.info(f"Riwayat harga {chart_ticker} belum tersedia.")
    else:
        render_chart(figure)
        st.caption("Notes: Harga telah disesuaikan terhadap stock split. Grafik dapat diperbesar dengan menyeret kursor. Klik dua kali pada grafik untuk kembali ke tampilan awal.")


# Halaman Prediksi Saham
def latest_close(history, ticker):
    """Harga penutupan terakhir satu saham beserta tanggalnya, atau None kalau belum ada."""
    subset = history[history["Ticker"] == ticker].sort_values("Date")
    subset = subset.assign(Close=pd.to_numeric(subset["Close"], errors="coerce")).dropna(subset=["Close"])
    if subset.empty:
        return None
    last = subset.iloc[-1]
    return float(last["Close"]), last["Date"]


def input_signature(ticker, source_version, mode, values):
    """Kunci unik dari isian indikator, buat tahu prediksi perlu dihitung ulang atau tidak.

    Args:
        source_version: Versi sumber model dari source_context, supaya kunci
            berubah setelah model diganti.
        mode: Metode prediksi, data terakhir atau isian sendiri.
        values: Nilai 10 indikator.

    Returns:
        Hash SHA-256 dalam bentuk teks hex.
    """
    payload = [ticker, source_version, mode, [float(values[k]) for k in FEATURES]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def feature_errors(values):
    """Validasi input teknikal indikator"""
    errors = []
    if not all(np.isfinite(v) for v in values.values()):
        return ["Semua indikator harus berupa angka yang dapat dihitung."]
    for key in ["RSI14", "Stochastic_%K"]:
        if not 0 <= values[key] <= 100:
            errors.append(f"{FEATURE_LABELS[key]} harus berada antara 0 dan 100.")
    for key in ["SMA10", "SMA20", "EMA10", "ATR14"]:
        if values[key] < 0:
            errors.append(f"{FEATURE_LABELS[key]} tidak boleh negatif.")
    if values["BB_Upper"] < values["BB_Lower"]:
        errors.append("Batas harga atas harus sama dengan atau lebih besar daripada batas bawah.")
    return errors


SHAP_HELP = ("SHAP adalah cara melihat indikator mana yang paling membuat angka prediksi naik atau turun. "
             "Setiap indikator diberi nilai: tanda plus berarti ikut menaikkan, tanda minus berarti ikut menurunkan.")

CALC_HELP = ("Model mulai dari nilai dasar prediksi, yaitu rata-rata return dari data lama. "
             "Lalu setiap indikator menambah atau mengurangi nilai itu, sampai totalnya menjadi angka prediksi.")


def render_explanation(model, values, prediction):
    """Penjelasan prediksi, kontribusi tiap indikator pakai SHAP dari XGBoost."""
    with st.expander("Lihat Nilai SHAP (Opsional)"):
        try:
            # model dibuat ulang dari JSON supaya cocok dengan modul XGBoost yang aktif
            from xgboost.core import Booster, DMatrix
            explanation_model = Booster(model_file=model.get_booster().save_raw())
            matrix = DMatrix(pd.DataFrame([values], columns=FEATURES))
            terms = explanation_model.predict(matrix, pred_contribs=True)[0]
            if not np.isclose(terms.sum(), prediction, atol=1e-6):
                raise ValueError("Contribution sum mismatch")
            frame = pd.DataFrame({"Indikator":[FEATURE_LABELS[f] for f in FEATURES],
                                  "Kontribusi (%)":terms[:-1] * 100})
            frame = frame.iloc[np.argsort(np.abs(terms[:-1]))]
            fig = go.Figure(go.Bar(x=frame.iloc[:,1], y=frame.iloc[:,0], orientation="h",
                marker_color=[CHART_GREEN if x >= 0 else CHART_RED for x in frame.iloc[:,1]],
                hovertemplate="%{y}<br>%{x:+.4f} %<extra></extra>"))
            fig.update_layout(height=390, margin=dict(l=0,r=15,t=15,b=10), xaxis_title="Kontribusi terhadap prediksi (%)")
            left, right = st.columns(2, gap="large")
            left.markdown(
                f'<p class="solyn-field-label">SHAP<i class="solyn-hint" data-tip="{escape(SHAP_HELP)}">?</i></p>'
                '<p class="solyn-section-note">Batang hijau mendorong angka prediksi ke atas, '
                'batang merah menariknya ke bawah. Makin panjang batangnya, makin besar pengaruhnya.</p>',
                unsafe_allow_html=True)
            right.markdown(
                f'<p class="solyn-field-label">Perhitungan<i class="solyn-hint" data-tip="{escape(CALC_HELP)}">?</i></p>'
                f'<p class="solyn-section-note">Nilai dasar {terms[-1]*100:+.3f}% ditambah kontribusi seluruh '
                f'indikator menghasilkan prediksi {prediction*100:+.3f}%.</p>', unsafe_allow_html=True)
            render_chart(fig)
            st.caption("Grafik dapat diperbesar dengan menyeret kursor. Klik dua kali pada grafik untuk kembali ke tampilan awal.")
            st.markdown(
                '<p class="v3-disclaimer">SHAP menjelaskan cara model menghitung angkanya, bukan penyebab naik '
                'atau turunnya harga saham di pasar.</p>', unsafe_allow_html=True)
        except Exception:
            LOGGER.exception("Explanation unavailable")
            st.info("Penjelasan belum dapat ditampilkan untuk model ini. Prediksi tetap tersedia.")


def render_latest_indicators(values, data_date):
    """Nilai indikator yang dipakai model, dengan penjelasan saat kursor diarahkan ke tanda tanya."""
    rows = "".join(
        f'<div><span>{escape(FEATURE_LABELS[f])}'
        f'<i class="solyn-hint" data-tip="{escape(FEATURE_HELP[f])}">?</i></span>'
        f'<b>{values[f]:,.4f}</b></div>' for f in FEATURES)
    st.markdown(f'<p class="solyn-field-label">Nilai indikator berdasarkan data tanggal {escape(date_label(data_date))}'
                f'<i class="solyn-hint" data-tip="{escape(INDICATOR_HELP)}">?</i></p>'
                f'<div class="solyn-indicators">{rows}</div>', unsafe_allow_html=True)


MODE_LATEST = "Data Terbaru (Rekomendasi)"
MODE_CUSTOM = "Skenario Sendiri"


def render_analysis():
    """Halaman Prediksi Saham: mengisi atau mengubah indikator lalu dapat prediksi return besok."""

    # Menampilkan judul dan deskripsi halaman Prediksi Saham
    page_heading("Prediksi return saham",
                 "Pilih saham, lalu tekan Buat prediksi untuk melihat perkiraan perubahan harga pada satu hari bursa berikutnya.")

    models, samples, metadata, version = source_context()

    # Menampilkan dropdown pilih saham
    ticker = ticker_choice(PORTFOLIO_TICKERS)

    # Menampilkan status tanggal dan sumber data yang dipakai
    show_data_status(metadata)
    if ticker not in models:
        st.info(f"{ticker} sudah masuk daftar saham aplikasi. Model prediksinya belum tersedia.")
        st.button("Buat prediksi", type="primary", disabled=True)
        return

    # Menampilkan radio button pilih metode prediksi: Data Terbaru atau Skenario Sendiri
    mode = st.radio("Pilih Metode Prediksi", [MODE_LATEST, MODE_CUSTOM], horizontal=True, key="v3_prediction_mode",
                    help=f"{MODE_LATEST} menggunakan nilai indikator dari data harga terakhir aplikasi. "
                         f"{MODE_CUSTOM} memungkinkan Anda memasukkan nilai indikator, lalu melihat prediksi return berdasarkan input Anda.")
    sample = samples[ticker]
    values = {f:float(sample[f]) for f in FEATURES}

    if mode == MODE_CUSTOM:
        # Menampilkan info dan 10 kolom input angka indikator untuk diisi manual
        st.info("Isi nilai indikator teknikal pada kolom di bawah ini untuk memprediksi return berdasarkan input Anda.")
        prefix = "v3_feature_" + hashlib.sha256(f"{ticker}-{version}".encode()).hexdigest()[:12]
        # isian dimulai kosong, tombol plus minus disembunyikan lewat CSS
        with st.container(key="solyn_scenario_inputs"):
            cols = st.columns(2)
            for i, f in enumerate(FEATURES):
                with cols[i%2]:
                    values[f] = st.number_input(FEATURE_LABELS[f], value=None, format="%.4f", key=prefix+f,
                                                help=FEATURE_HELP[f], placeholder=f"Contoh: {float(sample[f]):.4f}")
    else:
        # Menampilkan 10 nilai indikator teknikal dari data terakhir beserta tooltip
        render_latest_indicators(values, metadata.get("market_data_date"))
    
    # Validasi input user di kolom indikator teknikal
    missing = [f for f in FEATURES if values[f] is None]
    errors = [] if missing else feature_errors(values)
    for error in errors:
        st.error(error)
    if missing:
        st.caption(f"Masih ada {len(missing)} indikator yang belum diisi. Tombol 'Buat prediksi' akan aktif setelah semuanya terisi.")
    signature = None if missing else input_signature(ticker, version, mode, values)

    # Menampilkan tombol Buat prediksi, menjalankan model.predict() saat ditekan
    if st.button("Buat prediksi", type="primary",
                 disabled=bool(errors or missing)):
        try:
            with st.spinner("Menghitung prediksi..."):
                prediction = float(models[ticker].predict(pd.DataFrame([values], columns=FEATURES))[0])
                st.session_state["v3_prediction"] = {"signature":signature,"value":prediction}
        except Exception:
            LOGGER.exception("Prediction failed")
            st.error("Error saat melakukan prediksi. Silahkan coba lagi.")
    result = st.session_state.get("v3_prediction")
    if result and result["signature"] != signature:
        # Buang hasil lama kalau isian berubah sejak tombol ditekan
        st.session_state.pop("v3_prediction", None)
        result = None
    if not result:
        return

    # Menampilkan subjudul Hasil Prediksi
    st.markdown('<div class="solyn-section-head"><h2>Hasil Prediksi</h2></div>', unsafe_allow_html=True)
    value = result["value"]
    tone = "solyn-up" if value > 0 else "solyn-down" if value < 0 else ""
    if mode == MODE_CUSTOM:
        sentence = f"Berdasarkan skenario Anda, harga {ticker} diperkirakan {value*100:+.3f}% pada {prediction_target(metadata)}."
    else:
        sentence = (f"Harga {ticker} diperkirakan {value*100:+.3f}% pada {prediction_target(metadata)}, "
                    f"dibanding harga penutupan {date_label(metadata.get('market_data_date'))}.")

    # Estimasi harga = harga penutupan terakhir x (1 + prediksi return)
    history = metadata.get("session_history")
    if history is None:
        history = load_local_market_history()
    last = latest_close(history, ticker)
    price_line = ""
    if last:
        close, close_date = last
        price_line = (f'<div class="v3-result-price">Estimasi dalam bentuk harga<b>{rupiah(close * (1 + value))}</b>')

    # Menampilkan angka prediksi return dalam persen, estimasi harga, dan kalimat penjelasan
    st.markdown(
        f'<div class="v3-result"><strong class="{tone}">{value*100:+.3f}%</strong>'
        f'{price_line}'
        f'<p>{escape(sentence)}</p>'
        f'<small class="v3-disclaimer">Hasil merupakan estimasi berdasarkan prediksi model, bukan kepastian perubahan harga.</small></div>',
        unsafe_allow_html=True)

    # Menampilkan expander grafik SHAP
    render_explanation(models[ticker], values, value)

    # Button navigasi
    st.markdown('<p class="solyn-next-label">Langkah Selanjutnya</p>', unsafe_allow_html=True)
    a,b=st.columns(2)
    a.button("Buka simulasi portofolio", type="primary", on_click=set_view, args=("portfolio",), use_container_width=True)
    b.button("Lihat hasil pengujian", on_click=set_view, args=("performance",), use_container_width=True)


# Halaman Simulasi Portofolio
ALLOCATION_LIGHT = (234, 241, 250)
ALLOCATION_DARK = (143, 179, 216)


def allocation_color(share):
    """Warna satu potongan donut, ditentukan berdasarkan besar bobot."""
    position = float(share)
    return "#%02X%02X%02X" % tuple(
        round(light + (dark - light) * position)
        for light, dark in zip(ALLOCATION_LIGHT, ALLOCATION_DARK))


def allocation_problem(count, cap):
    """Pesan kalau jumlah saham terlalu sedikit untuk batas bobot yang dipilih.

    Jumlah minimal saham adalah 100 / cap dibulatkan ke atas, paling sedikit 2.

    Args:
        count: Jumlah saham yang dipilih.
        cap: Batas bobot per saham dalam persen (20 = 20%), beda dengan
            max_weight di optimize_max_sharpe yang memakai desimal.

    Returns:
        Pesan untuk pengguna, atau None jika jumlah saham sudah cukup.
    """
    minimum = max(2, int(np.ceil(100 / cap - 1e-10)))
    if count < minimum:
        return (f"Anda memilih {count} saham. Dengan batas maksimal {cap:g}% untuk satu saham, pilih minimal {minimum} saham. "
                f"Tambahkan {minimum-count} saham, atau naikkan batasnya pada Pengaturan Lanjutan (Opsional).")
    return None


PORTFOLIO_INPUTS = ["v3_assets", "v3_capital", "v3_rate", "v3_cap"]


def portfolio_key(name):
    """Nama isian portofolio, diberi nomor putaran supaya bisa dikembalikan ke nilai awal."""
    return f"{name}_{st.session_state.get('v3_portfolio_round', 0)}"


def reset_portfolio_inputs(announce=False):
    """Mengembalikan isian simulasi portofolio ke nilai awal."""
    st.session_state.pop("v3_allocation", None)
    for name in PORTFOLIO_INPUTS:
        st.session_state.pop(portfolio_key(name), None)
    st.session_state["v3_portfolio_round"] = st.session_state.get("v3_portfolio_round", 0) + 1
    if announce:
        st.session_state["v3_portfolio_reset_note"] = True


def simulation_csv(frame, data_date, capital, rate, cap):
    """Isi file CSV hasil simulasi: tabel alokasi di kiri, pengaturan simulasi di kanan.

    Args:
        frame: Tabel alokasi dengan kolom Saham, Bobot (%), Dana (Rp), dan Prediksi return (%).
        data_date: Tanggal data yang dipakai.
        capital: Dana simulasi dalam rupiah.
        rate: Risk-free rate tahunan dalam persen.
        cap: Batas bobot per saham dalam persen.

    Returns:
        Isi CSV.
    """
    table = [[row["Saham"], round(row["Bobot (%)"], 2), int(round(row["Dana (Rp)"])), round(row["Prediksi return (%)"], 4)]
             for _, row in frame.iterrows()]
    settings = [["Tanggal data", data_date], ["Dana simulasi", int(capital)],
                ["Bunga tahunan (%)", rate], ["Batas per saham (%)", cap]]
    rows = [["Saham", "Bobot (%)", "Dana (Rp)", "Prediksi return (%)", ""] + settings[0]]
    for i in range(max(len(table), len(settings) - 1)):
        left = table[i] if i < len(table) else ["", "", "", ""]
        right = settings[i + 1] if i + 1 < len(settings) else ["", ""]
        rows.append(left + [""] + right)
    output = StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def render_portfolio():
    """Halaman Simulasi Portofolio: bobot Max Sharpe dari prediksi model dan nominal modal."""

    # Menampilkan judul dan deskripsi halaman Simulasi Portofolio
    page_heading("Simulasi alokasi portofolio",
                 "Pilih saham dan masukkan dana yang ingin disimulasikan. Seluruh dana dibagi habis ke saham yang dipilih, "
                 "dan aplikasi menghitung persentase serta nominal alokasinya berdasarkan estimasi return dan risiko.")

    models, samples, metadata, version = source_context()

    # Menampilkan status tanggal dan sumber data yang dipakai
    show_data_status(metadata)
    portfolio = metadata["portfolio"]

    available = [ticker for ticker in PORTFOLIO_TICKERS if ticker in models]
    scope_version = version + "|" + ",".join(PORTFOLIO_TICKERS)
    if st.session_state.get("v3_portfolio_version") != scope_version:
        reset_portfolio_inputs()
        st.session_state["v3_portfolio_version"] = scope_version

    # Kolom input multiselect untuk milih saham
    selected = st.multiselect("Saham dalam portofolio", PORTFOLIO_TICKERS, default=available, key=portfolio_key("v3_assets"),
                              help="Alokasi dana akan dibagi ke Saham yang dipilih pada kolom ini.")
    minimum_note = st.empty()
    pending = [ticker for ticker in selected if ticker not in available]

    # Menampilkan input nominal dana simulasi investasi (rupiah)
    capital = st.number_input("Dana simulasi (rupiah, minimal Rp1.000.000)", min_value=1_000_000, value=100_000_000, step=1_000_000,
                              key=portfolio_key("v3_capital"),
                              help="Jumlah uang sebagai simulasi yang akan di alokasikan ke saham yang telah anda pilih.")
    st.caption(f"Dana simulasi: **{rupiah(capital)}**")
    default_rate = float(portfolio["risk_free_annual"]) * 100

    # Menampilkan expander pengaturan lanjutan
    with st.expander("Pengaturan Lanjutan (Opsional)"):
        rate = st.number_input("Risk-free Rate (%)", min_value=0.0, max_value=25.0,
             value=default_rate, step=.25, key=portfolio_key("v3_rate"),
             help="Bunga yang bisa didapat tanpa risiko, misalnya dari deposito. Angka ini dipakai sebagai pembanding: "
                  "keuntungan portofolio dihitung bagus kalau lebih tinggi daripada angka ini.")
        cap = st.number_input("Batas maksimal alokasi satu saham (%)", min_value=10.0, max_value=100.0, value=20.0, step=5.0,
                              key=portfolio_key("v3_cap"),
                              help="Persentase maksimal satu saham dalam portofolio.")
    problem = allocation_problem(len(selected), cap)
    minimum = max(2, int(np.ceil(100 / cap - 1e-10)))
    minimum_note.caption(f"Dengan batas maksimal {cap:g}% untuk satu saham, perlu minimal {minimum} saham. "
                         "Batas ini dapat diubah pada Pengaturan Lanjutan (Opsional).")

    # Menampilkan warning jika jumlah saham yang dipilih kurang dari batas minimum
    if problem:
        st.warning(problem)
    settings_changed = (set(selected) != set(available) or capital != 100_000_000
                        or abs(rate - default_rate) > 1e-9 or abs(cap - 20.0) > 1e-9)
    a,b = st.columns([1, 1.25])

    # Menampilkan tombol Hitung alokasi
    calculate = a.button("Hitung alokasi", type="primary", disabled=bool(problem or pending), use_container_width=True)

    # Menampilkan tombol Kembalikan ke pengaturan awal jika pengaturan diubah
    if settings_changed:
        b.button("Kembalikan ke pengaturan awal", on_click=reset_portfolio_inputs, args=(True,), use_container_width=True)
    if st.session_state.pop("v3_portfolio_reset_note", False):
        st.toast("Pengaturan berhasil dikembalikan")
    signature = json.dumps([version, selected, capital, rate, cap])

    # Menghitung pembagian dana portofolio saat tombol ditekan
    if calculate:
        try:
            with st.spinner("Menghitung pembagian dana..."):
                expected = pd.Series(portfolio["predicted_returns"], dtype=float).reindex(selected)
                covariance = pd.DataFrame(portfolio["covariance"], dtype=float).loc[selected,selected].to_numpy(copy=True)
                # rate tahunan dalam persen diubah ke harian (252 hari bursa)
                rf = rate / 100 / 252
                weights = optimize_max_sharpe(expected.to_numpy(), covariance.copy(), rf, cap/100)
                ret = float(weights @ expected.to_numpy())
                vol = float(np.sqrt(weights @ covariance @ weights))
                frame = pd.DataFrame({"Saham":selected,"Bobot (%)":weights*100,"Dana (Rp)":weights*capital,
                    "Prediksi return (%)":expected.to_numpy()*100}).sort_values("Bobot (%)", ascending=False)
                frame.loc[frame["Bobot (%)"] < 1e-7, ["Bobot (%)","Dana (Rp)"]] = 0
                st.session_state["v3_allocation"] = {"signature":signature,"frame":frame,"ret":ret,"vol":vol,"sharpe":(ret-rf)/vol}
        except Exception:
            LOGGER.exception("Portfolio calculation failed")
            st.error("Alokasi belum berhasil dihitung. Tekan Kembalikan ke pengaturan awal, atau gunakan lebih banyak saham.")
            st.session_state.pop("v3_allocation", None)
    result = st.session_state.get("v3_allocation")
    if result and result["signature"] != signature:
        # hasil lama dibuang kalau pengaturan berubah sejak tombol ditekan
        st.session_state.pop("v3_allocation", None)
        result = None
    if not result:
        return
    frame = result["frame"]

    # Menampilkan subjudul dan grafik batang prediksi return tiap saham
    st.subheader("Prediksi return tiap saham")
    data_source = "Unggahan Anda dan riwayat Ringkasan Saham IDX" if metadata.get("source") == "session_upload" else "Ringkasan Saham IDX"
    st.caption(f"Sumber data: {data_source}. Prediksi untuk hari bursa berikutnya: {prediction_target(metadata)}.")
    predictions = frame.set_index("Saham").loc[selected, "Prediksi return (%)"].sort_values()
    fig_pred = go.Figure(go.Bar(
        x=predictions.to_numpy(), y=list(predictions.index), orientation="h",
        marker_color=[CHART_GREEN if value >= 0 else CHART_RED for value in predictions],
        text=[f"{value:+.3f}%" for value in predictions], textposition="outside",
        hovertemplate="%{y}<br>%{x:+.3f}%<extra></extra>"))
    span = float(max(abs(predictions.min()), abs(predictions.max()), 1e-6))
    fig_pred.update_layout(height=430, margin=dict(l=0, r=20, t=10, b=10),
                           xaxis_title="Prediksi return hari bursa berikutnya (%)",
                           xaxis_range=[min(float(predictions.min()), 0) - span * .45,
                                        max(float(predictions.max()), 0) + span * .45])
    # Menampilkan grafik batang prediksi return
    render_chart(fig_pred)
    # Menampilkan subjudul dan tiga metrik hasil portofolio (Return, Volatilitas, Sharpe Ratio)
    st.subheader("Perkiraan hasil portofolio")
    a,b,c = st.columns(3)
    a.metric("Return", f"{result['ret']*100:+.3f}%", help="Perkiraan perubahan nilai portofolio untuk satu hari perdagangan berikutnya.")
    b.metric("Volatilitas", f"{result['vol']*100:.3f}%", help="Seberapa besar nilai portofolio biasanya naik turun dalam sehari. Makin besar, makin berisiko.")
    c.metric("Sharpe Ratio", f"{result['sharpe']:.3f}", help="Keuntungan untuk setiap satuan risiko, setelah dikurangi bunga tanpa risiko. Makin tinggi makin baik.")
    # Menampilkan subjudul dan pembagian alokasi dana
    st.subheader("Grafik alokasi portofolio")
    st.caption("Simulasi belum memperhitungkan biaya transaksi, pajak, slippage (selisih harga saat transaksi), dan pembulatan ke satuan lot pembelian saham.")
    pie_frame = frame[frame["Bobot (%)"] > 1e-7]
    weights_pct = pie_frame["Bobot (%)"].to_numpy()
    largest = float(weights_pct.max())
    steps = [allocation_color(value / largest) for value in weights_pct]
    fig = go.Figure(go.Pie(
        labels=pie_frame["Saham"], values=pie_frame["Bobot (%)"], hole=.45,
        marker=dict(colors=steps, line=dict(color=CHART_CANVAS, width=2)),
        text=[rupiah(value) for value in pie_frame["Dana (Rp)"]],
        texttemplate="%{label}<br>%{value:.1f}%<br>%{text}", textposition="inside",
        insidetextorientation="horizontal", textfont=dict(size=12, color=CHART_INK),
        hovertemplate="%{label}<br>Bobot %{value:.2f}%<br>%{text}<extra></extra>",
        sort=False, showlegend=False,
    ))
    fig.update_layout(height=520, margin=dict(l=8,r=8,t=10,b=10),
                      uniformtext=dict(minsize=11, mode="hide"),
                      annotations=[dict(text=f"Total dana<br><b>{rupiah(capital)}</b>", x=.5, y=.5,
                                        showarrow=False, font=dict(size=15, color=CHART_INK))])
    # Menampilkan diagram donat (pie chart) pembagian dana dan bobot saham
    render_chart(fig)
    # Menampilkan tombol unduh hasil simulasi dalam format CSV
    st.download_button("Unduh hasil simulasi CSV (Opsional)", simulation_csv(frame, metadata.get("market_data_date"), capital, rate, cap),
                       "simulasi_portofolio.csv", "text/csv")
    # Menampilkan tombol navigasi untuk pindah ke halaman Unggah Data
    st.button("Perbarui Model", type="primary", on_click=set_view, args=("upload",))


# Halaman Hasil Pengujian
@st.cache_data(show_spinner=False)
def read_research_file(filename, revision):
    """Membaca satu file hasil notebook, di-cache per versi file. Menerima .parquet dan .csv.

    Args:
        filename: Nama file hasil notebook.
        revision: Waktu modifikasi file. Tidak dipakai di isi fungsi, hanya
            menjadi kunci cache supaya file dibaca ulang setelah berubah.
    """
    path = research_path(filename)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def research_file(filename):
    """Membaca file hasil notebook, di-cache per versi file (waktu modifikasi)."""
    return read_research_file(filename, research_path(filename).stat().st_mtime_ns)


def render_metric_cards(cards):
    """Kartu metrik dengan tinggi sama: nama metrik, angka besar, dan penjelasan singkat."""
    html = "".join(f'<div class="solyn-metric-card"><b>{escape(name)}</b><div class="solyn-metric-value">{escape(value)}</div>'
                   f'<span>{escape(note)}</span></div>' for name, value, note in cards)
    st.markdown(f'<div class="solyn-metric-cards">{html}</div>', unsafe_allow_html=True)


def comparison_verdicts(rows):
    """Menentukan siapa yang lebih baik untuk tiap ukuran. Hasilnya daftar True, False, atau None kalau sama."""
    verdicts = []
    for _, _, model_value, benchmark_value, _, higher_is_better in rows:
        if model_value == benchmark_value:
            verdicts.append(None)
        else:
            verdicts.append(model_value > benchmark_value if higher_is_better else model_value < benchmark_value)
    return verdicts


def render_comparison_cards(rows):
    """Perbandingan model dan pembanding dalam kartu, satu kartu per ukuran."""
    cards = ""
    for index, ((name, note, model_value, benchmark_value, number_format, _), model_better) in enumerate(zip(rows, comparison_verdicts(rows))):
        if model_better is None:
            verdict, tone = "Hasilnya sama", "solyn-verdict-flat"
        elif model_better:
            verdict, tone = "Model lebih baik", "solyn-verdict-up"
        else:
            verdict, tone = "Pembanding lebih baik", "solyn-verdict-down"
        side = " solyn-hint-right" if index >= len(rows) / 2 else ""
        cards += (f'<div class="solyn-compare-card"><p class="solyn-field-label">{escape(name)}'
                  f'<i class="solyn-hint{side}" data-tip="{escape(note)}">?</i></p>'
                  f'<div class="solyn-compare-value">{escape(number_format.format(model_value))}</div>'
                  f'<div class="solyn-compare-side">Pembanding {escape(number_format.format(benchmark_value))}</div>'
                  f'<div class="{tone}">{escape(verdict)}</div></div>')
    st.markdown(f'<div class="solyn-compare-cards">{cards}</div>', unsafe_allow_html=True)


SECTION_PORTFOLIO = "Evaluasi Portofolio"
SECTION_MODEL = "Evaluasi Model"


def render_performance():
    """Halaman Hasil Pengujian (statis)"""
    # Menampilkan judul dan deskripsi halaman Hasil Pengujian
    page_heading("Hasil pengujian 2025",
                 "Seberapa baik prediksi model dan portofolio Solyn bekerja ketika diuji pada tahun 2025.")

    # Radiobutton Lihat Pengujian
    section = st.radio("Lihat pengujian", [SECTION_PORTFOLIO, SECTION_MODEL], horizontal=True)
    try:
        if section == SECTION_MODEL:
            frame = research_file("evaluation_metrics_2025.parquet")
            # Menampilkan dropdown pilih saham untuk evaluasi model
            ticker = ticker_choice(frame["Ticker"].tolist(), "v3_evaluation_ticker")
            row = frame.set_index("Ticker").loc[ticker]
            # RMSE kalau prediksi selalu 0%
            context = research_file("evaluation_rmse_context_2025.csv")
            zero_rmse = float(context.loc[context["Ticker"] == ticker, "RMSE Prediksi Nol"].iloc[0]) * 100
            rmse_note = ("Rata-rata selisih prediksi dan hasil aktual dalam poin persen, selisih besar dihitung lebih berat."
                         f" Kalau prediksinya selalu ditulis 0%, angkanya {zero_rmse:.3f}.")
            # Menampilkan empat kartu metrik evaluasi model (RMSE, MAE, R2 Score, Directional Accuracy)
            render_metric_cards([
                ("RMSE", f"{row['RMSE']*100:.3f}", rmse_note),
                ("MAE", f"{row['MAE']*100:.3f}", "Rata-rata besar meleset prediksi setiap hari dari hasil aktual, dalam poin persen."),
                ("R²", f"{row['R2 Score']:.3f}", "Nilai 0 setara dengan selalu menebak angka rata-rata. Di atas 0 lebih baik daripada tebakan itu, di bawah 0 lebih buruk."),
                ("Directional accuracy", f"{row['Directional Accuracy (%)']:.1f}%", "Persentase hari ketika arah naik atau turun ditebak dengan benar. Menebak arah secara acak menghasilkan sekitar 50%."),
            ])
            if float(row["R2 Score"]) < 0:
                st.markdown('<p class="solyn-section-note">Pada data 2025, model belum memprediksi return harian '
                            f'{escape(ticker)} lebih baik daripada tebakan sederhana. Hasil model lebih terlihat pada '
                            'penyusunan portofolio di bagian Evaluasi Portofolio.</p>', unsafe_allow_html=True)
            predictions = research_file("predictions_2025.parquet")
            selected = predictions[predictions["Kode Saham"] == ticker].sort_values("target_date")
            st.caption(f"Periode realisasi: {date_label(selected['target_date'].min())} sampai {date_label(selected['target_date'].max())} · {len(selected)} observasi.")
            actual = selected["target_next_return"].to_numpy(dtype=float) * 100
            forecast = selected["y_pred"].to_numpy(dtype=float) * 100
            # Membuat grafik sebaran return aktual vs garis prediksi model
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=selected["target_date"], y=actual, mode="markers", name="Return aktual",
                marker=dict(color=CHART_ACTUAL, size=6, opacity=0.75),
                hovertemplate="%{x|%d %b %Y}<br>Aktual %{y:+.2f}%<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=selected["target_date"], y=forecast, mode="lines", name="Prediksi model",
                line=dict(color=CHART_FORECAST, width=2.5),
                hovertemplate="%{x|%d %b %Y}<br>Prediksi %{y:+.2f}%<extra></extra>",
            ))
            fig.add_hline(y=0, line_color=CHART_REFERENCE, line_width=1)
            fig.update_layout(height=400, yaxis_title="Return satu hari berikutnya (%)",
                              hovermode="x unified", legend=dict(orientation="h", y=1.08))
            # Menampilkan grafik perbandingan return aktual dan prediksi model
            st.subheader("Aktual vs Prediksi Model")
            st.caption("Titik menunjukkan hasil aktual tiap hari. Garis menunjukkan prediksi model pada skala yang sama. Garis yang hampir datar berarti model belum menangkap perubahan harian dengan baik.")
            render_chart(fig)
            # Grafik selisih return aktual dan prediksi tiap hari
            delta = actual - forecast
            delta_fig = go.Figure(go.Bar(
                x=selected["target_date"], y=delta, name="Selisih",
                marker_color=np.where(delta >= 0, CHART_ACTUAL, CHART_ACTUAL_BELOW),
                hovertemplate="%{x|%d %b %Y}<br>Aktual dikurangi prediksi %{y:+.2f} %<extra></extra>",
            ))
            delta_fig.add_hline(y=0, line_color=CHART_REFERENCE, line_width=1)
            delta_fig.update_layout(height=320, yaxis_title="Selisih (%)",
                                    hovermode="x unified", showlegend=False)
            st.subheader("Selisih Aktual dan Prediksi Model")
            st.caption("Batang menunjukkan selisih return aktual dan prediksi model setiap hari. Batang di atas nol berarti return aktual lebih tinggi dari prediksi, dan di bawah nol berarti lebih rendah. Makin pendek batangnya, makin dekat prediksi dengan hasil aktual.")
            render_chart(delta_fig)
            return
        performance = research_file("portfolio_performance_summary.parquet")
        daily = research_file("portfolio_daily_returns_2025.parquet")
        quarterly = research_file("quarterly_returns_summary.parquet")
        daily["realized_date"] = pd.to_datetime(daily["realized_date"])
        daily = daily.sort_values("realized_date")
        model_col, benchmark_col = performance.columns[:2]
        # modal awal ikut dihitung, sama dengan grafik drawdown
        drawdowns = {}
        for name, returns_col in [(model_col, "return_xgb"), (benchmark_col, "return_hist")]:
            wealth = (1 + daily[returns_col]).cumprod()
            drawdowns[name] = float((wealth / wealth.cummax().clip(lower=1) - 1).min()) * 100

        def pair(row_name):
            """Nilai satu metrik untuk portofolio XGBoost dan pembanding."""
            return float(performance.loc[row_name, model_col]), float(performance.loc[row_name, benchmark_col])

        comparison_rows = [
            ("Return kumulatif", "Total keuntungan portofolio selama 2025. Makin tinggi makin baik.",
             *pair("Cumulative Return (%)"), "{:+.2f}%", True),
            ("Volatilitas tahunan", "Seberapa sering dan seberapa besar nilai portofolio naik turun dalam setahun. "
             "Makin rendah berarti pergerakannya makin tenang.",
             *pair("Annualized Volatility (%)"), "{:.2f}%", False),
            ("Max Drawdown", "Penurunan terdalam dari posisi tertinggi sebelum nilainya naik lagi. "
             "Makin mendekati nol berarti penurunannya makin dangkal. "
             "Contoh: dari Rp100 juta turun ke Rp85 juta, drawdown-nya −15%.",
             drawdowns[model_col], drawdowns[benchmark_col], "{:.2f}%", True),
            ("Sharpe ratio", "Keuntungan dibandingkan dengan risikonya. Mirip menilai pekerjaan dari gaji per jam, "
             "bukan total gajinya. Di atas 1 dianggap bagus, di bawah 0 berarti kalah dari menyimpan uang di deposito.",
             *pair("Sharpe Ratio"), "{:.3f}", True),
        ]
        # judul HTML supaya ikon tanda tanya selalu terlihat
        metric_help = (
            "Keempat ukuran ini dipakai untuk menilai apakah portofolio yang disusun dari prediksi model lebih baik "
            "daripada portofolio pembanding. Portofolio pembanding memakai Historical Mean, yaitu rata-rata return "
            "masa lalu tiap saham. Keduanya diuji pada saham dan periode "
            f"yang sama, yaitu {date_label(daily['realized_date'].min())} sampai "
            f"{date_label(daily['realized_date'].max())}.")
        # Menampilkan subjudul dan kartu metrik evaluasi portofolio
        st.markdown(f'<p class="solyn-sub-head">Metrik evaluasi portofolio'
                    f'<i class="solyn-hint" data-tip="{escape(metric_help)}">?</i></p>', unsafe_allow_html=True)
        # Menampilkan kartu perbandingan portofolio model vs pembanding (Return, Volatilitas, Drawdown, Sharpe)
        render_comparison_cards(comparison_rows)
        master = research_file("panel_calendar.parquet")
        bnli = master.loc[master["Kode Saham"] == "BNLI", ["snapshot_date", "close_adj"]].copy()
        bnli["snapshot_date"] = pd.to_datetime(bnli["snapshot_date"])
        bnli_2025 = bnli[bnli["snapshot_date"].dt.year == 2025].sort_values("snapshot_date")
        bnli_change = (bnli_2025["close_adj"].iloc[-1] / bnli_2025["close_adj"].iloc[0] - 1) * 100
        st.info(f"**Pengaruh saham BNLI**  \nHarga BNLI naik {bnli_change:.1f}% sepanjang 2025, sehingga sangat memengaruhi hasil portofolio.")
        cumulative, drawdown, quarter_chart = build_oos_charts(daily, quarterly)
        # Menampilkan subjudul dan grafik return kumulatif portofolio
        st.subheader("Perkembangan return kumulatif")
        st.caption("Return kumulatif adalah perubahan nilai yang terakumulasi sepanjang periode.")
        render_chart(cumulative)
        # Menampilkan subjudul dan grafik drawdown penurunan nilai portofolio
        st.subheader("Drawdown")
        st.caption("Penurunan nilai dari puncak portofolio sebelumnya.")
        render_chart(drawdown)
        # Menampilkan subjudul dan grafik batang return portofolio per kuartal
        st.subheader("Rincian per kuartal")
        render_chart(quarter_chart)
        st.caption("Hasil historis tidak menjamin hasil pada masa mendatang. Pembagian dana ini merupakan simulasi penelitian.")
    except Exception:
        LOGGER.exception("Research results unavailable")
        st.error("Hasil pengujian belum dapat dibaca. Silakan kembali ke analisis atau coba lagi nanti.")


# Halaman Unggah Data
def upload_widget_key():
    """Kunci kotak upload. Angkanya dinaikkan saat model sementara dihapus supaya kotaknya benar-benar kosong."""
    return f"market_uploads_{st.session_state.get('v3_upload_round', 0)}"


def sample_workbook():
    """Membuat contoh file Ringkasan Saham dari data bawaan, formatnya sama dengan file IDX."""
    history = load_local_market_history()
    date = history.groupby("Ticker")["Date"].max().min()
    frame = history[history["Date"] == date].copy()
    rename = {"Ticker":"Kode Saham", "Open":"Open Price", "High":"Tertinggi",
              "Low":"Terendah", "Close":"Penutupan"}
    frame = frame.rename(columns=rename)
    frame["Tanggal Perdagangan Terakhir"] = pd.Timestamp(date).strftime("%Y-%m-%d")
    buffer=BytesIO()
    frame[IDX_REQUIRED_COLUMNS].to_excel(buffer, index=False)
    return buffer.getvalue(), f"Ringkasan Saham-{pd.Timestamp(date):%Y%m%d}.xlsx"


# jeda antar tanggal upload paling lama 10 hari, untuk jaga-jaga hari libur atau suspensi bursa
IDX_SUMMARY_URL = "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham/"
UPLOAD_MAX_GAP_DAYS = 10
TRAINING_ERROR_MESSAGES = {
    "Data perdagangan kosong": "Tidak ada hari perdagangan yang dapat dipakai setelah data diperiksa. Periksa kolom harga dan Volume pada file.",
    "duplikat": "Terdapat saham dan tanggal yang sama lebih dari satu kali. Gunakan satu file untuk setiap tanggal.",
    "Harga atau volume kosong": "Terdapat harga atau Volume yang kosong. Gunakan file asli dari IDX tanpa diedit.",
    "Volume negatif": "Terdapat Volume bernilai negatif. Gunakan file asli dari IDX tanpa diedit.",
    "Close kalender": "Harga Penutupan harus terisi dan lebih dari nol untuk setiap saham.",
}


def training_error_message(error):
    """Mengubah error teknis waktu training jadi pesan yang gampang dipahami."""
    text = str(error)
    for key, message in TRAINING_ERROR_MESSAGES.items():
        if key in text:
            return message
    return "Model sementara belum dapat dibuat dari data ini. Periksa kembali tanggal dan isi file, lalu coba lagi."


def read_uploads_with_progress(uploaded, **limits):
    """Membaca file satu per satu sambil menampilkan progres."""
    bar = st.progress(0.0, text=f"Membaca file 0/{len(uploaded)}")

    def report(done, total, name):
        """Memperbarui progress bar saat file dibaca."""
        bar.progress(done / total if total else 1.0,
                     text=f"Membaca file {min(done + 1, total)}/{total}: {name}" if name else "Pemeriksaan file selesai")

    try:
        return parse_market_uploads(uploaded, on_progress=report, **limits)
    finally:
        bar.empty()


def date_range_label(dates):
    """Satu tanggal ditulis apa adanya, beberapa tanggal ditulis sebagai rentang supaya pesannya pendek."""
    return date_label(dates[0]) if len(dates) == 1 else f"{date_label(dates[0])} sampai {date_label(dates[-1])}"


def upload_date_errors(dates, last_date):
    """Tanggal upload harus setelah data utama dan tidak boleh ada jeda panjang.

    Args:
        dates: Tanggal dari nama file yang diunggah.
        last_date: Tanggal terakhir data utama aplikasi.

    Returns:
        Daftar pesan error. File yang tanggalnya sama dengan atau sebelum
        last_date ditolak, dan pemeriksaan berhenti di jeda pertama yang lebih
        dari UPLOAD_MAX_GAP_DAYS hari kalender.
    """
    errors = []
    dates = sorted(pd.Timestamp(date) for date in set(dates))
    not_newer = [date for date in dates if date <= last_date]
    if not_newer:
        errors.append(
            f"File bertanggal {date_range_label(not_newer)} tidak lebih baru dari data aplikasi. "
            f"Data aplikasi tersedia sampai {date_label(last_date)}, jadi gunakan file setelah tanggal tersebut."
        )
    new_dates = [date for date in dates if date > last_date]
    previous = last_date
    for date in new_dates:
        gap = (pd.Timestamp(date) - pd.Timestamp(previous)).days
        if gap > UPLOAD_MAX_GAP_DAYS:
            if previous == last_date:
                errors.append(
                    f"Data utama aplikasi baru tersedia sampai {date_label(last_date)}, sedangkan file Anda dimulai {date_label(date)} "
                    f"({gap} hari kemudian). Unggah file yang melanjutkan data utama, atau hubungi pengelola untuk memperbarui data."
                )
            else:
                errors.append(
                    f"Ada jeda {gap} hari antara {date_label(previous)} dan {date_label(date)}. "
                    "Lengkapi file untuk hari bursa di antaranya."
                )
            break
        previous = date
    return errors


GUIDE_DIR = BASE_DIR / "assets" / "panduan"

GUIDE_STEPS = [
    ("01_buka_idx.png", "Arahkan kursor ke Data Pasar, lalu pilih Ringkasan Saham."),
    ("02_pilih_tanggal.png", "Tekan kotak tanggal, lalu pilih tanggal hari bursa yang ingin diunduh."),
    ("03_tekan_kolom.png", 'Tekan tombol "Kolom", lalu tekan tombol "Pilih Semua" dan "Terapkan".'),
    ("04_unduh.png", "Tekan Unduh untuk mengunduh data tanggal tersebut."),
    ("05_cari_file.png", "Buka histori unduhan untuk menemukan filenya. Nama file jangan diganti."),
    ("06_unggah.png", "Unggah file hasil unduhan pada kotak yang disediakan aplikasi."),
    ("07_latih_model.png", 'Tekan tombol "Latih model sementara" untuk melatih model menggunakan data yang Anda unggah.'),
]


def move_guide_step(step):
    """Memindahkan gambar panduan satu langkah maju atau mundur."""
    position = st.session_state.get("v3_guide_step", 0) + step
    st.session_state["v3_guide_step"] = min(max(position, 0), len(GUIDE_STEPS) - 1)


def render_upload_guide():
    """Panduan bergambar dari GUIDE_STEPS, tampil satu gambar per halaman."""
    steps = [(GUIDE_DIR / name, caption) for name, caption in GUIDE_STEPS]
    total = len(steps)
    index = min(max(st.session_state.get("v3_guide_step", 0), 0), total - 1)
    picture, caption = steps[index]
    st.image(str(picture), use_container_width=True)
    st.markdown(f'<p class="solyn-guide-caption"><b>Langkah {index + 1}.</b> {escape(caption)}</p>',
                unsafe_allow_html=True)
    back, forward, _ = st.columns([1, 1, 2])
    back.button("Sebelumnya", key="v3_guide_back", disabled=index == 0, use_container_width=True,
                on_click=move_guide_step, args=(-1,))
    forward.button("Berikutnya", key="v3_guide_next", disabled=index == total - 1, use_container_width=True,
                   on_click=move_guide_step, args=(1,))


def render_upload():
    """Halaman Unggah Data, pengguna upload file Ringkasan Saham IDX terbaru."""
    # Menampilkan judul dan deskripsi halaman Unggah Data
    page_heading("Unggah data saham",
                 "Unggah file Ringkasan Saham untuk melatih model sementara. Untuk analisis rutin, data aplikasi sudah tersedia.")

    history = load_local_market_history()
    last_date = pd.Timestamp(history["Date"].max())
    active = current_session_model()
    last_text = date_label(last_date)
    # Menampilkan panduan bergambar langkah mengunduh file dari situs IDX
    st.subheader("Panduan Mengunduh Data")
    st.markdown(f"[Situs Bursa Efek Indonesia]({IDX_SUMMARY_URL})")
    render_upload_guide()
    # Menampilkan subjudul dan kotak unggah file Ringkasan Saham
    st.subheader("Unggah file Ringkasan Saham")
    next_text = date_label(last_date + pd.offsets.BDay(1))
    st.caption(f"Data aplikasi tersedia sampai {last_text}. Unggah file untuk tanggal {next_text} dan setelahnya.")
    # Menampilkan widget file uploader untuk memilih file Excel (.xlsx)
    uploaded = st.file_uploader("Pilih satu atau beberapa file Ringkasan Saham", type=["xlsx"], accept_multiple_files=True, key=upload_widget_key())
    if not uploaded:
        # Menampilkan kartu status model sementara jika sudah pernah dilatih sebelumnya
        render_session_model_card(active)
        example, name = sample_workbook()
        st.subheader("Punya data dari sumber lain?")
        st.markdown("Template berikut memperlihatkan susunan kolom yang harus diikuti agar data Anda dapat dibaca "
                    "aplikasi. Kalau file Anda diunduh dari situs IDX, template ini tidak diperlukan.")
        # Menampilkan tombol unduh file template Excel contoh format IDX
        st.download_button("Unduh Template", example, name,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        return
    data, errors = read_uploads_with_progress(uploaded, check_dates=lambda dates: upload_date_errors(dates, last_date))
    if errors:
        render_upload_problems(errors, clear_key="v3_upload_round")
        return
    render_skipped_files(data)
    if data["Date"].max() <= last_date:
        st.info(
            f"Semua file bertanggal sampai {date_label(last_date)} atau sebelumnya. Data ini menggantikan data utama pada tanggal yang sama, "
            "tetapi tidak menambah hari perdagangan baru."
        )
    # Menampilkan subjudul dan tabel ringkasan file data yang diunggah
    st.subheader("Periksa data sebelum melanjutkan")
    st.write(f"**{data['Source File'].nunique()} file** · **{data['Ticker'].nunique()} saham** · {date_label(data['Date'].min())} – {date_label(data['Date'].max())}")
    file_summary = (data.groupby(["Source File", "Date"])["Ticker"]
                    .agg(lambda codes: ", ".join(sorted(codes))).reset_index()
                    .rename(columns={"Source File": "File", "Date": "Tanggal", "Ticker": "Saham terbaca"}))
    file_summary["Tanggal"] = file_summary["Tanggal"].map(date_label)
    st.dataframe(file_summary, hide_index=True, use_container_width=True)
    upload_tickers = [ticker for ticker in PORTFOLIO_TICKERS if ticker in set(data["Ticker"])]
    # Menampilkan input multiselect pilihan saham yang akan diproses
    selected = st.multiselect("Saham yang diproses", upload_tickers, default=upload_tickers, key="v3_upload_selected")
    # Menampilkan expander tabel sampel data harga penutupan dan volume
    with st.expander("Sampel data"):
        sample = data.sort_values(["Date", "Ticker"]).head(len(PORTFOLIO_TICKERS))
        number = lambda value: f"{value:,.0f}".replace(",", ".")
        rows = "".join(f'<tr><td>{escape(date_label(row.Date))}</td><td><strong>{escape(str(row.Ticker))}</strong></td>'
                       f'<td class="num">{number(row.Close)}</td><td class="num">{number(row.Volume)}</td></tr>'
                       for row in sample.itertuples())
        st.markdown('<div class="solyn-table-wrap"><table class="solyn-table"><thead><tr><th>Tanggal</th><th>Saham</th>'
                    '<th class="num">Harga penutupan (Rp)</th><th class="num">Volume</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    # Menampilkan tombol untuk melatih model sementara dari data yang diunggah
    if st.button("Latih model sementara", type="primary", disabled=not selected):
        slot = training_slot()
        if not slot.acquire(blocking=False):
            st.toast("Pelatihan lain sedang berjalan, coba lagi setelah selesai")
            return
        try:
            with st.status("Memeriksa data...", expanded=True) as status:
                progress = st.progress(0, text="Menyiapkan riwayat saham")
                combined = pd.concat([history[history["Ticker"].isin(selected)], data[data["Ticker"].isin(selected)]], ignore_index=True)
                combined = combined.sort_values(["Ticker","Date"]).drop_duplicates(["Ticker","Date"],keep="last").reset_index(drop=True)
                prepared = prepare_market_data(combined)
                status.update(label="Menghitung indikator dan melatih model...")
                features = calculate_features(prepared)
                def report(done, total, ticker):
                    """Memperbarui progress bar training per saham."""
                    progress.progress(done/total, text=f"Memproses {ticker} · {done}/{total} saham selesai")
                result = train_baseline_models(features, selected, on_progress=report)
                result.update({"version":datetime.now(timezone.utc).isoformat(),
                    "source":"session_upload", "market_data_date":prepared['Date'].max().strftime('%Y-%m-%d'),
                    "session_history":prepared})
                result["portfolio"] = build_portfolio_allocation(prepared, result)
                st.session_state["session_model_result"] = result
                st.session_state["v3_source"] = "session"
                st.session_state.pop("v3_source_widget", None)
                st.session_state.pop("v3_prediction", None)
                st.session_state.pop("v3_allocation", None)
                status.update(label="Model sementara siap digunakan", state="complete", expanded=False)
            st.session_state["v3_scroll_to_session_card"] = True
            st.rerun()
        except CorporateActionSafetyError as error:
            st.warning(
                f"Data belum dapat dipakai: {error} Pastikan file berasal langsung dari IDX dan tidak diedit. "
                "Jika ada aksi korporasi baru, tunggu pengelola memperbarui data utama."
            )
        except ValueError as error:
            LOGGER.warning("Session training rejected: %s", error)
            st.warning(training_error_message(error))
        except Exception:
            LOGGER.exception("Session training failed")
            st.error(
                "Model sementara belum berhasil dibuat karena kesalahan yang tidak terduga. Coba kembali beberapa saat lagi. "
                "Jika masalah berulang, hubungi pengelola dan sebutkan nama file yang diunggah. Model sementara sebelumnya tetap tersedia jika ada."
            )
        finally:
            slot.release()
    # Menampilkan kartu status hasil pelatihan model sementara
    render_session_model_card(active)


def render_session_model_card(active):
    """Kartu hasil pelatihan model sementara, tampil tepat di bawah tombol Latih."""
    if not active:
        return
    with st.container(border=True, key="solyn_session_card"):
        # Menampilkan pesan sukses bahwa model berhasil dilatih ulang
        st.success("Model berhasil dilatih ulang menggunakan data unggahan Anda. Tekan tombol di bawah "
                   "untuk melihat prediksi atau membuat simulasi alokasi portofolio dari data Anda.")
        # Menampilkan dua tombol navigasi untuk melihat prediksi atau membuka simulasi
        with st.container(horizontal=True, gap="small"):
            st.button("Lihat prediksi dari data ini", type="primary", on_click=set_view, args=("analysis",))
            st.button("Buka simulasi dari data ini", on_click=set_view, args=("portfolio",))
    if st.session_state.pop("v3_scroll_to_session_card", False):
        # geser halaman ke kartu hasil
        import streamlit.components.v1 as components
        with st.container(key="solyn_scroll_card"):
            components.html("""<script>
setTimeout(() => {
    const card = window.parent.document.querySelector('.st-key-solyn_session_card');
    if (card) card.scrollIntoView({behavior: 'smooth', block: 'center'});
}, 300);
</script>""", height=0)


# Halaman Tentang
MANUAL_PATH = BASE_DIR / "assets" / "Buku_Manual_Solyn.pdf"


def render_about():
    """Halaman Tentang: tujuan aplikasi, cara penggunaan, pembuat, dan dosen pembimbing."""
    # Menampilkan judul dan deskripsi halaman Tentang
    page_heading("Tentang aplikasi",
                 "Tujuan aplikasi, cara penggunaan, serta pembuat dan pembimbing skripsi.")
    # Menampilkan tombol unduh Buku Manual aplikasi dalam bentuk PDF jika ada
    if MANUAL_PATH.exists():
        st.download_button("Unduh Buku Manual (PDF)", MANUAL_PATH.read_bytes(), MANUAL_PATH.name, "application/pdf")
    # Menampilkan subjudul dan deskripsi tujuan aplikasi Solyn
    st.subheader("Apa itu Solyn?")
    st.write("Aplikasi berbasis web untuk memprediksi return harian 11 saham perbankan "
             "di Bursa Efek Indonesia. Prediksi dibuat dengan model XGBoost dari sepuluh indikator teknikal, lalu "
             "dipakai untuk menyimulasikan pembagian dana portofolio dengan metode Mean-Variance Markowitz. "
             "Data berasal dari Ringkasan Saham IDX yang disesuaikan terhadap stock split. Hasil aplikasi adalah "
             "perkiraan model dan simulasi penelitian, bukan saran investasi.")
    # Menampilkan subjudul dan ringkasan fungsi masing-masing halaman
    st.subheader("Penjelasan halaman")
    sections = {
        "Home": "Lihat harga terakhir dan tanggal data. Pilih Prediksi Saham atau Simulasi Portofolio untuk mulai.",
        "Prediksi Saham": "Pilih saham lalu tekan Buat prediksi. Return adalah persentase perubahan harga saham. Periksa tanggal target dan indikator yang memengaruhi hasil. Gunakan Skenario Sendiri bila ingin mencoba nilai indikator lain.",
        "Simulasi Portofolio": "Pilih saham, masukkan modal, lalu tekan Hitung alokasi. Hasil menunjukkan bobot dan nominal pembagian dana. Dengan batas 20% per saham, pilih minimal 5 saham. Setelah mengubah masukan, hitung kembali.",
        "Unggah Data": "Unduh file Ringkasan Saham harian dari situs IDX tanpa mengubah nama atau isinya, lalu unggah maksimal 30 file sekaligus. Tanggal file harus melanjutkan data aplikasi. Setelah lolos pemeriksaan, pilih saham dan tekan Latih model sementara. Model ini hanya tersimpan selama sesi Anda.",
        "Hasil Pengujian": "Pilih Evaluasi Portofolio atau Evaluasi Model. Baca empat ukuran utama di bagian atas, lalu lihat grafik di bawahnya. Penjelasan setiap ukuran tersedia pada tanda tanya di sampingnya.",
    }
    for heading, body in sections.items():
        st.markdown(f"**{heading}**")
        st.write(body)
    # Menampilkan catatan penting mengenai batasan hasil penelitian
    st.subheader("Perlu diingat")
    st.markdown("- Hasil pengujian tahun 2025 tidak menjamin hasil yang sama di masa depan.\n"
                "- Simulasi portofolio belum menghitung biaya transaksi seperti fee broker dan pajak.")
    # Menampilkan kartu profil mahasiswa pembuat aplikasi
    st.subheader("Pembuat")
    st.markdown(f'''<div class="solyn-about-card">
<div><span>Nama</span><strong>{escape(AUTHOR["nama"])}</strong></div>
<div><span>NIM</span><strong>{escape(AUTHOR["nim"])}</strong></div>
<div><span>Fakultas</span><strong>{escape(AUTHOR["fakultas"])}</strong></div>
<div><span>Universitas</span><strong>{escape(AUTHOR["universitas"])}</strong></div>
</div>''', unsafe_allow_html=True)
    # Menampilkan daftar nama dosen pembimbing skripsi
    st.subheader("Dosen pembimbing")
    st.markdown('<ol class="solyn-about-list">' + "".join(f"<li>{escape(name)}</li>" for name in SUPERVISORS)
                + "</ol>", unsafe_allow_html=True)


# Halaman Admin
def format_approval_time(value):
    """Format waktu izin akses buat tabel admin."""
    if not str(value).strip():
        return ""
    try:
        date_time = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if date_time.tzinfo is None:
            return str(value)
        return date_time.astimezone(timezone(timedelta(hours=7))).strftime("%d/%m/%Y %H:%M WIB")
    except ValueError:
        return str(value)


def require_admin_action(expected_email):
    """Memastikan aksi admin dijalankan akun admin yang sama dan masih aktif."""
    actor = current_identity()
    if not actor or not actor.get("is_admin") or actor["email"] != expected_email:
        raise PermissionError("Tindakan memerlukan akun administrator yang aktif.")
    return actor


def baseline_idx_history():
    """Data mentah Ringkasan Saham IDX yang jadi dasar model bawaan."""
    path = BASE_DIR / DEFAULT_BASELINE_PATH
    with ZipFile(path) as archive:
        return pd.read_csv(BytesIO(archive.read("idx_history.csv")), parse_dates=["snapshot_date"])


def uploads_to_idx_rows(uploaded_market_data):
    """Mengubah hasil baca upload ke nama kolom Ringkasan Saham IDX."""
    frame = uploaded_market_data.rename(columns={
        "Ticker": "Kode Saham", "Date": "snapshot_date", "Open": "Open Price",
        "High": "Tertinggi", "Low": "Terendah", "Close": "Penutupan",
    })
    return frame[["Kode Saham", "snapshot_date", "Open Price", "Tertinggi", "Terendah",
                  "Penutupan", "Volume", "Listed Shares", "Remarks"]].copy()


def baseline_github_settings():
    """Pengaturan [github] dari Secrets buat menyimpan model bawaan ke repo."""
    return github_settings(get_secret_section("github"))


def write_local_bundle(bundle):
    """Menulis bundle baru ke server lewat file sementara biar file lama tidak rusak."""
    path = BASE_DIR / DEFAULT_BASELINE_PATH
    temporary_path = path.with_suffix(".manual.tmp")
    try:
        temporary_path.write_bytes(bundle)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_manual_baseline(uploaded_market_data, expected_email):
    """Melatih ulang 11 model dari data IDX dan menyimpan bundle baru.

    Bundle selalu disimpan ke file lokal, dan di-commit ke GitHub jika
    pengaturan GitHub tersedia.

    Args:
        uploaded_market_data: Hasil parse_market_uploads dari upload admin.
        expected_email: Email admin yang memulai proses, dicek ulang sebelum training.

    Returns:
        Tuple (tanggal data terbaru, jumlah tanggal baru, URL commit GitHub atau teks kosong).
    """
    actor = require_admin_action(expected_email)
    combined_raw, new_dates = append_idx_rows(baseline_idx_history(), uploads_to_idx_rows(uploaded_market_data))
    bundle, info = build_baseline_bundle(
        combined_raw, actor["email"], "Ringkasan Saham IDX, diperbarui oleh admin"
    )
    commit_url = ""
    settings = baseline_github_settings()
    if github_is_configured(settings):
        commit_url = publish_bundle(
            bundle, f"Perbarui model bawaan sampai {info['market_data_date']}", settings
        )
    write_local_bundle(bundle)
    return pd.Timestamp(new_dates[-1]).date(), len(new_dates), commit_url


def render_admin_monitoring():
    """Panel admin: status model bawaan, jumlah model, tanggal data, dan waktu training."""
    st.subheader("Status aplikasi")
    with ZipFile(BASE_DIR / DEFAULT_BASELINE_PATH) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        model_count = len([name for name in archive.namelist() if name.startswith("models/") and name.endswith(".json")])
    st.markdown(
        '<div class="solyn-admin-facts">'
        f'<span>Model siap <strong>{model_count}/{len(PORTFOLIO_TICKERS)}</strong></span>'
        f'<span>Data perdagangan terakhir <strong>{escape(date_label(manifest.get("market_data_date")))}</strong></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.write("**Pelatihan terakhir:** " + timestamp_label(manifest.get("training_completed_at")))
    st.caption("Sumber: " + str(manifest.get("source", "belum tercatat")))
    st.caption("Data dan model diperbarui melalui unggahan file Ringkasan Saham IDX pada bagian di bawah.")


def _render_admin_access_body(identity):
    """Panel admin: daftar permintaan akses dan tombol approve atau tolak."""
    st.subheader("Akses pengguna")
    toast = st.session_state.pop("v3_admin_toast", None)
    if toast:
        st.toast(toast)
    if not approval_is_configured():
        st.warning("Pengelola perlu melengkapi konfigurasi persetujuan dan Google Sheets sebelum menyetujui pengguna.")
        return
    try:
        if st.button("Muat ulang daftar"):
            load_approval_records.clear()
        records = pd.DataFrame(load_approval_records(), columns=APPROVAL_HEADERS)
        if records.empty:
            st.info("Belum ada permintaan akses.")
            return
        summary_slot = st.empty()
        search_col, status_col = st.columns([2, 1])
        search = search_col.text_input("Cari pengguna", placeholder="Nama atau email", key="v3_admin_user_search").strip()
        status_filter = status_col.selectbox("Filter status", ["Semua", "Menunggu", "Disetujui", "Ditolak"], key="v3_admin_user_status")
        def visible_records(source):
            """Menyaring akun sesuai filter dan pencarian, yang menunggu persetujuan di atas."""
            visible = source.copy()
            if status_filter != "Semua":
                wanted = {"Menunggu": "pending", "Disetujui": "approved", "Ditolak": "rejected"}[status_filter]
                visible = visible[visible["Status"].str.strip().str.lower() == wanted]
            if search:
                match = (visible["Nama"].str.contains(search, case=False, regex=False, na=False)
                         | visible["Email"].str.contains(search, case=False, regex=False, na=False))
                visible = visible[match]
            visible = visible.copy().reset_index(drop=True)
            visible["_priority"] = visible["Status"].str.strip().str.lower().map(
                {"pending": 0, "approved": 1, "rejected": 2}
            ).fillna(3)
            return visible.sort_values(["_priority", "Tanggal Daftar"], ascending=[True, False])

        filtered = visible_records(records)
        table_slot = st.empty()
        owners = owner_emails()
        if filtered.empty:
            st.info("Tidak ada pengguna yang cocok.")
        else:
            selected = st.selectbox("Pilih akun untuk diubah", filtered["Email"].tolist())
            row = records[records["Email"] == selected].iloc[0]
            # akun pemilik dan akun sendiri tidak bisa dicabut
            protected = selected.lower() in owners or selected.lower() == identity["email"]
            is_admin_row = selected.lower() in owners or role_of(row) == "Admin"
            reject_confirm = st.checkbox("Saya ingin menolak atau mencabut akses akun yang dipilih",
                                         key="v3_reject_"+selected, disabled=protected)
            a, b, c = st.columns(3)
            approve = a.button("Setujui akses", type="primary", use_container_width=True)
            reject = b.button("Tolak / cabut akses", disabled=protected or not reject_confirm, use_container_width=True)
            if is_admin_row:
                make_role = "User"
                role_button = c.button("Jadikan user", disabled=protected, use_container_width=True)
            else:
                make_role = "Admin"
                role_button = c.button("Jadikan admin", use_container_width=True)
            if protected:
                st.caption("Akses dan role admin pemilik aplikasi serta akun Anda sendiri tidak dapat dicabut.")
            if approve or reject or role_button:
                require_admin_action(identity["email"])
                try:
                    if role_button:
                        change_user_role(selected, make_role)
                        message = f"Role {selected} diubah menjadi {make_role}."
                    else:
                        review_approval_request(selected, "Approved" if approve else "Rejected")
                        message = "Status akses diperbarui."
                    st.session_state["v3_admin_toast"] = message
                    st.rerun(scope="fragment")
                except ValueError as error:
                    st.toast(str(error))
        with st.expander("Tambah pengguna"):
            st.caption("Email yang didaftarkan di sini langsung disetujui, tanpa masa coba gratis.")
            new_email = st.text_input("Email Google", placeholder="nama@gmail.com", key="v3_admin_new_email")
            new_name = st.text_input("Nama", key="v3_admin_new_name")
            new_role = st.selectbox("Role", ["User", "Admin"], key="v3_admin_new_role")
            if st.button("Tambah pengguna", key="v3_admin_add_user", disabled=not new_email.strip()):
                require_admin_action(identity["email"])
                try:
                    register_user(new_email, new_name, new_role)
                    st.session_state["v3_admin_toast"] = f"{new_email.strip().lower()} ditambahkan sebagai {new_role}."
                    st.rerun(scope="fragment")
                except ValueError as error:
                    st.toast(str(error))

        statuses = records["Status"].str.strip().str.lower()
        roles = pd.Series(["Admin" if r["Email"].lower() in owners or role_of(r) == "Admin" else "User"
                           for _, r in records.iterrows()], dtype=object)
        with summary_slot.container():
            st.markdown(
                '<div class="solyn-admin-counts">'
                f'<span><strong>{int((statuses == "pending").sum())}</strong> menunggu</span>'
                f'<span><strong>{int((statuses == "approved").sum())}</strong> disetujui</span>'
                f'<span><strong>{int((statuses == "rejected").sum())}</strong> ditolak</span>'
                f'<span><strong>{int(roles.eq("Admin").sum())}</strong> admin</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        display_records = filtered[["Nama", "Email", "Status", "Tanggal Daftar", TRIAL_COLUMN]].copy()
        display_records.insert(3, "Role", ["Admin (pemilik)" if r["Email"].lower() in owners else role_of(r)
                                           for _, r in filtered.iterrows()])
        display_records["Status"] = display_records["Status"].str.strip().str.lower().map(
            {"pending": "Menunggu", "approved": "Disetujui", "rejected": "Ditolak", "": "Belum mengajukan"}
        ).fillna(display_records["Status"])
        display_records["Tanggal Daftar"] = display_records["Tanggal Daftar"].map(format_approval_time)
        display_records[TRIAL_COLUMN] = display_records[TRIAL_COLUMN].map(
            lambda value: format_approval_time(value) if str(value).strip() else "Belum")
        with table_slot.container():
            # tinggi tabel mengikuti jumlah baris
            st.dataframe(display_records, hide_index=True, use_container_width=True,
                         height=min(250, 35 * (len(display_records) + 1) + 3))
    except Exception:
        LOGGER.exception("Admin access management failed")
        st.error("Daftar akses belum dapat diperbarui. Coba muat ulang daftar.")


@st.fragment
def render_admin_access(identity):
    """Bagian akses pengguna yang refresh sendiri tanpa reload seluruh halaman."""
    actor = current_identity()
    if not actor or not actor.get("is_admin") or actor["email"] != identity["email"]:
        st.error("Halaman ini hanya tersedia untuk administrator.")
        return
    with st.container(key="v3_admin_access"):
        _render_admin_access_body(actor)


def render_admin_baseline(identity):
    """Panel admin buat update model bawaan pakai file Ringkasan Saham IDX terbaru."""
    st.subheader("Pembaruan data dan model")
    st.caption(
        "Unggah file Ringkasan Saham IDX untuk tanggal setelah data terakhir. Setiap file harus memuat 11 saham. "
        "Aplikasi menyesuaikan stock split, menghitung indikator, dan melatih ulang 11 model dengan hyperparameter penelitian."
    )
    settings = baseline_github_settings()
    if github_is_configured(settings):
        st.caption(f"Model baru disimpan ke GitHub: {settings['repository']} · branch {settings['branch']}.")
    else:
        st.warning(
            "Penyimpanan GitHub belum diatur di Secrets. Pembaruan hanya berlaku di server ini dan akan hilang saat aplikasi dimuat ulang."
        )
    uploaded = st.file_uploader(
        "Pilih file Ringkasan Saham IDX", type=["xlsx"], accept_multiple_files=True,
        key=f"v3_admin_idx_files_{st.session_state.get('v3_admin_upload_round', 0)}"
    )
    st.caption(f"Maksimal {MAX_ADMIN_UPLOAD_FILES} file sekaligus.")
    parsed = None
    preview_ok = False
    if uploaded:
        parsed, errors = read_uploads_with_progress(
            uploaded, max_files=MAX_ADMIN_UPLOAD_FILES,
            max_total_bytes=MAX_ADMIN_UPLOAD_TOTAL_BYTES, repair_prices=False,
        )
        if errors:
            render_upload_problems(errors, clear_key="v3_admin_upload_round")
        else:
            render_skipped_files(parsed)
            try:
                existing = baseline_idx_history()
                _, new_dates = append_idx_rows(existing, uploads_to_idx_rows(parsed))
                last_date = pd.Timestamp(existing["snapshot_date"].max())
                gap_days = (pd.Timestamp(new_dates[0]) - last_date).days
                preview_ok = True
                st.success(
                    f"{len(new_dates)} tanggal baru siap diproses: {date_label(new_dates[0])} sampai {date_label(new_dates[-1])}."
                )
                if gap_days > 7:
                    st.warning(
                        f"Tanggal baru pertama berjarak {gap_days} hari dari data terakhir ({date_label(last_date)}). "
                        "Pastikan tidak ada hari bursa yang terlewat."
                    )
            except ValueError as error:
                st.toast(str(error))
            except Exception:
                LOGGER.exception("Admin IDX preview failed")
                st.error("File belum dapat diperiksa. Coba unggah ulang file IDX.")
    if st.button("Latih ulang dan simpan model", type="primary", key="v3_admin_publish_idx", disabled=not preview_ok):
        slot = training_slot()
        if not slot.acquire(blocking=False):
            st.toast("Pelatihan lain sedang berjalan, coba lagi setelah selesai")
            return
        try:
            with st.spinner("Menghitung indikator, melatih 11 model, dan menyimpan hasil."):
                date, count, commit_url = publish_manual_baseline(parsed, identity["email"])
            message = f"Model bawaan diperbarui sampai {date_label(date)} ({count} tanggal baru)."
            if commit_url:
                message += " Tersimpan di GitHub. Aplikasi akan dimuat ulang otomatis dalam 1 sampai 2 menit."
            st.session_state["v3_admin_flash"] = message
            clear_uploads("v3_admin_upload_round")
            st.rerun()
        except (ValueError, GitHubStoreError) as error:
            st.toast(str(error))
        except PermissionError as error:
            st.error(str(error))
        except Exception:
            LOGGER.exception("Admin IDX update failed")
            st.error("Pembaruan gagal. Model sebelumnya tetap dipakai.")
        finally:
            slot.release()


def render_admin():
    """Halaman Admin, khusus email admin."""
    identity = current_identity()
    if not identity.get("is_admin"):
        st.error("Halaman ini hanya tersedia untuk administrator.")
        return
    st.title("Panel Admin")
    flash = st.session_state.pop("v3_admin_flash", None)
    if flash:
        st.success(flash)
    with st.container(key="v3_admin_status"):
        render_admin_monitoring()
    with st.container(key="v3_admin_baseline"):
        render_admin_baseline(identity)
    render_admin_access(identity)


# Timer coba gratis dan fungsi utama
@st.fragment(run_every=5)
def render_trial_timer():
    """Menghitung mundur coba gratis tiap 5 detik, logout otomatis walaupun pengguna diam saja."""
    trial = trial_state()
    if not trial:
        enforce_trial_expiry()
        if st.session_state.get(TRIAL_NOTICE_KEY) or st.session_state.get(APPROVED_NOTICE_KEY):
            st.rerun(scope="app")
        return
    left = trial_seconds_left(trial)
    if left <= 0:
        enforce_trial_expiry()
        st.rerun(scope="app")
    minutes, seconds = divmod(left, 60)
    st.markdown(
        f'<div class="solyn-trial-timer">Sisa waktu percobaan <strong>{minutes:02d}:{seconds:02d}</strong></div>',
        unsafe_allow_html=True,
    )
    if left <= 60:
        st.caption("Waktu hampir habis. Simpan hasil yang Anda perlukan.")


def set_page_language():
    """Menandai halaman sebagai bahasa Indonesia (lang="id") supaya pembaca layar memakai pelafalan yang benar."""
    import streamlit.components.v1 as components
    with st.container(key="solyn_page_lang"):
        components.html("<script>window.parent.document.documentElement.lang = 'id';</script>", height=0)


def main():
    """Fungsi utama, dijalankan ulang Streamlit setiap pengguna berinteraksi."""
    set_page_language()
    enforce_trial_expiry()
    identity = current_identity()
    view = st.query_params.get("view", "home")
    if view not in VIEWS:
        view = "home"
    if identity is None:
        render_login_page()
        return
    status = access_status(identity)
    if status in {"new", "pending"} and trial_available() and not trial_record_window(identity["email"]):
        if start_trial(identity, view) == "started":
            st.rerun()
    if status != "approved":
        render_access_page(identity)
        return
    with st.sidebar:
        encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        st.markdown(f'<div class="v3-logo"><img src="data:image/png;base64,{encoded_logo}" alt="Solyn"></div>', unsafe_allow_html=True)
        st.markdown('<div class="solyn-brand-note">Riset saham perbankan</div>',unsafe_allow_html=True)
        def nav_buttons(destinations):
            """Tombol navigasi, halaman yang sedang dibuka diberi warna utama."""
            for destination in destinations:
                st.button(VIEWS[destination], type="primary" if view == destination else "secondary",
                          key="v3_nav_" + destination, use_container_width=True,
                          on_click=set_view, args=(destination,))

        with st.container(key="v3_nav"):
            nav_buttons(["home", "analysis", "portfolio", "upload"])
        with st.container(key="v3_sidebar_footer"):
            with st.container(key="v3_info"):
                bottom = ["performance", "about"]
                if identity.get("is_admin"):
                    bottom.append("admin")
                nav_buttons(bottom)
            render_data_sidebar()
            with st.container(key="v3_account"):
                if identity.get("trial"):
                    render_trial_timer()
                    with st.popover(identity["name"], use_container_width=True):
                        st.write(identity["email"])
                        st.caption("Sedang memakai coba gratis.")
                        st.button("Keluar", key="v3_logout", on_click=sign_out_v3, use_container_width=True)
                else:
                    with st.popover(identity["name"], use_container_width=True):
                        st.write(identity["email"])
                        st.button("Keluar", key="v3_logout", on_click=sign_out_v3, use_container_width=True)
                    if local_approval_bypass(identity):
                        st.caption("Masuk lokal")
    if st.session_state.get("v3_rendered_view") != view:
        # scroll ke atas hanya saat pindah halaman
        import streamlit.components.v1 as components
        with st.container(key="solyn_page_focus"):
            components.html('''<script>
requestAnimationFrame(() => {
    const main = window.parent.document.querySelector('[data-testid="stMain"]');
    if (main) main.scrollTo({top: 0, behavior: 'instant'});
});
</script>''' + "<!-- " + view + " -->", height=0)
        st.session_state["v3_rendered_view"] = view
    st.markdown(f'<div class="solyn-topline"><span>Solyn &nbsp; / &nbsp; {escape(VIEWS[view])}</span><span>Riset & simulasi · NFA</span></div>',unsafe_allow_html=True)
    routes = {"home":render_home,"analysis":render_analysis,"portfolio":render_portfolio,
              "performance":render_performance,"upload":render_upload,"about":render_about,"admin":render_admin}
    notice = st.session_state.pop(TRIAL_NOTICE_KEY, None)
    if notice:
        st.toast(notice)
        st.info(notice)
    if st.session_state.pop(APPROVED_NOTICE_KEY, None):
        st.toast("Akses Anda sudah disetujui pengelola.")
        st.success("Akses Anda sudah disetujui pengelola. Semua fitur dapat dipakai tanpa batas waktu.")
    routes[view]()
    render_nfa()


if __name__ == "__main__":
    main()
