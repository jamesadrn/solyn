"""Membuat bundle model bawaan aplikasi dari data Ringkasan Saham IDX.

Pengolahan datanya sama dengan notebook 02 dan hyperparameternya hasil validasi 2024
dari notebook 03 (MODEL_PARAMS). Model akhir dilatih di semua data yang sudah punya
target, lalu dipakai prediksi return hari bursa berikutnya.
"""
from datetime import datetime, timezone

import pandas as pd

from application.modeling import (
    PORTFOLIO_TICKERS,
    build_portfolio_allocation,
    create_baseline_bundle,
    read_baseline_bundle,
    train_baseline_models,
)
from feature_pipeline import RAW_COLUMNS, build_feature_panel
from model_training_metadata import add_training_metadata

# 20 persen data terakhir dipakai buat menghitung metrik test model bawaan
EVALUATION_RATIO = 0.20


def normalize_raw_history(raw):
    """Memastikan data mentah IDX cuma berisi kolom dan saham yang dipakai."""
    frame = raw.copy()
    if "Remarks" not in frame:
        frame["Remarks"] = "--"
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom riwayat IDX belum lengkap: {missing}")
    frame = frame.loc[frame["Kode Saham"].isin(PORTFOLIO_TICKERS), RAW_COLUMNS].copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
    frame["Remarks"] = frame["Remarks"].fillna("--").astype(str).str.strip()
    numeric = ["Open Price", "Tertinggi", "Terendah", "Penutupan", "Volume", "Listed Shares"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if frame.duplicated(["Kode Saham", "snapshot_date"]).any():
        raise ValueError("Riwayat IDX memuat saham dan tanggal duplikat.")
    counts = frame.groupby("snapshot_date")["Kode Saham"].nunique()
    incomplete = counts[counts != len(PORTFOLIO_TICKERS)]
    if not incomplete.empty:
        dates = ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in incomplete.index[:5])
        raise ValueError(f"Setiap tanggal harus memuat 11 saham. Belum lengkap: {dates}.")
    return frame.sort_values(["Kode Saham", "snapshot_date"]).reset_index(drop=True)


def market_history_from_calendar(calendar):
    """Format riwayat harga yang dibaca halaman aplikasi.

    Harga sudah disesuaikan stock split, jadi Remarks diisi "--".
    Kode Remarks aslinya tetap ada di data mentah IDX.
    """
    history = pd.DataFrame({
        "Date": calendar["snapshot_date"],
        "Ticker": calendar["Kode Saham"],
        "Open": calendar["open_adj"],
        "High": calendar["high_adj"],
        "Low": calendar["low_adj"],
        "Close": calendar["close_adj"],
        "Volume": calendar["volume_adj"],
        "Listed Shares": calendar["Listed Shares"],
        "Remarks": "--",
    })
    return history.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def build_baseline_bundle(raw_history, published_by, source):
    """Hasilnya bytes baseline_bundle.zip plus ringkasan training."""
    # urutannya: merapikan data IDX, menghitung fitur seperti notebook 02, melatih 11 model,
    # menghitung portofolio, lalu menyimpan semuanya ke satu file zip
    raw = normalize_raw_history(raw_history)
    started_at = datetime.now(timezone.utc).isoformat()
    calendar, all_features, labelled, split_events = build_feature_panel(raw, PORTFOLIO_TICKERS)
    features = all_features.rename(columns={"Kode Saham": "Ticker", "snapshot_date": "Date"})
    result = train_baseline_models(features, PORTFOLIO_TICKERS, EVALUATION_RATIO)
    if set(result["models"]) != set(PORTFOLIO_TICKERS):
        raise RuntimeError("Pelatihan belum menghasilkan 11 model lengkap.")
    history = market_history_from_calendar(calendar)
    result["portfolio"] = build_portfolio_allocation(history, result)
    last_date = pd.Timestamp(raw["snapshot_date"].max())
    result.update({
        "source": source,
        "market_data_date": last_date.strftime("%Y-%m-%d"),
        "training_data_start": pd.Timestamp(raw["snapshot_date"].min()).strftime("%Y-%m-%d"),
        "training_data_end": last_date.strftime("%Y-%m-%d"),
        "market_history": history,
        "idx_history": raw,
    })
    completed_at = datetime.now(timezone.utc).isoformat()
    bundle = create_baseline_bundle(result, {"email": published_by})
    bundle = add_training_metadata(bundle, started_at, completed_at)
    # mengecek terakhir, zip baru harus bisa dibaca dan isinya 11 saham
    _, _, manifest = read_baseline_bundle(bundle, load_models=False)
    if set(manifest.get("tickers", [])) != set(PORTFOLIO_TICKERS):
        raise RuntimeError("Bundle baru belum memuat 11 saham.")
    return bundle, {
        "market_data_date": result["market_data_date"],
        "labelled_rows": int(len(labelled)),
        "split_events": split_events,
        "summary": result["summary"],
    }


def append_idx_rows(existing_raw, uploaded_raw):
    """Menambahkan tanggal baru dari upload admin ke data mentah IDX."""
    existing = normalize_raw_history(existing_raw)
    incoming = normalize_raw_history(uploaded_raw)
    last_date = existing["snapshot_date"].max()
    new_rows = incoming[incoming["snapshot_date"] > last_date]
    if new_rows.empty:
        raise ValueError(
            f"Unggahan belum memuat tanggal setelah {pd.Timestamp(last_date):%d/%m/%Y}, tanggal terakhir model bawaan."
        )
    combined = pd.concat([existing, new_rows], ignore_index=True)
    return normalize_raw_history(combined), sorted(new_rows["snapshot_date"].unique())
