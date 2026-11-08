"""Membuat bundle model bawaan aplikasi dari data Ringkasan Saham IDX."""
from datetime import datetime, timezone

import pandas as pd

from application.modeling import (
    build_portfolio_allocation,
    create_baseline_bundle,
    read_baseline_bundle,
    train_baseline_models,
)
from feature_pipeline import RAW_COLUMNS, build_feature_panel
from notebook_config import tickers as PORTFOLIO_TICKERS


def normalize_raw_history(raw):
    """Memastikan data mentah IDX cuma berisi kolom dan 11 saham yang dipakai.

    Args:
        raw: Data mentah Ringkasan Saham IDX. Kolom Remarks boleh tidak ada.

    Returns:
        Data dengan kolom RAW_COLUMNS saja, tanggal tanpa jam, harga, volume, dan
        Listed Shares berupa angka, diurutkan per saham dan tanggal.

    Raises:
        ValueError: Jika kolom wajib tidak ada, ada nilai angka yang tidak
            terbaca, ada pasangan saham dan tanggal ganda, atau ada tanggal yang
            tidak memuat lengkap 11 saham.
    """
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
    """Mengubah kalender harga ke format riwayat harga yang dibaca halaman aplikasi.

    Args:
        calendar: Kalender harga dari build_feature_panel dengan harga dan volume
            yang sudah disesuaikan split (open_adj sampai volume_adj).

    Returns:
        Tabel dengan kolom Date, Ticker, Open, High, Low, Close, Volume, Listed
        Shares, dan Remarks, diurutkan per saham dan tanggal.
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
    """Melatih ulang 11 model dari riwayat IDX dan menyimpannya sebagai baseline_bundle.zip.

    Dipakai di Panel Admin dan scripts/build_baseline.py. Urutannya: rapikan data
    mentah, bentuk fitur, latih model, siapkan bahan portofolio, lalu simpan ke zip.

    Args:
        raw_history: Data mentah Ringkasan Saham IDX untuk 11 saham.
        published_by: Email admin yang menjalankan training.
        source: Keterangan asal data, disimpan di manifest.

    Returns:
        Tuple (bundle, info).
        bundle: Isi baseline_bundle.zip dalam bytes.
        info: Ringkasan berisi market_data_date, labelled_rows, dan summary
            prediksi per saham.

    Raises:
        ValueError: Jika data mentah tidak lolos normalize_raw_history.
        RuntimeError: Jika isi bundle tidak lengkap 11 saham.
    """
    raw = normalize_raw_history(raw_history)
    started_at = datetime.now(timezone.utc).isoformat()
    calendar, all_features, labelled = build_feature_panel(raw, PORTFOLIO_TICKERS)
    features = all_features.rename(columns={"Kode Saham": "Ticker", "snapshot_date": "Date"})
    result = train_baseline_models(features, PORTFOLIO_TICKERS)
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
        "training_started_at": started_at,
        "training_completed_at": datetime.now(timezone.utc).isoformat(),
    })
    bundle = create_baseline_bundle(result, {"email": published_by})
    _, _, manifest = read_baseline_bundle(bundle, load_models=False)
    if set(manifest.get("tickers", [])) != set(PORTFOLIO_TICKERS):
        raise RuntimeError("Bundle baru belum memuat 11 saham.")
    return bundle, {
        "market_data_date": result["market_data_date"],
        "labelled_rows": int(len(labelled)),
        "summary": result["summary"],
    }


def append_idx_rows(existing_raw, uploaded_raw):
    """Menambahkan tanggal baru dari upload admin ke data mentah IDX.

    Hanya tanggal setelah tanggal terakhir riwayat yang diambil. Tanggal yang
    sudah ada tidak ditimpa.

    Args:
        existing_raw: Riwayat IDX yang dipakai model bawaan saat ini.
        uploaded_raw: Data IDX dari file unggahan admin.

    Returns:
        Tuple (combined, new_dates).
        combined: Riwayat lama ditambah baris baru, sudah dirapikan.
        new_dates: Daftar tanggal baru yang ditambahkan, terurut.

    Raises:
        ValueError: Jika unggahan tidak memuat tanggal baru, atau salah satu data
            tidak lolos normalize_raw_history.
    """
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
