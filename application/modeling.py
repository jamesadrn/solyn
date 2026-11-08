"""Bagian inti aplikasi: fitur, model XGBoost, dan portofolio Markowitz."""
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_pipeline import compute_technical_indicators
from notebook_config import feature_columns as FEATURES, target as TARGET
from pipeline_utils import attach_calendar_targets

# risk-free rate tahunan, nilai awal di halaman Simulasi Portofolio
RISK_FREE_ANNUAL = 0.06

# jumlah hari terakhir yang dipakai buat menghitung kovarians return
COVARIANCE_WINDOW = 20

# hyperparameter hasil tuning notebook 03
MODEL_PARAMS = {
    "BBCA": {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "BBNI": {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    "BBRI": {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 100, "subsample": 0.7, "colsample_bytree": 1.0, "reg_alpha": 0.1, "reg_lambda": 5.0},
    "BBTN": {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 5.0},
    "BDMN": {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 50, "subsample": 0.7, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "BJBR": {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.1, "reg_lambda": 2.0},
    "BMRI": {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 1.0, "reg_alpha": 1.0, "reg_lambda": 1.0},
    "BNGA": {"max_depth": 5, "learning_rate": 0.01, "n_estimators": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 2.0},
    "BNLI": {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 100, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 2.0},
    "BTPS": {"max_depth": 4, "learning_rate": 0.08, "n_estimators": 50, "subsample": 0.8, "colsample_bytree": 1.0, "reg_alpha": 0.1, "reg_lambda": 2.0},
    "PNBN": {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 200, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 2.0},
}


def filter_non_trading_rows(data):
    """Membuang baris yang bukan hari transaksi dari data unggahan, sebelum indikator dihitung.

    Satu baris dianggap tidak ada transaksi jika Open, High, Low, dan Close
    bernilai sama, lalu Volume 0 atau Close sama dengan Close hari sebelumnya.

    Args:
        data: Data harga dengan kolom Ticker, Date, Open, High, Low, Close, dan Volume.

    Returns:
        Salinan data yang diurutkan per saham dan tanggal, tanpa baris yang
        tidak ada transaksi.

    Raises:
        ValueError: Jika kolom wajib tidak ada, ticker atau tanggal kosong, ada
            pasangan saham dan tanggal ganda, harga atau volume kosong atau bukan
            angka, atau volume negatif.
    """
    key_columns = ["Ticker", "Date"]
    price_columns = ["Open", "High", "Low", "Close"]
    numeric_columns = [*price_columns, "Volume"]
    missing_columns = sorted(set(key_columns + numeric_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Kolom OHLCV tidak lengkap: {missing_columns}")

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame[key_columns].isna().any().any():
        raise ValueError("Ticker atau tanggal kosong")
    if frame.duplicated(key_columns).any():
        raise ValueError("Data OHLCV memuat ticker dan tanggal duplikat")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Harga atau volume kosong perlu diperiksa sebelum preprocessing")
    if (frame["Volume"] < 0).any():
        raise ValueError("Volume negatif perlu diperiksa sebelum preprocessing")
    frame = frame.sort_values(key_columns, kind="stable").reset_index(drop=True)

    previous_close = frame.groupby("Ticker", sort=False)["Close"].shift(1)
    flat_ohlc = frame[price_columns].eq(frame["Close"], axis=0).all(axis=1)
    removed = flat_ohlc & (frame["Volume"].eq(0) | frame["Close"].eq(previous_close))
    return frame.loc[~removed].reset_index(drop=True)


def calculate_features(data):
    """Menghitung 10 indikator teknikal dan target return hari bursa berikutnya.

    Indikator dihitung per saham dari data yang sudah dibersihkan dengan
    filter_non_trading_rows. Target dihitung dari data sebelum dibersihkan
    melalui attach_calendar_targets.

    Args:
        data: Data harga dengan kolom Ticker, Date, Open, High, Low, Close, dan Volume.

    Returns:
        Tabel berisi kolom data, 10 kolom indikator (FEATURES), target_date, dan
        target_next_return. Baris yang indikatornya belum lengkap di awal periode
        sudah dibuang. Baris terakhir tiap saham tetap ada dengan target kosong,
        karena baris ini dipakai sebagai input prediksi.

    Raises:
        ValueError: Jika data tidak lolos filter_non_trading_rows atau kosong
            setelah dibersihkan.
    """
    calendar_data = data.copy()
    data = filter_non_trading_rows(data)
    if data.empty:
        raise ValueError("Data perdagangan kosong setelah preprocessing")
    feature_frames = []
    for ticker, ticker_data in data.groupby("Ticker"):
        frame = ticker_data.sort_values("Date").copy()
        indicator_input = frame.rename(columns={
            "Date": "snapshot_date", "Close": "close_adj", "High": "high_adj", "Low": "low_adj",
        })
        indicators = compute_technical_indicators(indicator_input)
        frame = frame.reset_index(drop=True)
        for column in FEATURES:
            frame[column] = indicators[column].to_numpy()
        feature_frames.append(frame)

    features = pd.concat(feature_frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    features = attach_calendar_targets(features, calendar_data, ticker_col="Ticker", date_col="Date", close_col="Close")
    return features.dropna(subset=FEATURES).reset_index(drop=True)


def create_model(ticker):
    """Membuat model XGBoost satu saham dengan hyperparameter hasil tuning notebook 03.

    Args:
        ticker: Kode saham, harus ada di MODEL_PARAMS.

    Returns:
        XGBRegressor yang belum dilatih, dengan objective reg:squarederror dan
        random_state 42 seperti di notebook 03.

    Raises:
        ValueError: Jika ticker tidak ada di MODEL_PARAMS.
    """
    if ticker not in MODEL_PARAMS:
        raise ValueError(f"Model belum mendukung saham {ticker}.")
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=min(2, os.cpu_count() or 1),
        verbosity=0,
        **MODEL_PARAMS[ticker],
    )


def train_baseline_models(features, selected_tickers, on_progress=None):
    """Melatih satu model XGBoost untuk tiap saham dan menghitung prediksi terbarunya.

    Model dilatih dari semua baris yang targetnya sudah diketahui. Prediksi
    dibuat dari baris terakhir tiap saham, yaitu return hari bursa berikutnya
    setelah tanggal data terakhir.

    Args:
        features: Hasil calculate_features untuk satu atau beberapa saham.
        selected_tickers: Daftar kode saham yang akan dilatih.
        on_progress: Fungsi opsional on_progress(indeks, total, ticker) untuk
            menampilkan progres training di aplikasi.

    Returns:
        Dictionary berisi:
        summary: Tabel per saham berisi tanggal data terakhir, prediksi return
            dalam persen, dan jumlah data latih.
        models: Model yang sudah dilatih per saham.
        latest_features: Nilai 10 indikator terakhir per saham.
    """
    rows, models, latest_features = [], {}, {}
    for ticker_index, ticker in enumerate(selected_tickers):
        if on_progress:
            on_progress(ticker_index, len(selected_tickers), ticker)
        ticker_data = features[features["Ticker"] == ticker].sort_values("Date").copy()
        labelled = ticker_data.dropna(subset=[TARGET])

        model = create_model(ticker).fit(labelled[FEATURES], labelled[TARGET])
        latest = ticker_data.iloc[-1]
        latest_features[ticker] = latest[FEATURES].astype(float).to_dict()
        latest_input = pd.DataFrame([latest_features[ticker]], columns=FEATURES)
        models[ticker] = model
        rows.append({
            "Saham": ticker,
            "Data terakhir": latest["Date"],
            "Prediksi return (%)": float(model.predict(latest_input)[0]) * 100,
            "Data latih": len(labelled),
        })

    if on_progress:
        on_progress(len(selected_tickers), len(selected_tickers), "Selesai")
    return {
        "summary": pd.DataFrame(rows),
        "models": models,
        "latest_features": latest_features,
    }


def build_portfolio_allocation(market_data, baseline_result):
    """Menyiapkan bahan optimasi untuk halaman Simulasi Portofolio, disimpan jadi portfolio.json.

    Expected return diambil dari prediksi model dan kovarians dari 20 return
    harian terakhir (COVARIANCE_WINDOW) yang lengkap untuk semua saham. Bobot
    tidak dihitung di sini, karena modal, risk-free rate, dan batas bobot
    diisi pengguna di halaman Simulasi Portofolio.

    Args:
        market_data: Riwayat harga dengan kolom Date, Ticker, dan Close.
        baseline_result: Hasil train_baseline_models.

    Returns:
        Dictionary berisi tickers, predicted_returns (desimal), covariance
        (kovarians return harian), dan risk_free_annual (nilai awal 6%).
    """
    tickers = sorted(baseline_result["models"])
    close_prices = (
        market_data[market_data["Ticker"].isin(tickers)]
        .pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
        .sort_index()
    )
    returns = close_prices.pct_change(fill_method=None).dropna(how="any")

    covariance = returns[tickers].tail(COVARIANCE_WINDOW).cov()
    summary = baseline_result["summary"].set_index("Saham")
    expected_returns = summary.loc[tickers, "Prediksi return (%)"].astype(float) / 100
    return {
        "tickers": tickers,
        "predicted_returns": expected_returns.to_dict(),
        "covariance": {
            ticker: {other: float(covariance.loc[ticker, other]) for other in tickers}
            for ticker in tickers
        },
        "risk_free_annual": RISK_FREE_ANNUAL,
    }


def read_baseline_bundle(bundle, load_models=True):
    """Membaca file zip model bawaan: model tiap saham, contoh input, dan manifest.

    Args:
        bundle: Isi file baseline_bundle.zip dalam bytes.
        load_models: Jika False, model tidak dimuat. Dipakai saat hanya butuh
            manifest supaya lebih cepat.

    Returns:
        Tuple (models, samples, manifest).
        models: Model XGBoost per saham, kosong jika load_models False.
        samples: Nilai indikator terakhir per saham dari samples.json.
        manifest: Isi manifest.json. Jika ada portfolio.json, isinya disimpan
            di manifest["portfolio"].
    """
    models, samples, manifest = {}, {}, {}
    with ZipFile(BytesIO(bundle)) as archive:
        if "manifest.json" in archive.namelist():
            manifest = json.loads(archive.read("manifest.json"))
        if "samples.json" in archive.namelist():
            samples = json.loads(archive.read("samples.json"))
        if "portfolio.json" in archive.namelist():
            manifest["portfolio"] = json.loads(archive.read("portfolio.json"))
        if load_models:
            for filename in archive.namelist():
                if filename.startswith("models/") and filename.endswith(".json"):
                    ticker = Path(filename).stem
                    model = xgb.XGBRegressor()
                    model.load_model(bytearray(archive.read(filename)))
                    models[ticker] = model
    return models, samples, manifest


def create_baseline_bundle(result, identity):
    """Menyimpan hasil training ke satu file zip (format baseline_bundle.zip).

    Args:
        result: Hasil train_baseline_models. Boleh ditambah portfolio,
            market_history, idx_history, dan keterangan sumber serta waktu training.
        identity: Data pengguna yang menerbitkan model, minimal berisi email.

    Returns:
        Isi file zip dalam bytes, berisi manifest.json, samples.json, dan
        models/<ticker>.json, ditambah portfolio.json, market_history.csv, dan
        idx_history.csv jika tersedia.
    """
    manifest = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "published_by": identity["email"],
        "tickers": sorted(result["models"]),
        "format_version": 3,
    }
    for key in ("source", "market_data_date", "training_data_start", "training_data_end", "training_started_at", "training_completed_at"):
        if result.get(key) is not None:
            manifest[key] = result[key]
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("samples.json", json.dumps(result["latest_features"], indent=2))
        if result.get("portfolio"):
            archive.writestr("portfolio.json", json.dumps(result["portfolio"], indent=2))
        if result.get("market_history") is not None:
            archive.writestr("market_history.csv", result["market_history"].to_csv(index=False))
        if result.get("idx_history") is not None:
            archive.writestr("idx_history.csv", result["idx_history"].to_csv(index=False))
        for ticker, model in result["models"].items():
            model_bytes = bytes(model.get_booster().save_raw(raw_format="json"))
            archive.writestr(f"models/{ticker}.json", model_bytes)
    return output.getvalue()
