"""Bagian inti aplikasi: fitur, model XGBoost, dan portofolio Markowitz.

Isinya:
- konstanta penelitian: daftar saham, batas bobot, risk-free rate, hyperparameter, 10 indikator
- calculate_features: dari data harga jadi 10 indikator plus target return besok
- train_baseline_models: melatih satu model XGBoost per saham
- build_portfolio_allocation dan optimize_max_sharpe: bobot portofolio Max Sharpe
- read_baseline_bundle dan create_baseline_bundle: membaca dan menulis runtime/baseline_bundle.zip
"""
from contextlib import closing
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xgboost as xgb

from pipeline_utils import attach_calendar_targets, filter_non_trading_rows
from feature_pipeline import compute_technical_indicators

# 11 saham bank yang diteliti
PORTFOLIO_TICKERS = ["BBCA", "BBNI", "BBRI", "BBTN", "BDMN", "BJBR", "BMRI", "BNGA", "BNLI", "BTPS", "PNBN"]

# bobot maksimal satu saham di portofolio, 20 persen
MAX_PORTFOLIO_WEIGHT = 0.20

# risk-free rate tahunan 6 persen buat menghitung Sharpe Ratio
RISK_FREE_ANNUAL = 0.06

# jumlah hari terakhir yang dipakai buat menghitung kovarians return
COVARIANCE_WINDOW = 20


# jumlah saham beredar terakhir, dipakai kalau file upload tidak punya kolom Listed Shares
LATEST_LISTED_SHARES = {
    "BBCA": 122_042_299_500,
    "BBRI": 150_043_411_587,
    "BMRI": 92_399_999_996,
    "BBNI": 36_924_339_786,
    "BBTN": 13_894_099_969,
    "BNGA": 24_890_783_784,
    "BDMN": 9_675_817_341,
    "BJBR": 10_416_229_249,
    "BNLI": 35_819_545_925,
    "BTPS": 7_626_663_000,
    "PNBN": 23_837_645_998,
}


# hyperparameter terbaik tiap saham dari tuning notebook 03 (training 2020 sampai 2023, validasi 2024)
MODEL_PARAMS = {
    "BBCA": {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "BBNI": {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5.0},
    "BBRI": {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5.0},
    "BBTN": {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0},
    "BDMN": {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 5.0},
    "BJBR": {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 5.0},
    "BMRI": {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0},
    "BNGA": {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 5.0},
    "BNLI": {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 2.0},
    "BTPS": {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 50, "subsample": 1.0, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 5.0},
    "PNBN": {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 200, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 2.0},
}


# 10 indikator teknikal yang jadi input model
FEATURES = [
    "SMA10", "SMA20", "EMA10", "RSI14", "MACD",
    "Signal", "BB_Upper", "BB_Lower", "Stochastic_%K", "ATR14",
]


# target: return harga penutupan di hari bursa berikutnya
TARGET = "target_next_return"


def calculate_features(data, *, audit_dir=None, check_flat_price=True):
    """Menghitung 10 indikator teknikal dan target return hari berikutnya.

    Baris yang bukan hari transaksi dibuang dulu (dicatat ke file audit), lalu indikator
    dihitung per saham pakai rumus yang sama dengan notebook 02, terus target return besok
    ditempel berdasarkan kalender bursa. Baris awal yang indikatornya belum lengkap dibuang.
    """
    calendar_data = data.copy()
    if audit_dir is None:
        audit_dir = Path(__file__).resolve().parents[1] / "runtime" / "preprocessing_audit" / uuid4().hex
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_logger = logging.Logger("ohlcv_preprocessing", level=logging.INFO)
    with closing(logging.FileHandler(audit_dir / "preprocessing.log", encoding="utf-8")) as audit_handler:
        audit_logger.addHandler(audit_handler)
        try:
            data, _, _, _ = filter_non_trading_rows(
                data, check_flat_price=check_flat_price, audit_dir=audit_dir,
                logger=audit_logger,
            )
        finally:
            audit_logger.removeHandler(audit_handler)
    if data.empty:
        raise ValueError("Data perdagangan kosong setelah preprocessing")
    feature_frames = []
    for ticker, ticker_data in data.groupby("Ticker"):
        # rumus indikator sama dengan notebook 02
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
    """Membuat model XGBoost satu saham pakai hyperparameter hasil tuning notebook 03."""
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


def prediction_label(value):
    """Mengubah angka prediksi jadi label: Naik (di atas 0,1 persen), Turun (di bawah -0,1 persen), atau Stabil."""
    if value > 0.001:
        return "Naik"
    if value < -0.001:
        return "Turun"
    return "Stabil"


def train_baseline_models(features, selected_tickers, test_ratio, on_progress=None):
    """Melatih satu model XGBoost untuk tiap saham yang dipilih.

    Tiap saham punya dua model:
    1. model evaluasi, dilatih di data awal lalu dites di bagian akhir data buat dapat MAE, RMSE, dan akurasi arah
    2. model final, dilatih di semua data lalu dipakai prediksi return hari bursa berikutnya
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    rows, evaluations, models, latest_features, skipped = [], [], {}, {}, []
    for ticker_index, ticker in enumerate(selected_tickers):
        if on_progress:
            on_progress(ticker_index, len(selected_tickers), ticker)
        ticker_data = features[features["Ticker"] == ticker].sort_values("Date").copy()
        labelled = ticker_data.dropna(subset=[TARGET])
        test_count = max(10, int(round(len(labelled) * test_ratio)))
        split_at = len(labelled) - test_count
        if split_at < 30:
            skipped.append(f"{ticker}: data belum cukup")
            continue

        train_data, test_data = labelled.iloc[:split_at], labelled.iloc[split_at:]
        # model evaluasi buat menghitung metrik, model final buat prediksi besok
        evaluation_model = create_model(ticker).fit(train_data[FEATURES], train_data[TARGET])
        test_prediction = evaluation_model.predict(test_data[FEATURES])
        final_model = create_model(ticker).fit(labelled[FEATURES], labelled[TARGET])
        latest = ticker_data.iloc[-1]
        latest_input = pd.DataFrame([latest[FEATURES].astype(float).to_dict()], columns=FEATURES)
        next_prediction = float(final_model.predict(latest_input)[0])
        direction = prediction_label(next_prediction)

        models[ticker] = final_model
        latest_features[ticker] = latest[FEATURES].astype(float).to_dict()
        rows.append({
            "Saham": ticker,
            "Data terakhir": latest["Date"],
            "Prediksi return (%)": next_prediction * 100,
            "Arah": direction,
            "MAE uji (%)": mean_absolute_error(test_data[TARGET], test_prediction) * 100,
            "RMSE uji (%)": np.sqrt(mean_squared_error(test_data[TARGET], test_prediction)) * 100,
            "Akurasi arah (%)": np.mean(np.sign(test_data[TARGET]) == np.sign(test_prediction)) * 100,
            "Data latih": len(train_data),
            "Data uji": len(test_data),
        })
        evaluations.append(pd.DataFrame({
            "Date": test_data["Date"].to_numpy(),
            "Ticker": ticker,
            "Actual": test_data[TARGET].to_numpy() * 100,
            "Predicted": test_prediction * 100,
        }))

    if on_progress:
        on_progress(len(selected_tickers), len(selected_tickers), "Selesai")
    if not rows:
        raise ValueError("Riwayat gabungan belum cukup untuk melatih model yang dipilih.")
    return {
        "summary": pd.DataFrame(rows),
        "evaluation": pd.concat(evaluations, ignore_index=True),
        "models": models,
        "latest_features": latest_features,
        "skipped": skipped,
    }


def optimize_max_sharpe(expected_returns, covariance, risk_free_daily, max_weight):
    """Mencari bobot portofolio dengan Sharpe Ratio paling tinggi (Markowitz).

    Syaratnya total bobot 100 persen dan tiap bobot antara 0 sampai max_weight, tanpa short selling.
    Dihitung pakai SLSQP dari SciPy.
    """
    asset_count = len(expected_returns)
    if asset_count < 2:
        raise ValueError("Optimasi portofolio memerlukan sedikitnya dua saham.")
    if max_weight * asset_count < 1:
        raise ValueError("Batas bobot terlalu kecil untuk jumlah saham yang tersedia.")

    expected = np.asarray(expected_returns, dtype=float)
    covariance_matrix = np.asarray(covariance, dtype=float)
    covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # ditambah angka super kecil biar matriks kovarians tidak singular
    covariance_matrix += np.eye(asset_count) * 1e-10

    def negative_sharpe(weights):
        """SciPy mencari nilai minimum, jadi Sharpe Ratio dibuat negatif."""
        portfolio_return = float(weights @ expected)
        portfolio_variance = float(weights @ covariance_matrix @ weights)
        portfolio_volatility = np.sqrt(max(portfolio_variance, 1e-12))
        return -((portfolio_return - risk_free_daily) / portfolio_volatility)

    initial_weights = np.full(asset_count, 1 / asset_count)
    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * asset_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Optimasi portofolio gagal: {result.message}")

    # merapikan sisa pembulatan biar total bobot pas 100 persen dan tidak lewat batas
    weights = np.clip(result.x, 0.0, max_weight)
    residual = 1.0 - weights.sum()
    if abs(residual) > 1e-6:
        raise ValueError("Optimasi menghasilkan jumlah bobot yang tidak valid.")
    for index in range(asset_count):
        if abs(residual) <= 1e-12:
            break
        adjustment = min(residual, max_weight - weights[index]) if residual > 0 else max(residual, -weights[index])
        weights[index] += adjustment
        residual -= adjustment
    if abs(weights.sum() - 1.0) > 1e-9 or weights.max() > max_weight + 1e-12:
        raise ValueError("Bobot portofolio melampaui batas yang dipilih.")
    return weights


def build_portfolio_allocation(
    market_data,
    baseline_result,
    risk_free_annual=RISK_FREE_ANNUAL,
    covariance_window=COVARIANCE_WINDOW,
    max_weight=MAX_PORTFOLIO_WEIGHT,
):
    """Menghitung bobot portofolio dari prediksi model dan kovarians 20 hari terakhir.

    Expected return diambil dari prediksi XGBoost tiap saham, risikonya dari kovarians
    return harian. Hasilnya dict yang disimpan jadi portfolio.json.
    """
    tickers = sorted(baseline_result["models"])
    close_prices = (
        market_data[market_data["Ticker"].isin(tickers)]
        .pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
        .sort_index()
    )
    returns = close_prices.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < covariance_window:
        raise ValueError(f"Optimasi memerlukan sedikitnya {covariance_window + 1} tanggal lengkap.")

    returns = returns.tail(covariance_window)
    summary = baseline_result["summary"].set_index("Saham")
    expected_returns = summary.loc[tickers, "Prediksi return (%)"].astype(float) / 100
    covariance_frame = returns[tickers].cov()
    covariance = covariance_frame.to_numpy()
    # mengubah risk-free tahunan jadi harian, 252 hari bursa per tahun
    risk_free_daily = risk_free_annual / 252
    if len(tickers) == 1:
        weights = np.array([1.0])
    else:
        effective_max_weight = max(max_weight, 1 / len(tickers))
        weights = optimize_max_sharpe(
            expected_returns.to_numpy(), covariance, risk_free_daily, effective_max_weight
        )
    expected_return_daily = float(weights @ expected_returns.to_numpy())
    volatility_daily = float(np.sqrt(weights @ covariance @ weights))
    sharpe_daily = (
        (expected_return_daily - risk_free_daily) / volatility_daily
        if volatility_daily > 0
        else 0.0
    )

    metrics = baseline_result["summary"].copy()
    metrics["Data terakhir"] = pd.to_datetime(metrics["Data terakhir"]).dt.strftime("%Y-%m-%d")
    return {
        "as_of_date": pd.Timestamp(returns.index.max()).strftime("%Y-%m-%d"),
        "tickers": tickers,
        "weights": dict(zip(tickers, map(float, weights))),
        "predicted_returns": expected_returns.astype(float).to_dict(),
        "covariance": {
            ticker: {
                other_ticker: float(covariance_frame.loc[ticker, other_ticker])
                for other_ticker in tickers
            }
            for ticker in tickers
        },
        "expected_return_daily": expected_return_daily,
        "volatility_daily": volatility_daily,
        "sharpe_daily": float(sharpe_daily),
        "risk_free_annual": float(risk_free_annual),
        "covariance_window": int(covariance_window),
        "max_weight": float(max_weight),
        "model_metrics": metrics.to_dict(orient="records"),
    }


def read_baseline_bundle(bundle, load_models=True):
    """Membaca file zip model bawaan: model tiap saham, contoh input, dan manifest."""
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

    Isinya manifest.json, samples.json, training_summary.csv, portfolio.json,
    market_history.csv, idx_history.csv, dan models/<SAHAM>.json.
    """
    summary = result["summary"].copy()
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
        archive.writestr("training_summary.csv", summary.to_csv(index=False))
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
