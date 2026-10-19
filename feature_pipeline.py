"""Preprocessing dan feature engineering, sama persis dengan notebook 02.

Dipakai waktu membuat model bawaan dan waktu admin update data, biar data aplikasi
diolah sama seperti di penelitian: memperbaiki baris suspensi, menyesuaikan stock split
dari rasio Listed Shares, menghitung 10 indikator, dan membuat target return besok.
compute_technical_indicators disalin dari notebook 02.
"""
import numpy as np
import pandas as pd

from notebook_config import feature_columns, target
from pipeline_utils import adjust_prices_for_splits, attach_calendar_targets

# nama kolom ikut file Ringkasan Saham IDX, snapshot_date itu tanggal transaksi
KEY_COLUMNS = ["Kode Saham", "snapshot_date"]
PRICE_COLUMNS = ["Open Price", "Tertinggi", "Terendah", "Penutupan"]
RAW_COLUMNS = KEY_COLUMNS + PRICE_COLUMNS + ["Volume", "Listed Shares", "Remarks"]
CALENDAR_COLUMNS = KEY_COLUMNS + PRICE_COLUMNS + [
    "Volume", "Listed Shares", "listed_shares_ratio", "price_adj_factor",
    "volume_adj_factor", "open_adj", "high_adj", "low_adj", "close_adj",
    "volume_adj", "simple_return",
]


def repair_suspension_rows(raw):
    """Mengisi OHL baris suspensi pakai Close sebelumnya, volume tetap nol."""
    raw_df = raw.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True).copy()
    previous_close = raw_df.groupby("Kode Saham")["Penutupan"].shift(1)
    zero_volume = raw_df["Volume"].eq(0)
    zero_ohl = raw_df[["Open Price", "Tertinggi", "Terendah"]].eq(0).all(axis=1)
    suspension_rows = zero_volume & zero_ohl & raw_df["Penutupan"].eq(previous_close)
    for column in PRICE_COLUMNS:
        raw_df.loc[suspension_rows, column] = previous_close.loc[suspension_rows]
    if raw_df.duplicated(KEY_COLUMNS).any():
        raise ValueError("Data memuat saham dan tanggal duplikat")
    return raw_df


def compute_technical_indicators(group_df):
    # mengurutkan data saham berdasarkan tanggal sebelum menghitung indikator
    """Menghitung 10 indikator teknikal untuk satu saham, disalin dari notebook 02.

    Inputnya data harian satu saham dengan kolom snapshot_date, close_adj, high_adj, low_adj.
    Hasilnya data yang sama ditambah SMA10, SMA20, EMA10, RSI14, MACD, Signal, BB_Upper,
    BB_Lower, Stochastic_%K, dan ATR14. Baris awal masih NaN karena indikator butuh data beberapa hari.
    """
    stock_data = group_df.copy().sort_values("snapshot_date").reset_index(drop=True)
    close_values = stock_data["close_adj"].values
    high_values = stock_data["high_adj"].values
    low_values = stock_data["low_adj"].values
    observation_count = len(close_values)

    # menghitung SMA dari 10 dan 20 observasi terakhir
    stock_data["SMA10"] = stock_data["close_adj"].rolling(10).mean()
    stock_data["SMA20"] = stock_data["close_adj"].rolling(20).mean()

    # mean sepuluh close pertama sebagai nilai awal EMA 10
    ema10 = np.full(observation_count, np.nan)
    ema10_alpha = 2.0 / 11.0
    for i in range(observation_count):
        if i < 9:
            continue
        elif i == 9:
            ema10[i] = np.mean(close_values[:10])
        else:
            ema10[i] = close_values[i] * ema10_alpha + ema10[i - 1] * (
                1.0 - ema10_alpha
            )
    stock_data["EMA10"] = ema10

    # menghitung nilai awal RSI, lalu melanjutkan dengan Wilder smoothing
    change = np.diff(close_values, prepend=np.nan)
    gain = np.where(change > 0, change, 0.0)
    loss = np.where(change < 0, -change, 0.0)
    average_gain = np.full(observation_count, np.nan)
    average_loss = np.full(observation_count, np.nan)
    rsi14 = np.full(observation_count, np.nan)
    if observation_count >= 15:
        average_gain[14] = np.mean(gain[1:15])
        average_loss[14] = np.mean(loss[1:15])
        relative_strength = (
            average_gain[14] / average_loss[14] if average_loss[14] != 0 else np.nan
        )
        rsi14[14] = 100.0 - 100.0 / (1.0 + relative_strength)
        for i in range(15, observation_count):
            average_gain[i] = (average_gain[i - 1] * 13.0 + gain[i]) / 14.0
            average_loss[i] = (average_loss[i - 1] * 13.0 + loss[i]) / 14.0
            relative_strength = (
                average_gain[i] / average_loss[i] if average_loss[i] != 0 else np.nan
            )
            rsi14[i] = 100.0 - 100.0 / (1.0 + relative_strength)
    stock_data["RSI14"] = rsi14

    # menghitung EMA 12 dan EMA 26 untuk MACD
    ema12 = np.full(observation_count, np.nan)
    ema12_alpha = 2.0 / 13.0
    for i in range(observation_count):
        if i < 11:
            continue
        elif i == 11:
            ema12[i] = np.mean(close_values[:12])
        else:
            ema12[i] = close_values[i] * ema12_alpha + ema12[i - 1] * (
                1.0 - ema12_alpha
            )

    ema26 = np.full(observation_count, np.nan)
    ema26_alpha = 2.0 / 27.0
    for i in range(observation_count):
        if i < 25:
            continue
        elif i == 25:
            ema26[i] = np.mean(close_values[:26])
        else:
            ema26[i] = close_values[i] * ema26_alpha + ema26[i - 1] * (
                1.0 - ema26_alpha
            )

    stock_data["MACD"] = ema12 - ema26

    # memulai signal dari mean sembilan nilai MACD pertama
    signal = np.full(observation_count, np.nan)
    signal_alpha = 2.0 / 10.0
    valid_macd_idx = [
        i for i in range(observation_count) if not np.isnan(stock_data["MACD"].iloc[i])
    ]
    if len(valid_macd_idx) >= 9:
        start_sig = valid_macd_idx[8]
        signal[start_sig] = np.mean(
            [stock_data["MACD"].iloc[k] for k in valid_macd_idx[:9]]
        )
        for i in range(start_sig + 1, observation_count):
            signal[i] = stock_data["MACD"].iloc[i] * signal_alpha + signal[i - 1] * (
                1.0 - signal_alpha
            )
    stock_data["Signal"] = signal

    # menghitung Bollinger Bands dengan standard deviation populasi
    std20 = (
        stock_data["close_adj"].rolling(20).apply(lambda x: np.std(x, ddof=0), raw=True)
    )
    stock_data["BB_Upper"] = stock_data["SMA20"] + 2.0 * std20
    stock_data["BB_Lower"] = stock_data["SMA20"] - 2.0 * std20

    # memakai nilai 50 pada Stochastic %K jika rentang harga nol
    lowest_14 = stock_data["low_adj"].rolling(14).min()
    highest_14 = stock_data["high_adj"].rolling(14).max()
    price_range = highest_14 - lowest_14
    stock_data["Stochastic_%K"] = np.where(
        price_range != 0,
        (stock_data["close_adj"] - lowest_14) / price_range * 100.0,
        50.0,
    )

    # menghitung true range, lalu ATR 14 dengan Wilder smoothing
    true_range = np.full(observation_count, np.nan)
    for i in range(1, observation_count):
        true_range[i] = max(
            high_values[i] - low_values[i],
            abs(high_values[i] - close_values[i - 1]),
            abs(low_values[i] - close_values[i - 1]),
        )
    atr14 = np.full(observation_count, np.nan)
    if observation_count >= 15:
        atr14[14] = np.mean(true_range[1:15])
        for i in range(15, observation_count):
            atr14[i] = (atr14[i - 1] * 13.0 + true_range[i]) / 14.0
    stock_data["ATR14"] = atr14

    return stock_data


def build_feature_panel(raw, tickers):
    """Hasilnya kalender harga, fitur semua baris, dan fitur yang sudah punya target.

    raw pakai kolom Ringkasan Saham IDX: Kode Saham, snapshot_date, Open Price,
    Tertinggi, Terendah, Penutupan, Volume, dan Listed Shares.
    """
    raw_df = raw.loc[raw["Kode Saham"].isin(tickers)].copy()
    raw_df["snapshot_date"] = pd.to_datetime(raw_df["snapshot_date"])
    # memperbaiki baris suspensi, lalu menyesuaikan harga kalau ada stock split
    raw_df = repair_suspension_rows(raw_df)
    calendar_data, split_events = adjust_prices_for_splits(
        raw_df,
        ticker_col="Kode Saham",
        date_col="snapshot_date",
        listed_shares_col="Listed Shares",
        price_columns={
            "Open Price": "open_adj",
            "Tertinggi": "high_adj",
            "Terendah": "low_adj",
            "Penutupan": "close_adj",
        },
        volume_col="Volume",
    )
    calendar_data = calendar_data[CALENDAR_COLUMNS].copy()
    # menghitung indikator per saham, lalu menempelkan target return hari bursa berikutnya
    processed_list = [
        compute_technical_indicators(group) for _, group in calendar_data.groupby("Kode Saham")
    ]
    fe_df = pd.concat(processed_list, ignore_index=True)
    fe_df = fe_df.sort_values(["Kode Saham", "snapshot_date"]).reset_index(drop=True)
    fe_df = attach_calendar_targets(fe_df, calendar_data, return_col="simple_return")
    # all_features itu semua baris yang indikatornya lengkap, termasuk hari terakhir yang belum punya target
    # labelled itu baris yang sudah punya target, dipakai buat training
    all_features = fe_df.dropna(subset=feature_columns).reset_index(drop=True)
    labelled = fe_df.dropna(subset=feature_columns + [target]).reset_index(drop=True)
    return calendar_data, all_features, labelled, split_events
