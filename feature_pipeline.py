"""Preprocessing dan feature engineering yang dipakai bersama notebook 02 dan aplikasi."""
import numpy as np
import pandas as pd

from notebook_config import feature_columns, target
from pipeline_utils import adjust_prices_for_splits, attach_calendar_targets

KEY_COLUMNS = ["Kode Saham", "snapshot_date"]
PRICE_COLUMNS = ["Open Price", "Tertinggi", "Terendah", "Penutupan"]
RAW_COLUMNS = KEY_COLUMNS + PRICE_COLUMNS + ["Volume", "Listed Shares", "Remarks"]
CALENDAR_COLUMNS = KEY_COLUMNS + PRICE_COLUMNS + [
    "Volume", "Listed Shares", "listed_shares_ratio", "price_adj_factor",
    "volume_adj_factor", "open_adj", "high_adj", "low_adj", "close_adj",
    "volume_adj", "simple_return",
]


def repair_suspension_rows(raw):
    """Mengisi Open, High, Low, dan Close baris suspensi dengan Close hari sebelumnya.

    Baris dianggap suspensi jika Volume 0, Open, High, dan Low bernilai 0, dan
    Close sama dengan Close hari sebelumnya.

    Raises:
        ValueError: Jika ada pasangan saham dan tanggal ganda.
    """
    raw_df = raw.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True).copy()
    previous_close = raw_df.groupby("Kode Saham")["Penutupan"].shift(1)
    zero_volume = raw_df["Volume"].eq(0)
    zero_ohl = raw_df[["Open Price", "Tertinggi", "Terendah"]].eq(0).all(axis=1)
    suspension_rows = zero_volume & zero_ohl & raw_df["Penutupan"].eq(previous_close)
    for column in PRICE_COLUMNS:
        raw_df.loc[suspension_rows, column] = previous_close.loc[suspension_rows]
    if raw_df.duplicated(KEY_COLUMNS).any():
        raise ValueError("terdapat saham dan tanggal yang duplikat")
    return raw_df


def exponential_moving_average(values, n):
    """EMA = alpha * nilai hari ini + (1 - alpha) * EMA kemarin, alpha = 2 / (n + 1), nilai awalnya SMA n hari."""
    ema = np.full(len(values), np.nan)
    alpha = 2.0 / (n + 1)
    for i in range(n - 1, len(values)):
        if i == n - 1:
            ema[i] = np.mean(values[:n])
        else:
            ema[i] = values[i] * alpha + ema[i - 1] * (1.0 - alpha)
    return ema


def wilder_smoothing(values, n):
    """Rata-rata Wilder = ((n - 1) * rata-rata kemarin + nilai hari ini) / n, nilai awalnya rata-rata n hari."""
    # indeks 0 dilewati karena perubahan harga dan True Range butuh Close hari sebelumnya
    average = np.full(len(values), np.nan)
    if len(values) > n:
        average[n] = np.mean(values[1:n + 1])
        for i in range(n + 1, len(values)):
            average[i] = (average[i - 1] * (n - 1) + values[i]) / n
    return average


def compute_technical_indicators(group_df):
    """Menghitung 10 indikator teknikal untuk satu saham.

    Butuh kolom snapshot_date, close_adj, high_adj, dan low_adj. Baris awal
    bernilai kosong sampai window tiap indikator terpenuhi.
    """
    stock_data = group_df.copy().sort_values("snapshot_date").reset_index(drop=True)
    close = stock_data["close_adj"]
    close_values = close.values
    high_values = stock_data["high_adj"].values
    low_values = stock_data["low_adj"].values

    stock_data["SMA10"] = close.rolling(window=10).mean()
    stock_data["SMA20"] = close.rolling(window=20).mean()

    stock_data["EMA10"] = exponential_moving_average(close_values, n=10)

    change = np.diff(close_values, prepend=np.nan)
    gain = np.where(change > 0, change, 0.0)
    loss = np.where(change < 0, -change, 0.0)
    average_gain = wilder_smoothing(gain, n=14)
    average_loss = wilder_smoothing(loss, n=14)
    # rata-rata loss 0 diganti NaN supaya tidak membagi nol, jadi RSI di baris itu ikut kosong
    relative_strength = average_gain / np.where(average_loss != 0, average_loss, np.nan)
    stock_data["RSI14"] = 100.0 - 100.0 / (1.0 + relative_strength)

    macd = exponential_moving_average(close_values, n=12) - exponential_moving_average(close_values, n=26)
    stock_data["MACD"] = macd

    # MACD baru ada mulai indeks 25 (hari ke-26), jadi EMA9 untuk Signal dihitung dari situ
    signal = np.full(len(macd), np.nan)
    signal[25:] = exponential_moving_average(macd[25:], n=9)
    stock_data["Signal"] = signal

    # ddof=0 karena Bollinger Bands memakai standar deviasi populasi
    std20 = close.rolling(window=20).apply(lambda x: np.std(x, ddof=0), raw=True)
    stock_data["BB_Upper"] = stock_data["SMA20"] + 2.0 * std20
    stock_data["BB_Lower"] = stock_data["SMA20"] - 2.0 * std20

    # bernilai 50 kalau High tertinggi sama dengan Low terendah, supaya tidak membagi nol
    lowest_14 = stock_data["low_adj"].rolling(window=14).min()
    highest_14 = stock_data["high_adj"].rolling(window=14).max()
    price_range = highest_14 - lowest_14
    stock_data["Stochastic_%K"] = np.where(price_range != 0, (close - lowest_14) / price_range * 100.0, 50.0)

    previous_close = close.shift(1).values
    true_range = np.maximum.reduce([
        high_values - low_values,
        np.abs(high_values - previous_close),
        np.abs(low_values - previous_close),
    ])
    stock_data["ATR14"] = wilder_smoothing(true_range, n=14)

    return stock_data


def build_feature_panel(raw, tickers):
    """Membentuk kalender harga dan fitur dari data mentah IDX.

    Args:
        raw: Data mentah IDX dengan kolom RAW_COLUMNS.
        tickers: Kode saham yang diproses.

    Returns:
        Tuple (calendar_data, all_features, labelled).
        calendar_data: Harga harian yang sudah disesuaikan split, kolom CALENDAR_COLUMNS.
        all_features: Baris dengan 10 indikator lengkap, termasuk baris terakhir
            tiap saham yang targetnya belum diketahui.
        labelled: Baris dengan indikator dan target lengkap, dipakai untuk training.
    """
    raw_df = raw.loc[raw["Kode Saham"].isin(tickers)].copy()
    raw_df["snapshot_date"] = pd.to_datetime(raw_df["snapshot_date"])
    raw_df = repair_suspension_rows(raw_df)
    calendar_data, _ = adjust_prices_for_splits(raw_df)
    calendar_data = calendar_data[CALENDAR_COLUMNS].copy()
    processed_list = [
        compute_technical_indicators(group) for _, group in calendar_data.groupby("Kode Saham")
    ]
    fe_df = pd.concat(processed_list, ignore_index=True)
    fe_df = fe_df.sort_values(["Kode Saham", "snapshot_date"]).reset_index(drop=True)
    fe_df = attach_calendar_targets(fe_df, calendar_data, return_col="simple_return")
    all_features = fe_df.dropna(subset=feature_columns).reset_index(drop=True)
    labelled = fe_df.dropna(subset=feature_columns + [target]).reset_index(drop=True)
    return calendar_data, all_features, labelled
