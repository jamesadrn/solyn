"""Fungsi bersama yang dipakai notebook, aplikasi, dan tes.

- show_table: menampilkan tabel rapi di notebook
- filter_non_trading_rows: membuang baris yang bukan hari transaksi
- attach_calendar_targets: menempelkan target return hari bursa berikutnya
- adjust_prices_for_splits: menyesuaikan harga kalau ada stock split
- build_exchange_calendar: tanggal keputusan dan tanggal realisasi buat backtest notebook 05
- calculate_held_weights: bobot portofolio setelah harga bergerak, dipakai notebook 05
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def show_table(frame, *, display_names=None, max_rows=15, row_formats=None):
    """Menampilkan salinan tabel dengan nama kolom dan format angka yang sesuai."""
    from IPython.display import display

    names = {} if display_names is None else display_names
    view = frame.head(max_rows).copy()
    view = view.rename(columns=names, index=names)
    view = view.rename_axis(
        index=lambda name: names.get(name, name),
        columns=lambda name: names.get(name, name),
    )
    formats = {}
    for column in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[column]):
            view[column] = view[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_integer_dtype(view[column]):
            formats[column] = "{:,}"
        elif pd.api.types.is_float_dtype(view[column]):
            formats[column] = "{:,.6g}"
    styled = view.style.format(formats, na_rep="Tidak tersedia")
    for row, value_format in (row_formats or {}).items():
        if row in view.index:
            styled = styled.format(value_format, subset=pd.IndexSlice[[row], :])
    display(styled)


def filter_non_trading_rows(
    data,
    *,
    ticker_col="Ticker",
    date_col="Date",
    volume_col="Volume",
    open_col="Open",
    high_col="High",
    low_col="Low",
    close_col="Close",
    check_flat_price=True,
    audit_dir=None,
    logger=None,
):
    """Return data bersih, baris dihapus, anomali, dan ringkasan per ticker.

    Volume nol dengan Open, High, Low nol dan Close tetap juga dihapus.
    Volume nol dengan pola harga lain tetap disimpan sebagai anomali.
    Pengecekan flat-price tambahan memakai Close sebelumnya per ticker.
    audit_dir menyimpan removed_rows.csv, anomalies.csv, dan summary.csv.
    Data masukan tidak diubah dan tanggal yang dihapus tidak diisi kembali.
    """
    audit_logger = LOGGER if logger is None else logger
    key_columns = [ticker_col, date_col]
    price_columns = [open_col, high_col, low_col, close_col]
    required_columns = [*key_columns, *price_columns, volume_col]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Kolom OHLCV tidak lengkap: {missing_columns}")

    frame = data.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    if frame[key_columns].isna().any().any():
        raise ValueError("Ticker atau tanggal kosong")
    if frame.duplicated(key_columns).any():
        raise ValueError("Data OHLCV memuat ticker dan tanggal duplikat")
    numeric_columns = [*price_columns, volume_col]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Harga atau volume kosong perlu diperiksa sebelum preprocessing")
    if (frame[volume_col] < 0).any():
        raise ValueError("Volume negatif perlu diperiksa sebelum preprocessing")
    frame = frame.sort_values(key_columns, kind="stable").reset_index(drop=True)

    # membandingkan ohlc dan close sebelumnya pada ticker yang sama
    previous_close = frame.groupby(ticker_col, sort=False)[close_col].shift(1)
    flat_ohlc = frame[price_columns].eq(frame[close_col], axis=0).all(axis=1)
    zero_volume = frame[volume_col].eq(0)
    zero_ohl_unchanged_close = (
        zero_volume
        & frame[[open_col, high_col, low_col]].eq(0).all(axis=1)
        & frame[close_col].eq(previous_close)
    )
    anomaly_mask = zero_volume & ~flat_ohlc & ~zero_ohl_unchanged_close
    flat_previous_close = flat_ohlc & frame[close_col].eq(previous_close)
    removal_mask = (zero_volume & flat_ohlc) | zero_ohl_unchanged_close
    if check_flat_price:
        removal_mask = removal_mask | flat_previous_close
    removal_mask = removal_mask & ~anomaly_mask

    removed = frame.loc[removal_mask, key_columns].copy().reset_index(drop=True)
    anomalies = frame.loc[anomaly_mask, required_columns].copy().reset_index(drop=True)
    cleaned = frame.loc[~removal_mask].copy().reset_index(drop=True)
    summary = frame.groupby(ticker_col, sort=False).size().to_frame("Input Rows")
    summary["Removed"] = removed.groupby(ticker_col).size().reindex(summary.index, fill_value=0)
    summary["Anomalies"] = anomalies.groupby(ticker_col).size().reindex(summary.index, fill_value=0)
    summary["Retained"] = summary["Input Rows"] - summary["Removed"]
    summary = summary.reset_index().rename(columns={ticker_col: "Ticker"})
    totals = {column: int(summary[column].sum()) for column in summary.columns if column != "Ticker"}
    summary = pd.concat([summary, pd.DataFrame([{"Ticker": "Total", **totals}])], ignore_index=True)

    # memvalidasi kunci dan urutan setelah penghapusan
    if cleaned.duplicated(key_columns).any():
        raise ValueError("Hasil preprocessing memuat ticker dan tanggal duplikat")
    if not cleaned[key_columns].equals(cleaned.sort_values(key_columns)[key_columns]):
        raise ValueError("Hasil preprocessing belum terurut")

    audit_logger.info("Ringkasan preprocessing OHLCV\n%s", summary.to_string(index=False))
    if not anomalies.empty:
        log_anomalies = anomalies.copy()
        log_anomalies[date_col] = log_anomalies[date_col].dt.strftime("%Y-%m-%d")
        audit_logger.warning("Volume nol dengan OHLC tidak flat. Baris dipertahankan\n%s", log_anomalies.to_string(index=False))
    if audit_dir is not None:
        audit_path = Path(audit_dir)
        audit_path.mkdir(parents=True, exist_ok=True)
        removed.to_csv(audit_path / "removed_rows.csv", index=False, date_format="%Y-%m-%d")
        anomalies.to_csv(audit_path / "anomalies.csv", index=False, date_format="%Y-%m-%d")
        summary.to_csv(audit_path / "summary.csv", index=False)
    return cleaned, removed, anomalies, summary


def attach_calendar_targets(indicators, calendar, *, ticker_col="Kode Saham", date_col="snapshot_date", close_col="close_adj", return_col=None):
    """Memasangkan indikator dengan target hari bursa berikutnya dari kalender harga.

    target_next_return di tanggal t itu return dari t ke hari bursa berikutnya (t+1).
    Jadi model pakai indikator hari ini buat menebak return besok.
    """
    columns = [ticker_col, date_col, close_col]
    if return_col is not None:
        columns.append(return_col)
    prices = calendar[columns].copy()
    prices[date_col] = pd.to_datetime(prices[date_col], errors="raise")
    if prices.duplicated([ticker_col, date_col]).any():
        raise ValueError("Kalender harga memuat ticker dan tanggal duplikat")
    if prices[close_col].isna().any() or (prices[close_col] <= 0).any():
        raise ValueError("Close kalender harus tersedia dan positif")
    prices = prices.sort_values([ticker_col, date_col])
    grouped = prices.groupby(ticker_col, sort=False)
    prices["target_date"] = grouped[date_col].shift(-1)
    if return_col is None:
        prices["target_next_return"] = grouped[close_col].shift(-1) / prices[close_col] - 1
    else:
        prices["target_next_return"] = grouped[return_col].shift(-1)
    targets = prices[[ticker_col, date_col, "target_date", "target_next_return"]]
    frame = indicators.drop(columns=["target_date", "target_next_return"], errors="ignore").copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    return frame.merge(targets, on=[ticker_col, date_col], how="left", validate="one_to_one").sort_values([ticker_col, date_col]).reset_index(drop=True)


def adjust_prices_for_splits(
    panel,
    *,
    ticker_col="Kode Saham",
    date_col="snapshot_date",
    listed_shares_col="Listed Shares",
    price_columns=None,
    volume_col="Volume",
):
    """Menyesuaikan harga dan volume ke belakang pakai rasio Listed Shares.

    Stock split ketahuan kalau jumlah saham beredar berubah dengan rasio bulat, misalnya jadi
    5 kali lipat. Harga sebelum tanggal split dibagi rasio itu dan volumenya dikali, jadi grafik
    harga tidak kelihatan anjlok palsu di tanggal split.
    """
    if price_columns is None:
        price_columns = {
            "Open Price": "open_adj",
            "Tertinggi": "high_adj",
            "Terendah": "low_adj",
            "Penutupan": "close_adj",
        }
    required = {
        ticker_col,
        date_col,
        listed_shares_col,
        volume_col,
        *price_columns.keys(),
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Kolom untuk penyesuaian split belum lengkap: {sorted(missing)}")

    prices = panel.sort_values([ticker_col, date_col], kind="stable").copy()
    if prices.duplicated([ticker_col, date_col]).any():
        raise ValueError("Ada ticker dan tanggal duplikat pada data harga")

    previous_shares = prices.groupby(ticker_col)[listed_shares_col].shift()
    prices["listed_shares_ratio"] = prices[listed_shares_col] / previous_shares
    ratio = prices["listed_shares_ratio"]
    # split biasa kalau rasionya bulat dan minimal 2, reverse split kalau rasionya 0,5 ke bawah
    split_up = ratio.ge(2) & np.isclose(ratio, np.round(ratio), rtol=0, atol=1e-8)
    split_reverse = (
        ratio.gt(0)
        & ratio.le(0.5)
        & np.isclose(1 / ratio, np.round(1 / ratio), rtol=0, atol=1e-8)
    )
    split_rows = prices.loc[
        split_up | split_reverse, [ticker_col, date_col, "listed_shares_ratio"]
    ].copy()

    prices["price_adj_factor"] = 1.0
    prices["volume_adj_factor"] = 1.0
    # tiap split mengoreksi semua tanggal sebelum split di saham itu
    for event in split_rows.itertuples(index=False, name=None):
        ticker, split_date, share_ratio = event
        before_split = prices[ticker_col].eq(ticker) & prices[date_col].lt(split_date)
        prices.loc[before_split, "price_adj_factor"] *= 1 / share_ratio
        prices.loc[before_split, "volume_adj_factor"] *= share_ratio

    for source_column, adjusted_column in price_columns.items():
        prices[adjusted_column] = prices[source_column] * prices["price_adj_factor"]
    prices["volume_adj"] = prices[volume_col] * prices["volume_adj_factor"]
    prices["simple_return"] = prices.groupby(ticker_col)["close_adj"].pct_change(
        fill_method=None
    )
    split_rows = split_rows.rename(
        columns={ticker_col: "Kode Saham", date_col: "snapshot_date"}
    ).reset_index(drop=True)
    return prices, split_rows


def build_exchange_calendar(return_panel, *, start_date, end_date):
    """Mempertahankan seluruh tanggal keputusan yang memiliki hari bursa berikutnya."""
    dates = pd.DatetimeIndex(return_panel.index).sort_values()
    if dates.has_duplicates:
        raise ValueError("Kalender bursa memuat tanggal duplikat")
    targets = pd.Series(dates, index=dates).shift(-1)
    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date)) & targets.notna().to_numpy()
    decisions = dates[mask]
    return decisions, targets.loc[decisions]


def calculate_held_weights(previous_weights, realized_returns):
    """Menghitung bobot setelah perubahan harga dengan jumlah saham tetap."""
    weights = np.asarray(previous_weights, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    if weights.shape != returns.shape or not np.isfinite(weights).all() or not np.isfinite(returns).all():
        raise ValueError("Bobot atau return posisi tidak valid")
    if (weights < -1e-12).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Jumlah bobot posisi harus satu")
    values = weights * (1 + returns)
    if (values < 0).any() or values.sum() <= 0:
        raise ValueError("Nilai posisi sesudah perubahan harga tidak valid")
    return values / values.sum()
