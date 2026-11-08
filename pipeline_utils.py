"""Fungsi yang dipakai lebih dari satu tempat."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


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


def attach_calendar_targets(indicators, calendar, *, ticker_col="Kode Saham", date_col="snapshot_date", close_col="close_adj", return_col=None):
    """Memasangkan setiap baris indikator dengan target return hari bursa berikutnya.

    Target diambil dari kalender harga, bukan dari tabel indikator, supaya hari
    bursa berikutnya tetap benar walaupun ada baris yang dibuang dari tabel indikator.

    Args:
        indicators: Tabel indikator dengan kolom ticker_col dan date_col. Kolom
            target_date dan target_next_return yang sudah ada akan diganti.
        calendar: Tabel harga lengkap per saham dan tanggal.
        ticker_col: Nama kolom kode saham.
        date_col: Nama kolom tanggal observasi.
        close_col: Nama kolom harga penutupan.
        return_col: Nama kolom return yang sudah ada di kalender. Jika None,
            return dihitung dari close_col, yaitu close(t+1) / close(t) - 1.

    Returns:
        Salinan indicators ditambah kolom target_date (tanggal hari bursa
        berikutnya) dan target_next_return (simple return pada tanggal itu, dalam
        desimal), diurutkan per saham dan tanggal. Baris terakhir tiap saham
        bernilai kosong karena belum punya hari bursa berikutnya.

    Raises:
        ValueError: Jika kalender memuat pasangan saham dan tanggal ganda, ada
            close yang kosong atau tidak positif, atau indicators memuat
            pasangan saham dan tanggal ganda.
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


SPLIT_PRICE_COLUMNS = {
    "Open Price": "open_adj",
    "Tertinggi": "high_adj",
    "Terendah": "low_adj",
    "Penutupan": "close_adj",
}


def adjust_prices_for_splits(panel):
    """Menyesuaikan harga dan volume ke belakang berdasarkan perubahan Listed Shares.

    Stock split terdeteksi jika Listed Shares naik menjadi kelipatan bulat,
    minimal 2 kali dari hari sebelumnya. Reverse stock split terdeteksi jika
    Listed Shares turun menjadi 1/2, 1/3, dan seterusnya. Harga sebelum tanggal
    split dibagi rasio tersebut dan volume dikali rasio tersebut, sehingga
    return tidak melonjak palsu pada tanggal split.

    Args:
        panel: Data mentah IDX dengan kolom Kode Saham, snapshot_date, Listed
            Shares, Volume, Open Price, Tertinggi, Terendah, dan Penutupan.

    Returns:
        Tuple (prices, split_rows).
        prices: Salinan panel yang diurutkan per saham dan tanggal, ditambah
            kolom open_adj, high_adj, low_adj, close_adj, volume_adj, faktor
            penyesuaian, dan simple_return dari close_adj. Return hari pertama
            tiap saham kosong.
        split_rows: Daftar kejadian split berisi Kode Saham, snapshot_date,
            dan listed_shares_ratio.
    """
    prices = panel.sort_values(["Kode Saham", "snapshot_date"], kind="stable").copy()

    previous_shares = prices.groupby("Kode Saham")["Listed Shares"].shift()
    prices["listed_shares_ratio"] = prices["Listed Shares"] / previous_shares
    ratio = prices["listed_shares_ratio"]
    stock_split = (
        ratio.ge(2) 
        & np.isclose(ratio, np.round(ratio), rtol=0, atol=1e-8)
    )
    reverse_stock_split = (
        ratio.gt(0)
        & ratio.le(0.5)
        & np.isclose(1 / ratio, np.round(1 / ratio), rtol=0, atol=1e-8)
    )
    split_rows = prices.loc[
        stock_split | reverse_stock_split, ["Kode Saham", "snapshot_date", "listed_shares_ratio"]
    ].copy()

    prices["price_adj_factor"] = 1.0
    prices["volume_adj_factor"] = 1.0
    for event in split_rows.itertuples(index=False, name=None):
        ticker, split_date, share_ratio = event
        before_split = prices["Kode Saham"].eq(ticker) & prices["snapshot_date"].lt(split_date)
        prices.loc[before_split, "price_adj_factor"] *= 1 / share_ratio
        prices.loc[before_split, "volume_adj_factor"] *= share_ratio

    for source_column, adjusted_column in SPLIT_PRICE_COLUMNS.items():
        prices[adjusted_column] = prices[source_column] * prices["price_adj_factor"]
    prices["volume_adj"] = prices["Volume"] * prices["volume_adj_factor"]
    prices["simple_return"] = prices.groupby("Kode Saham")["close_adj"].pct_change(
        fill_method=None
    )
    return prices, split_rows.reset_index(drop=True)


def optimize_max_sharpe(expected_returns, covariance, risk_free_daily, max_weight=0.20):
    """Mencari bobot portofolio dengan Sharpe Ratio ex-ante tertinggi menggunakan SLSQP.

    Optimasi dimulai dari bobot sama rata 1/n, dengan total bobot 100% dan tanpa
    short selling.

    Args:
        expected_returns: Expected return harian tiap saham dalam desimal
            (0.01 = 1%), urutannya sama dengan baris dan kolom covariance.
        covariance: Matriks kovarians return harian berukuran n x n.
        risk_free_daily: Risk-free rate harian dalam desimal, yaitu BI Rate
            tahunan dibagi 252.
        max_weight: Batas bobot maksimum tiap saham dalam desimal (0.20 = 20%).

    Returns:
        Array bobot sepanjang n dengan total 1 dan tiap bobot 0 sampai max_weight.

    Raises:
        ValueError: Jika n x max_weight kurang dari 1, sehingga total 100% tidak
            mungkin tercapai.
        RuntimeError: Jika SLSQP gagal atau bobot hasilnya melanggar batas.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n_assets = len(expected_returns)

    # Cek total saham harus 100% jumlahnya, di bawah itu bobot gakbisa dialokasikan maksimal.
    if n_assets * max_weight < 1.0 - 1e-12:
        raise ValueError('Batas maksimum weight tidak memungkinkan total weight 100%')

    def calculate_negative_sharpe(weights):
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights @ covariance @ weights
        # penalti kalau variance hampir nol
        if portfolio_variance <= 1e-16:
            return 1e6
        return -((portfolio_return - risk_free_daily) / np.sqrt(portfolio_variance))

    result = minimize(
        calculate_negative_sharpe,
        np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Optimization gagal: {result.message}")
    candidate = np.clip(result.x, 0.0, max_weight)
    residual = 1.0 - candidate.sum()
    if abs(residual) > 1e-6:
        raise RuntimeError(f"Optimization menghasilkan weight yang tidak memenuhi batas: {result.message}")
    # Menghitung selisih antara total bobot aktual dengan 100%
    for index in range(n_assets):
        if abs(residual) <= 1e-12:
            break
        adjustment = min(residual, max_weight - candidate[index]) if residual > 0 else max(residual, -candidate[index])
        candidate[index] += adjustment
        residual -= adjustment
    if abs(candidate.sum() - 1.0) > 1e-9 or candidate.max() > max_weight + 1e-12:
        raise RuntimeError('Perbaikan weight tidak memenuhi batas portfolio')
    return candidate
