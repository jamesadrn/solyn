# Solyn

Solyn adalah aplikasi berbasis web untuk memprediksi return harian saham perbankan Indonesia dan menyimulasikan pembagian dana portofolio. Prediksi dibuat dengan model XGBoost dari sepuluh indikator teknikal, lalu dipakai untuk menyusun bobot portofolio dengan metode Mean-Variance Markowitz.

Aplikasi ini dibuat sebagai bagian dari skripsi Fakultas Teknologi Informasi, Universitas Tarumanagara.

Alamat aplikasi: [Solyn](https://solynapp.streamlit.app)

> Hasil aplikasi adalah perkiraan model dan simulasi penelitian, bukan saran investasi.

## Daftar isi

1. [Fitur](#fitur)
2. [Cara menggunakan](#cara-menggunakan)
3. [Saham yang dianalisis](#saham-yang-dianalisis)
4. [Metode](#metode)
5. [Struktur repositori](#struktur-repositori)
6. [Menjalankan di komputer sendiri](#menjalankan-di-komputer-sendiri)
7. [Pembuat](#pembuat)

## Fitur

| Halaman | Kegunaan |
| --- | --- |
| Login | Halaman pertama yang dilihat pengguna. Isi aplikasi baru dapat dibuka setelah masuk dengan akun Google |
| Persetujuan akses | Muncul untuk akun yang belum disetujui admin setelah masa coba gratis 5 menit berakhir. Pengguna dapat mengajukan akses atau keluar |
| Home | Ringkasan harga terakhir, prediksi return hari bursa berikutnya, dan grafik harga |
| Analisis Saham | Prediksi return satu saham beserta indikator yang paling memengaruhi hasilnya |
| Simulasi Portofolio | Pembagian dana ke beberapa saham dengan Sharpe Ratio tertinggi |
| Hasil Pengujian | Evaluasi model dan portofolio pada data tahun 2025 |
| Unggah Data | Melatih model sementara dari file Ringkasan Saham IDX terbaru milik pengguna |
| Tentang | Tujuan aplikasi, cara penggunaan, pembuat, dan dosen pembimbing |
| Panel Admin | Persetujuan akses pengguna dan pembaruan data serta model (khusus admin) |

## Cara menggunakan

1. Buka alamat aplikasi, lalu tekan **Masuk dengan Google**. Akun yang baru pertama kali masuk otomatis terdaftar.
2. Akun baru langsung dapat memakai semua fitur selama **5 menit**, satu kali per akun. Pada saat yang sama, permintaan akses dikirim ke admin.
3. Setelah admin menyetujui, semua fitur dapat dipakai tanpa batas waktu. Jika disetujui di tengah 5 menit tersebut, aplikasi langsung menyesuaikan tanpa perlu masuk ulang.
4. Pilih halaman dari menu di sebelah kiri. Panduan singkat setiap halaman tersedia di halaman **Tentang**.

## Saham yang dianalisis

Aplikasi menganalisis 11 saham perbankan yang tercatat di Bursa Efek Indonesia.

| No | Kode | Nama bank |
| --- | --- | --- |
| 1 | BBCA | Bank Central Asia |
| 2 | BBNI | Bank Negara Indonesia |
| 3 | BBRI | Bank Rakyat Indonesia |
| 4 | BBTN | Bank Tabungan Negara |
| 5 | BDMN | Bank Danamon Indonesia |
| 6 | BJBR | Bank Pembangunan Daerah Jawa Barat dan Banten |
| 7 | BMRI | Bank Mandiri |
| 8 | BNGA | Bank CIMB Niaga |
| 9 | BNLI | Bank Permata |
| 10 | BTPS | Bank BTPN Syariah |
| 11 | PNBN | Bank Pan Indonesia |

## Metode

| Tahap | Keterangan |
| --- | --- |
| Data | File Ringkasan Saham harian dari Bursa Efek Indonesia, Januari 2020 sampai Desember 2025 |
| Pengolahan | Perbaikan baris suspensi, penyesuaian harga terhadap stock split, dan perhitungan simple return |
| Fitur | SMA10, SMA20, EMA10, RSI14, MACD, Signal, Bollinger Bands atas dan bawah, Stochastic %K, ATR14 |
| Model | XGBoost per saham. Latih 2020 sampai 2023, validasi 2024, uji 2025 dengan pelatihan ulang setiap kuartal |
| Interpretasi | Gain Importance, Permutation Importance, dan SHAP |
| Portofolio | Mean-Variance Markowitz (Sharpe Ratio tertinggi, bobot maksimal 20% per saham), dibandingkan dengan portofolio Historical Mean |

Model yang dipakai aplikasi dilatih ulang pada seluruh data 2020 sampai 2025 dengan hyperparameter hasil validasi 2024.

## Struktur repositori

```
application/        kode aplikasi (app.py sebagai titik masuk)
notebooks/          notebook penelitian 01 sampai 05
data/interim/       data gabungan hasil notebook 01
data/processed/     fitur, kalender harga, dan prediksi 2025
output/research/    hasil evaluasi, interpretasi, dan portofolio
runtime/            model yang dipakai aplikasi
scripts/            skrip untuk membangun ulang model aplikasi
assets/             logo dan tampilan aplikasi
```

Berkas `feature_pipeline.py`, `pipeline_utils.py`, dan `notebook_config.py` dipakai bersama oleh notebook dan aplikasi, sehingga keduanya mengolah data dengan cara yang sama.

| Notebook | Isi |
| --- | --- |
| `01_eda.ipynb` | Menggabungkan file Ringkasan Saham harian dan memeriksa kualitas data |
| `02_preprocessing.ipynb` | Pembersihan data, penyesuaian stock split, dan indikator teknikal |
| `03_modelling.ipynb` | Hyperparameter tuning, pelatihan ulang per kuartal, dan evaluasi prediksi |
| `04_interpretabilitas.ipynb` | Gain Importance, Permutation Importance, dan SHAP |
| `05_optimasi_portofolio.ipynb` | Optimasi portofolio, perbandingan dengan Historical Mean, dan uji sensitivitas |

## Menjalankan di komputer sendiri

Membutuhkan Python 3.11.

**Aplikasi**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run application/app.py
```

Aplikasi terbuka di `http://localhost:8501`. Login Google memerlukan berkas `.streamlit/secrets.toml`. Contoh isinya tersedia di `.streamlit/secrets.toml.example`.

**Notebook**

```
python -m venv .venv-notebooks
.venv-notebooks\Scripts\activate
pip install -r requirements-notebooks.txt
jupyter lab
```

Notebook dijalankan berurutan dari 01 sampai 05. File Excel mentah dari BEI (sekitar 1 GB) tidak disertakan. Hasil notebook 01 sudah tersedia di `data/interim/panel_raw.parquet`, jadi notebook 02 sampai 05 dapat langsung dijalankan. Untuk menjalankan notebook 01, unduh Ringkasan Saham harian dari [idx.co.id](https://www.idx.co.id) dan simpan di folder `data/`.

## Pembuat

| Keterangan | Detail |
| --- | --- |
| Nama | James Anderson |
| NIM | 535230065 |
| Fakultas | Fakultas Teknologi Informasi, Universitas Tarumanagara |
| Dosen pembimbing 1 | Tony, S.Kom., M.Kom., Ph. D. |
| Dosen pembimbing 2 | Irvan Lewenusa, S.Kom., M.Kom. |
