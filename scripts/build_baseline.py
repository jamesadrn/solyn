"""Membuat runtime/baseline_bundle.zip dari data penelitian 2020 sampai 2025.

Jalankan dari folder proyek setelah notebook 01 menghasilkan data/interim/panel_raw.parquet:

    python scripts/build_baseline.py
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from application.baseline_builder import build_baseline_bundle  # noqa: E402

DEFAULT_SOURCE = ROOT_DIR / "data" / "interim" / "panel_raw.parquet"
DEFAULT_OUTPUT = ROOT_DIR / "runtime" / "baseline_bundle.zip"


def main():
    """Membaca data mentah IDX, melatih 11 model, lalu menulis zip model bawaan."""
    parser = argparse.ArgumentParser(description="Bangun model bawaan aplikasi dari data IDX.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--published-by", default="notebook-baseline")
    arguments = parser.parse_args()

    raw = pd.read_parquet(arguments.source)
    bundle, info = build_baseline_bundle(
        raw, arguments.published_by, "Ringkasan Saham IDX (data penelitian 2020-2025)"
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # menulis ke file sementara dulu biar file lama aman kalau prosesnya berhenti di tengah
    temporary = arguments.output.with_suffix(".tmp")
    temporary.write_bytes(bundle)
    temporary.replace(arguments.output)
    print(f"Bundle dibuat sampai {info['market_data_date']} dari {info['labelled_rows']} baris latih.")
    print(info["summary"][["Saham", "Data terakhir", "Prediksi return (%)", "Data latih"]].to_string(index=False))


if __name__ == "__main__":
    main()
