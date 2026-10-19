"""Baca dan tulis Google Sheet daftar pengguna: izin akses dan coba gratis.

Satu sheet dipakai untuk dua hal:
- kolom Status untuk izin dari admin (Pending, Approved, Rejected)
- kolom Percobaan Gratis untuk waktu mulai coba gratis, jadi satu email cuma bisa coba sekali
- kolom Role untuk peran akun, User atau Admin. Admin bisa memberikan role Admin ke akun lain

Fungsi di sini cuma butuh objek worksheet dari gspread, jadi gampang dites pakai worksheet palsu.
"""
from datetime import datetime, timedelta

BASE_HEADERS = ["Email", "Nama", "Tanggal Daftar", "Status", "Tanggal Persetujuan"]
TRIAL_COLUMN = "Percobaan Gratis"
ROLE_COLUMN = "Role"
APPROVAL_HEADERS = BASE_HEADERS + [TRIAL_COLUMN, ROLE_COLUMN]
ROLES = ["User", "Admin"]
STATUS_CELL = "D"
TRIAL_CELL = "F"
ROLE_CELL = "G"


def ensure_headers(worksheet):
    """Pastikan baris judul sesuai. Sheet lama yang belum punya kolom Percobaan Gratis atau Role ditambah otomatis."""
    header = [str(cell).strip() for cell in worksheet.row_values(1)]
    if not header:
        worksheet.append_row(APPROVAL_HEADERS, value_input_option="RAW")
        return
    if header == APPROVAL_HEADERS:
        return
    if header not in (BASE_HEADERS, BASE_HEADERS + [TRIAL_COLUMN]):
        raise RuntimeError("Header Google Sheets tidak sesuai. Gunakan: " + ", ".join(APPROVAL_HEADERS))
    if worksheet.col_count < len(APPROVAL_HEADERS):
        worksheet.add_cols(len(APPROVAL_HEADERS) - worksheet.col_count)
    missing = APPROVAL_HEADERS[len(header):]
    worksheet.update([missing], range_name=f"{chr(ord('A') + len(header))}1", value_input_option="RAW")


def role_of(record):
    """Role satu baris. Kosong atau isian lain dianggap User."""
    value = record.get(ROLE_COLUMN, "") if record is not None else ""
    return "Admin" if str(value).strip().lower() == "admin" else "User"


def clean_records(worksheet):
    """Semua baris sheet sebagai dict dengan teks yang sudah dirapikan."""
    return [{str(key).strip(): str(value).strip() for key, value in record.items()}
            for record in worksheet.get_all_records()]


def find_row(records, email):
    """Mencari posisi baris untuk satu email. Hasilnya (index, record) atau (None, None)."""
    wanted = str(email).strip().lower()
    matches = [(index, record) for index, record in enumerate(records)
               if record.get("Email", "").lower() == wanted]
    if len(matches) > 1:
        raise RuntimeError("Duplicate approval records")
    return matches[0] if matches else (None, None)


def parse_time(text):
    """Ubah teks waktu ISO di sheet jadi datetime. Kosong atau rusak hasilnya None."""
    try:
        value = datetime.fromisoformat(str(text).strip())
    except ValueError:
        return None
    return value if value.tzinfo else None


def trial_window(record, minutes):
    """Waktu mulai dan selesai coba gratis dari satu baris, atau None kalau belum pernah coba."""
    started = parse_time((record or {}).get(TRIAL_COLUMN, ""))
    if started is None:
        return None
    return started, started + timedelta(minutes=minutes)


def claim_trial(worksheet, email, name, now):
    """Mencatat mulai coba gratis untuk satu email, cuma boleh sekali.

    Hasilnya "started" kalau berhasil, "used" kalau email ini sudah pernah coba,
    atau "rejected" kalau aksesnya sudah ditolak admin.
    """
    records = clean_records(worksheet)
    index, record = find_row(records, email)
    stamp = now.isoformat(timespec="seconds")
    if record is None:
        # akun baru: baris dibuat dengan Status kosong (belum mengajukan akses)
        worksheet.append_row([email, name, stamp, "", "", stamp, "User"], value_input_option="RAW")
        return "started"
    if record.get(TRIAL_COLUMN):
        return "used"
    if record.get("Status", "").lower() == "rejected":
        return "rejected"
    worksheet.update([[stamp]], range_name=f"{TRIAL_CELL}{index + 2}", value_input_option="RAW")
    return "started"


def request_access(worksheet, email, name, now):
    """Mengajukan akses ke admin. Baris yang dibuat waktu coba gratis cukup diubah statusnya jadi Pending."""
    records = clean_records(worksheet)
    index, record = find_row(records, email)
    if record is None:
        worksheet.append_row([email, name, now.isoformat(timespec="seconds"), "Pending", "", "", "User"],
                             value_input_option="RAW")
        return {"Email": email, "Status": "Pending"}
    if not record.get("Status"):
        worksheet.update([["Pending"]], range_name=f"{STATUS_CELL}{index + 2}", value_input_option="RAW")
        record["Status"] = "Pending"
    return record


def set_role(worksheet, email, role, now):
    """Mengubah role satu akun. Role Admin sekaligus menyetujui akses akun tersebut.

    Hasilnya semua baris sheet setelah diubah, dipakai untuk memperbarui tabel di Panel Admin.
    """
    if role not in ROLES:
        raise ValueError("Role tidak valid.")
    records = clean_records(worksheet)
    index, record = find_row(records, email)
    if record is None:
        raise ValueError("Akun tidak ditemukan.")
    row = index + 2
    if role == "Admin" and record.get("Status", "").lower() != "approved":
        stamp = now.isoformat(timespec="seconds")
        worksheet.update([["Approved", stamp]], range_name=f"{STATUS_CELL}{row}:E{row}", value_input_option="RAW")
        record["Status"], record["Tanggal Persetujuan"] = "Approved", stamp
    worksheet.update([[role]], range_name=f"{ROLE_CELL}{row}", value_input_option="RAW")
    record[ROLE_COLUMN] = role
    return records


def add_user(worksheet, email, name, role, now):
    """Admin mendaftarkan email baru yang langsung disetujui, dengan role User atau Admin."""
    if role not in ROLES:
        raise ValueError("Role tidak valid.")
    records = clean_records(worksheet)
    _, record = find_row(records, email)
    if record is not None:
        raise ValueError("Email ini sudah terdaftar. Ubah role atau status akunnya dari tabel pengguna.")
    stamp = now.isoformat(timespec="seconds")
    worksheet.append_row([email, name, stamp, "Approved", stamp, "", role], value_input_option="RAW")
    return clean_records(worksheet)
