"""Menyimpan bundle model bawaan ke repo GitHub lewat GitHub REST API.

Streamlit Community Cloud tidak menyimpan perubahan file secara permanen.
Jadi setelah admin update model, bundle baru di-commit ke repo dan
Streamlit Cloud reload aplikasi pakai model itu.
"""
import base64

import requests

API_URL = "https://api.github.com"
DEFAULT_PATH = "runtime/baseline_bundle.zip"
DEFAULT_BRANCH = "main"


class GitHubStoreError(RuntimeError):
    """Error waktu menyimpan ke GitHub, pesannya sudah bahasa Indonesia buat ditampilkan ke admin."""
    pass


def github_settings(section):
    """Melengkapi pengaturan [github] dari Secrets dengan branch dan path default."""
    settings = dict(section or {})
    settings.setdefault("branch", DEFAULT_BRANCH)
    settings.setdefault("bundle_path", DEFAULT_PATH)
    return settings


def github_is_configured(settings):
    """Mengecek token dan nama repo (pemilik/nama-repo) sudah diisi atau belum."""
    return bool(settings.get("token")) and "/" in str(settings.get("repository", ""))


def _headers(token):
    """Header standar GitHub REST API plus token."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _explain(response):
    """Mengubah kode status HTTP dari GitHub jadi pesan yang gampang dipahami admin."""
    messages = {
        401: "Token GitHub tidak valid atau sudah kedaluwarsa.",
        403: "Token GitHub belum memiliki izin Contents: Read and write untuk repositori ini.",
        404: "Repositori, branch, atau path GitHub tidak ditemukan. Periksa pengaturan [github] di Secrets.",
        409: "Berkas di GitHub berubah saat penyimpanan. Coba simpan kembali.",
        422: "GitHub menolak isi commit. Periksa nama branch dan path berkas.",
    }
    return messages.get(response.status_code, f"GitHub mengembalikan status {response.status_code}.")


def publish_bundle(content, message, settings, session=None):
    """Commit bundle ke GitHub dan mengembalikan link commit-nya.

    Pertama mengambil "sha" file lama kalau sudah ada, ini wajib buat menimpa file.
    Habis itu mengirim isi file baru dalam base64 pakai PUT.
    """
    if not github_is_configured(settings):
        raise GitHubStoreError("Penyimpanan GitHub belum dikonfigurasi.")
    http = session or requests
    repository = settings["repository"].strip()
    path = settings["bundle_path"].strip("/")
    url = f"{API_URL}/repos/{repository}/contents/{path}"
    headers = _headers(settings["token"])
    current = http.get(url, headers=headers, params={"ref": settings["branch"]}, timeout=30)
    if current.status_code == 200:
        sha = current.json().get("sha")
    elif current.status_code == 404:
        sha = None
    else:
        raise GitHubStoreError(_explain(current))
    body = {
        "message": message,
        "content": base64.b64encode(content).decode("ascii"),
        "branch": settings["branch"],
    }
    if sha:
        body["sha"] = sha
    response = http.put(url, headers=headers, json=body, timeout=60)
    if response.status_code not in (200, 201):
        raise GitHubStoreError(_explain(response))
    return response.json().get("commit", {}).get("html_url", "")
