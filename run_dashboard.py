"""Script Peluncur Python untuk Streamlit Dashboard — Internal Collusion Radar.

Menjalankan dashboard secara lokal tanpa Docker dan tanpa MLflow.
Dapat dijalankan langsung dengan:
    python run_dashboard.py
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    app_path = root / "app" / "streamlit_app.py"

    # Prioritas binary streamlit
    venv_streamlit = root / ".venv" / "bin" / "streamlit"
    root_venv_streamlit = root.parent / ".venv" / "bin" / "streamlit"
    bootcamp_streamlit = Path("/Users/wandi/miniforge3/envs/bootcamp/bin/streamlit")
    if venv_streamlit.exists() and os.access(venv_streamlit, os.X_OK):
        streamlit_exec = str(venv_streamlit)
    elif root_venv_streamlit.exists() and os.access(root_venv_streamlit, os.X_OK):
        streamlit_exec = str(root_venv_streamlit)
    elif bootcamp_streamlit.exists() and os.access(bootcamp_streamlit, os.X_OK):
        streamlit_exec = str(bootcamp_streamlit)
    else:
        streamlit_exec = "streamlit"

    print("=" * 66)
    print("🛡️  INTERNAL COLLUSION RADAR v9 — STREAMLIT DASHBOARD")
    print("=" * 66)
    print("  • Mode           : Standalone Lokal (Tanpa Docker & MLflow)")
    print("  • Model Aktif    : Random Forest (KUR) & Isolation Forest (Approver)")
    print("  • Akun Demo      : investigator / investigator123")
    print("                     manajemen / manajemen123")
    print("  • URL Browser    : http://localhost:8501")
    print("=" * 66)
    print("Menjalankan Streamlit...")

    cmd = [
        streamlit_exec,
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "localhost",
        "--server.headless",
        "true"
    ]

    try:
        subprocess.run(cmd, cwd=str(root))
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard dihentikan oleh pengguna.")


if __name__ == "__main__":
    main()

