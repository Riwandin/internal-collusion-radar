#!/usr/bin/env bash
# ==============================================================================
# Peluncur Dashboard Streamlit — Internal Collusion Radar
# Mode Standalone Lokal (Tanpa Docker & Tanpa MLflow)
# ==============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Gunakan environment conda bootcamp yang memiliki dependensi lengkap
if [ -x "/Users/wandi/miniforge3/envs/bootcamp/bin/streamlit" ]; then
    STREAMLIT_EXEC="/Users/wandi/miniforge3/envs/bootcamp/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
    STREAMLIT_EXEC="$(command -v streamlit)"
else
    echo "❌ Error: Streamlit tidak ditemukan."
    echo "Aktifkan conda environment dengan: conda activate bootcamp"
    exit 1
fi

echo "=================================================================="
echo "🛡️  INTERNAL COLLUSION RADAR v9 — STREAMLIT DASHBOARD"
echo "=================================================================="
echo "  • Status         : Berjalan Standalone Lokal"
echo "  • Infrastruktur  : Tanpa Docker & Tanpa MLflow"
echo "  • Model ML       : Random Forest (Kredit) & Isolation Forest (Approver)"
echo "  • Akun Demo      : investigator / investigator123"
echo "                     manajemen / manajemen123"
echo "                     (atau klik tombol 'Masuk Cepat' di halaman login)"
echo "  • Akses URL      : http://localhost:8501"
echo "=================================================================="
echo ""

exec "$STREAMLIT_EXEC" run app/streamlit_app.py --server.port 8501 --server.address localhost

