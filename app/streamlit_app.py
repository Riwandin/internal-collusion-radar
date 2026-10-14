"""Internal Collusion Radar — dashboard Streamlit dengan login 2 role (investigator & manajemen).

Jalankan langsung:  streamlit run app/streamlit_app.py
Tanpa Docker, tanpa MLflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Collusion Radar", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from radar_app.auth import ROLE_LABELS, demo_users, verify_login  # noqa: E402
from radar_app.config import SETTINGS  # noqa: E402
from radar_app.ui import components as ui  # noqa: E402
from radar_app.ui import services  # noqa: E402
from radar_app.views import (assistant, engine, impact, inv_overview, mgmt_overview, mlops, network, run_model)  # noqa: E402

ui.inject_css()


def login_page() -> None:
    left, center, right = st.columns([1, 1.4, 1])
    with center:
        st.markdown("## 🛡️ Internal Collusion Radar")
        st.markdown("<div class='page-subtitle'>Deteksi indikasi kolusi requester ↔ approver pada persetujuan KUR</div>",
                    unsafe_allow_html=True)

        st.markdown("##### Masuk Cepat (1-Klik Demo)")
        q1, q2 = st.columns(2)
        users = {u["role"]: u for u in demo_users()}

        if q1.button("Masuk: Investigator", type="primary", use_container_width=True):
            user = {"username": "investigator", "display_name": users["investigator"]["display_name"], "role": "investigator"}
            services.get_store().record_login("investigator", "investigator", True)
            st.session_state["user"] = user
            st.rerun()

        if q2.button("Masuk: Manajemen", type="secondary", use_container_width=True):
            user = {"username": "manajemen", "display_name": users["manajemen"]["display_name"], "role": "manajemen"}
            services.get_store().record_login("manajemen", "manajemen", True)
            st.session_state["user"] = user
            st.rerun()

        st.markdown("<div style='text-align: center; color: #6e7781; margin: 12px 0;'>— atau masukkan kredensial manual —</div>", unsafe_allow_html=True)

        with st.form("login"):
            username = st.text_input("Username", placeholder="investigator / manajemen")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Masuk dengan Password", use_container_width=True)

        if submitted:
            store = services.get_store()
            user = verify_login(username, password, store.engine)
            store.record_login(username or "-", user["role"] if user else "-", user is not None)
            if user:
                st.session_state["user"] = user
                st.rerun()
            st.error("Username atau password salah.")

        st.markdown(
            "<div class='small-note'>Akun demo: <b>investigator</b> / <b>investigator123</b> · <b>manajemen</b> / "
            "<b>manajemen123</b>. Berjalan langsung secara lokal tanpa Docker dan tanpa MLflow.</div>",
            unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.markdown(
            ui.kpi_card(
                "Investigator (5 Menu Fokus)",
                "Operasional",
                "Meja Investigasi 360° (skor 4 pilar + Dossier PDF), Peta Jaringan, Cek Berkas Baru, Risk Engine, Copilot AI",
                ui.ORANGE,
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            ui.kpi_card(
                "Manajemen (5 Menu Fokus)",
                "Strategis",
                "Command Center, Simulasi Dampak & Kapasitas, Risk Engine 4 Pilar, Kesehatan Sistem & Tata Kelola, Copilot AI",
                ui.BLUE,
            ),
            unsafe_allow_html=True,
        )


def sidebar_status(user: dict) -> None:
    with st.sidebar:
        st.markdown(f"**{user['display_name']}**")
        st.markdown(ui.badge(ROLE_LABELS[user["role"]], ui.ORANGE if user["role"] == "investigator" else ui.BLUE),
                    unsafe_allow_html=True)
        store = services.get_store()
        risk_model = services.get_risk_model()
        st.caption(f"Backend: {store.backend}")
        engine_ready = risk_model is not None and hasattr(risk_model, "pillar_scores")
        st.caption(f"Risk Engine 4 pilar: {'🟢 Aktif' if engine_ready else '🟡 Terbatas (skor dari notebook)'}")
        st.caption(f"Data s.d.: {services.app_settings().get('data_end_date', '-')}")

        llm = services.get_llm()
        llm_ok = llm.has_model(SETTINGS.generator_model)
        st.caption(f"LLM: {'aktif' if llm_ok else 'mode cadangan offline'}")

        if st.button("Keluar", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def page(render, title: str, icon: str, url_path: str, default: bool = False):
    return st.Page(render, title=title, icon=icon, url_path=url_path, default=default)


if "user" not in st.session_state:
    login_page()
    st.stop()

user = services.current_user()
if user["role"] == "investigator":
    pages = {
        "Menu Investigator": [
            page(inv_overview.render, "Overview", ":material/person_search:", "overview", True),
            page(network.render, "Peta Jaringan & Komplotan", ":material/hub:", "jaringan"),
            page(run_model.render, "Cek Berkas Baru (Live Test)", ":material/play_circle:", "live-test"),
            page(engine.render, "Risk Engine 4 Pilar", ":material/account_tree:", "risk-engine"),
            page(assistant.render, "Copilot AI & Regulasi KUR", ":material/smart_toy:", "copilot-ai"),
        ],
    }
else:
    pages = {
        "Menu Manajemen": [
            page(mgmt_overview.render, "Overview Monitoring", ":material/space_dashboard:", "overview-monitoring", True),
            page(impact.render, "Simulasi Dampak Bisnis & Kebijakan", ":material/payments:", "dampak-kebijakan"),
            page(engine.render, "Risk Engine 4 Pilar", ":material/account_tree:", "risk-engine"),
            page(mlops.render, "Kesehatan Sistem & Tata Kelola", ":material/verified_user:", "tata-kelola"),
            page(assistant.render, "Copilot AI & Regulasi KUR", ":material/smart_toy:", "copilot-ai"),
        ],
    }

navigation = st.navigation(pages)
sidebar_status(user)
navigation.run()
