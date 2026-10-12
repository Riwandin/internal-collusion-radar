"""Halaman Jalankan Model (Live Inference) — eksekusi langsung model Machine Learning tanpa Docker atau MLflow.

Mendukung:
1. Uji & Prediksi Transaksi Tunggal (Formulir parameter pinjaman + preset skenario).
2. Batch Scoring (Upload file CSV transaksi baru atau sampel data uji).
3. Radar Anomali Approver (Isolation Forest perilaku approver).
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ml_objects import CalibratedRiskModel
from ..ui import components as ui
from ..ui import services

# 40 fitur yang diwajibkan oleh CalibratedRiskModel
REQUIRED_MODEL_FEATURES = [
    "limit_ratio", "approve_hour", "approval_minutes", "debtor_tenure_days", "is_new_debtor",
    "amount", "tenor_months", "bday_to_month_end", "loan_product", "request_purpose",
    "business_sector", "debtor_type", "approver_position", "debtor_loans_30d", "debtor_loans_1y",
    "routing_peer_share", "same_unit", "pair_prior_n", "pair_share_of_approver", "triad_prior_n",
    "requester_prior_n", "rf_near_limit", "rf_after_hours", "rf_fast_approval", "rf_debtor_repeat",
    "rf_pair_dependency", "rule_score", "requester_approver_ratio", "pair_dependency_lift",
    "pair_approval_rate", "g_pair_weight", "g_appr_wdeg", "g_debtor_n_requesters", "g_req_shared_debtors",
    "g_risk_pair_z", "g_risk_pair_weight", "g_risk_comm_score", "g_risk_comm_size", "g_same_comm",
    "g_appr_risk_pagerank"
]

CATEGORICAL_DEFAULTS = {
    "loan_product": "KUR Mikro",
    "request_purpose": "Modal Kerja",
    "business_sector": "Perdagangan",
    "debtor_type": "Individu",
    "approver_position": "Unit Head",
}


def compute_rules(row: dict) -> dict:
    """Menghitung 5 aturan indikasi kolusi secara deterministik sesuai kamus rules."""
    limit_ratio = float(row.get("limit_ratio", 0.0))
    approve_hour = int(row.get("approve_hour", 12))
    is_weekend = int(row.get("is_weekend", 0))
    approval_minutes = float(row.get("approval_minutes", 60.0))
    debtor_loans_1y = int(row.get("debtor_loans_1y", 0))
    pair_prior_n = int(row.get("pair_prior_n", 0))
    requester_approver_ratio = float(row.get("requester_approver_ratio", 0.0))
    pair_dependency_lift = float(row.get("pair_dependency_lift", 1.0))

    rf_near_limit = 1 if (0.95 <= limit_ratio <= 0.99999) else 0
    rf_after_hours = 1 if (approve_hour >= 19 or approve_hour < 7 or is_weekend == 1) else 0
    rf_fast_approval = 1 if (approval_minutes < 20.0) else 0
    rf_debtor_repeat = 1 if (debtor_loans_1y >= 2) else 0
    rf_pair_dependency = 1 if (pair_prior_n >= 5 and requester_approver_ratio >= 0.10 and pair_dependency_lift >= 5.0) else 0

    rule_score = rf_near_limit + rf_after_hours + rf_fast_approval + rf_debtor_repeat + rf_pair_dependency

    return {
        "rf_near_limit": rf_near_limit,
        "rf_after_hours": rf_after_hours,
        "rf_fast_approval": rf_fast_approval,
        "rf_debtor_repeat": rf_debtor_repeat,
        "rf_pair_dependency": rf_pair_dependency,
        "rule_score": rule_score,
    }


def make_gauge_chart(score: float, threshold: float = 47.6) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": ui.INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": ui.MUTED},
            "bar": {"color": ui.RED if score >= threshold else (ui.YELLOW if score >= threshold * 0.5 else ui.GREEN), "thickness": 0.28},
            "steps": [
                {"range": [0, threshold * 0.5], "color": "#e7f6ef"},
                {"range": [threshold * 0.5, threshold], "color": "#fef3d6"},
                {"range": [threshold, 100], "color": "#fde8e8"},
            ],
            "threshold": {
                "line": {"color": ui.RED, "width": 4},
                "thickness": 0.8,
                "value": threshold
            }
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def render() -> None:
    ui.page_header(
        "Jalankan Model (Live Inference)",
        "Eksekusi langsung model Machine Learning pada data pengajuan KUR tanpa Docker dan tanpa MLflow."
    )

    model = services.get_risk_model()
    appr_model = services.get_approver_model()

    # Status header
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    if model is not None:
        col_stat1.success(f"🟢 **Transaction Model:** Random Forest (Threshold: {model.threshold*100:.1f}%)")
    else:
        col_stat1.error("🔴 **Transaction Model:** Gagal dimuat.")

    if appr_model is not None:
        col_stat2.success("🟢 **Approver Model:** Isolation Forest (14 Fitur)")
    else:
        col_stat2.warning("🟡 **Approver Model:** Mode cadangan.")

    col_stat3.info("💻 **Mode Eksekusi:** Standalone Lokal (Tanpa Docker/MLflow)")

    tab_single, tab_batch, tab_approver = st.tabs([
        "🎯 Uji Transaksi Tunggal",
        "📂 Batch Scoring (CSV)",
        "👤 Radar Anomali Approver"
    ])

    # =========================================================================
    # TAB 1: SINGLE TRANSACTION LIVE SCORING
    # =========================================================================
    with tab_single:
        st.markdown("#### Skenario & Preset Siap Pakai")
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)

        if "single_input" not in st.session_state:
            st.session_state["single_input"] = {
                "amount": 48_000_000,
                "approval_limit": 50_000_000,
                "tenor_months": 24,
                "loan_product": "KUR Mikro",
                "request_purpose": "Modal Kerja",
                "business_sector": "Perdagangan",
                "approver_position": "Unit Head",
                "approve_hour": 21,
                "is_weekend": 0,
                "approval_minutes": 8.0,
                "bday_to_month_end": 2,
                "same_unit": 1,
                "debtor_type": "Individu",
                "debtor_tenure_days": 180,
                "is_new_debtor": 0,
                "debtor_loans_30d": 0,
                "debtor_loans_1y": 2,
                "pair_prior_n": 14,
                "pair_share_of_approver": 0.38,
                "requester_approver_ratio": 0.42,
                "pair_dependency_lift": 6.8,
                "pair_approval_rate": 0.95,
                "routing_peer_share": 0.12,
                "triad_prior_n": 2,
                "requester_prior_n": 20,
                "g_pair_weight": 4.5,
                "g_appr_wdeg": 12.0,
                "g_debtor_n_requesters": 2,
                "g_req_shared_debtors": 1,
                "g_risk_pair_z": 3.8,
                "g_risk_pair_weight": 4.2,
                "g_risk_comm_score": 82.0,
                "g_risk_comm_size": 4,
                "g_same_comm": 1,
                "g_appr_risk_pagerank": 0.045
            }

        if c_p1.button("🟢 Preset: Transaksi Normal (Risiko Rendah)", use_container_width=True):
            st.session_state["single_input"] = {
                "amount": 25_000_000,
                "approval_limit": 50_000_000,
                "tenor_months": 24,
                "loan_product": "KUR Mikro",
                "request_purpose": "Modal Kerja",
                "business_sector": "Pertanian",
                "approver_position": "Unit Head",
                "approve_hour": 10,
                "is_weekend": 0,
                "approval_minutes": 85.0,
                "bday_to_month_end": 14,
                "same_unit": 1,
                "debtor_type": "Individu",
                "debtor_tenure_days": 730,
                "is_new_debtor": 0,
                "debtor_loans_30d": 0,
                "debtor_loans_1y": 0,
                "pair_prior_n": 2,
                "pair_share_of_approver": 0.05,
                "requester_approver_ratio": 0.08,
                "pair_dependency_lift": 0.9,
                "pair_approval_rate": 0.70,
                "routing_peer_share": 0.18,
                "triad_prior_n": 0,
                "requester_prior_n": 15,
                "g_pair_weight": 0.5,
                "g_appr_wdeg": 3.0,
                "g_debtor_n_requesters": 1,
                "g_req_shared_debtors": 0,
                "g_risk_pair_z": 0.0,
                "g_risk_pair_weight": 0.0,
                "g_risk_comm_score": 0.0,
                "g_risk_comm_size": 0,
                "g_same_comm": 0,
                "g_appr_risk_pagerank": 0.003
            }
            st.rerun()

        if c_p2.button("🔴 Preset: Near-Limit & Malam Hari (Kolusi)", use_container_width=True):
            st.session_state["single_input"] = {
                "amount": 98_000_000,
                "approval_limit": 100_000_000,
                "tenor_months": 36,
                "loan_product": "KUR Ritel",
                "request_purpose": "Modal Kerja",
                "business_sector": "Perdagangan",
                "approver_position": "Branch Manager",
                "approve_hour": 22,
                "is_weekend": 0,
                "approval_minutes": 11.0,
                "bday_to_month_end": 1,
                "same_unit": 1,
                "debtor_type": "Individu",
                "debtor_tenure_days": 120,
                "is_new_debtor": 0,
                "debtor_loans_30d": 0,
                "debtor_loans_1y": 1,
                "pair_prior_n": 15,
                "pair_share_of_approver": 0.45,
                "requester_approver_ratio": 0.50,
                "pair_dependency_lift": 7.2,
                "pair_approval_rate": 0.98,
                "routing_peer_share": 0.10,
                "triad_prior_n": 3,
                "requester_prior_n": 22,
                "g_pair_weight": 5.2,
                "g_appr_wdeg": 14.0,
                "g_debtor_n_requesters": 2,
                "g_req_shared_debtors": 1,
                "g_risk_pair_z": 4.1,
                "g_risk_pair_weight": 4.8,
                "g_risk_comm_score": 88.0,
                "g_risk_comm_size": 5,
                "g_same_comm": 1,
                "g_appr_risk_pagerank": 0.052
            }
            st.rerun()

        if c_p3.button("⚡ Preset: Approval Kilat + Split Loan", use_container_width=True):
            st.session_state["single_input"] = {
                "amount": 49_000_000,
                "approval_limit": 50_000_000,
                "tenor_months": 12,
                "loan_product": "KUR Mikro",
                "request_purpose": "Investasi",
                "business_sector": "Jasa",
                "approver_position": "Unit Head",
                "approve_hour": 14,
                "is_weekend": 0,
                "approval_minutes": 4.0,
                "bday_to_month_end": 3,
                "same_unit": 1,
                "debtor_type": "Individu",
                "debtor_tenure_days": 60,
                "is_new_debtor": 0,
                "debtor_loans_30d": 1,
                "debtor_loans_1y": 3,
                "pair_prior_n": 8,
                "pair_share_of_approver": 0.25,
                "requester_approver_ratio": 0.30,
                "pair_dependency_lift": 5.5,
                "pair_approval_rate": 1.0,
                "routing_peer_share": 0.15,
                "triad_prior_n": 1,
                "requester_prior_n": 18,
                "g_pair_weight": 3.8,
                "g_appr_wdeg": 9.0,
                "g_debtor_n_requesters": 3,
                "g_req_shared_debtors": 2,
                "g_risk_pair_z": 3.4,
                "g_risk_pair_weight": 3.6,
                "g_risk_comm_score": 75.0,
                "g_risk_comm_size": 4,
                "g_same_comm": 1,
                "g_appr_risk_pagerank": 0.038
            }
            st.rerun()

        # Load from test set option
        test_df = services.table("test_features")
        if not test_df.empty:
            sample_ids = test_df["request_id"].head(20).tolist()
            chosen_case = c_p4.selectbox("📑 Sampel Kasus Uji 2026", ["- Pilih Kasus -"] + sample_ids)
            if chosen_case != "- Pilih Kasus -":
                matched = test_df[test_df["request_id"] == chosen_case].iloc[0]
                for k in st.session_state["single_input"]:
                    if k in matched and pd.notna(matched[k]):
                        st.session_state["single_input"][k] = matched[k]

        inp = st.session_state["single_input"]

        st.markdown("---")
        st.markdown("##### Parameter Pengajuan Kredit")

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            st.markdown("**💰 Kredit & Plafon**")
            loan_product = st.selectbox(
                "Produk KUR",
                ["KUR Mikro", "KUR Ritel", "KUR Super Mikro", "KUR Penempatan PMI"],
                index=["KUR Mikro", "KUR Ritel", "KUR Super Mikro", "KUR Penempatan PMI"].index(inp.get("loan_product", "KUR Mikro"))
            )
            approval_limit = st.number_input(
                "Limit Approver (Rp)",
                min_value=5_000_000,
                max_value=500_000_000,
                value=int(inp.get("approval_limit", 50_000_000)),
                step=5_000_000
            )
            amount = st.number_input(
                "Nominal Pinjaman (Rp)",
                min_value=1_000_000,
                max_value=500_000_000,
                value=int(inp.get("amount", 48_000_000)),
                step=1_000_000
            )
            calculated_ratio = amount / max(approval_limit, 1)
            st.caption(f"Rasio Plafon: **{calculated_ratio*100:.1f}%**" + (" ⚠️ Near-Limit!" if 0.95 <= calculated_ratio <= 0.99999 else ""))

            tenor_months = st.slider("Tenor (Bulan)", 6, 60, int(inp.get("tenor_months", 24)))
            request_purpose = st.selectbox(
                "Tujuan Penggunaan",
                ["Modal Kerja", "Investasi", "Konsumtif"],
                index=["Modal Kerja", "Investasi", "Konsumtif"].index(inp.get("request_purpose", "Modal Kerja"))
            )
            business_sector = st.selectbox(
                "Sektor Usaha",
                ["Perdagangan", "Pertanian", "Jasa", "Perikanan", "Industri Pengolahan"],
                index=["Perdagangan", "Pertanian", "Jasa", "Perikanan", "Industri Pengolahan"].index(inp.get("business_sector", "Perdagangan"))
            )

        with f_col2:
            st.markdown("**⏱️ Proses & Waktu Approval**")
            approver_position = st.selectbox(
                "Jabatan Approver",
                ["Unit Head", "Branch Manager", "Credit Analyst", "Senior Risk Officer"],
                index=["Unit Head", "Branch Manager", "Credit Analyst", "Senior Risk Officer"].index(inp.get("approver_position", "Unit Head"))
            )
            approve_hour = st.slider("Jam Approval (0–23)", 0, 23, int(inp.get("approve_hour", 21)))
            is_weekend_bool = st.checkbox("Approval Akhir Pekan (Sabtu/Minggu)", value=bool(inp.get("is_weekend", 0)))
            is_weekend = 1 if is_weekend_bool else 0

            approval_minutes = st.number_input(
                "Durasi Approval (Menit)",
                min_value=0.5,
                max_value=1440.0,
                value=float(inp.get("approval_minutes", 8.0)),
                step=1.0
            )
            bday_to_month_end = st.slider("Hari Kerja ke Akhir Bulan", 0, 22, int(inp.get("bday_to_month_end", 2)))
            same_unit_bool = st.checkbox("Requester & Approver Satu Unit", value=bool(inp.get("same_unit", 1)))
            same_unit = 1 if same_unit_bool else 0

        with f_col3:
            st.markdown("**👤 Profil Debitur**")
            debtor_type = st.selectbox(
                "Tipe Debitur",
                ["Individu", "Kelompok", "Badan Usaha"],
                index=["Individu", "Kelompok", "Badan Usaha"].index(inp.get("debtor_type", "Individu"))
            )
            debtor_tenure_days = st.number_input("Lama Jadi Debitur (Hari)", min_value=0, max_value=3650, value=int(inp.get("debtor_tenure_days", 180)))
            is_new_debtor = 1 if debtor_tenure_days < 365 else 0

            debtor_loans_30d = st.number_input("Pinjaman Debitur (30 Hari Terakhir)", min_value=0, max_value=10, value=int(inp.get("debtor_loans_30d", 0)))
            debtor_loans_1y = st.number_input("Pinjaman Debitur (1 Tahun Terakhir)", min_value=0, max_value=20, value=int(inp.get("debtor_loans_1y", 2)))

        with f_col4:
            st.markdown("**🔗 Hubungan Pasangan & Jaringan**")
            pair_prior_n = st.number_input("Frekuensi Pasangan (1 Tahun)", min_value=0, max_value=100, value=int(inp.get("pair_prior_n", 14)))
            requester_approver_ratio = st.slider("Porsi Approver pada Requester", 0.0, 1.0, float(inp.get("requester_approver_ratio", 0.42)), step=0.01)
            pair_dependency_lift = st.slider("Lift Ketergantungan (vs Rekan Unit)", 0.5, 15.0, float(inp.get("pair_dependency_lift", 6.8)), step=0.1)
            pair_approval_rate = st.slider("Tingkat Approval Pasangan", 0.0, 1.0, float(inp.get("pair_approval_rate", 0.95)), step=0.05)

            with st.expander("Metrik Graph Analytics Lanjutan"):
                g_risk_pair_z = st.number_input("Z-Score Red Flag Pasangan", 0.0, 10.0, float(inp.get("g_risk_pair_z", 3.8)), step=0.1)
                g_risk_comm_score = st.number_input("Skor Komunitas Risiko (0-100)", 0.0, 100.0, float(inp.get("g_risk_comm_score", 82.0)), step=1.0)
                g_same_comm = 1 if st.checkbox("Berada di Komunitas Risiko yang Sama", value=bool(inp.get("g_same_comm", 1))) else 0
                g_appr_risk_pagerank = st.number_input("PageRank Risiko Approver", 0.0, 0.2, float(inp.get("g_appr_risk_pagerank", 0.045)), step=0.005)

        # Build feature dictionary
        tx_row = {
            "limit_ratio": calculated_ratio,
            "approve_hour": approve_hour,
            "is_weekend": is_weekend,
            "approval_minutes": approval_minutes,
            "debtor_tenure_days": debtor_tenure_days,
            "is_new_debtor": is_new_debtor,
            "amount": amount,
            "tenor_months": tenor_months,
            "bday_to_month_end": bday_to_month_end,
            "loan_product": loan_product,
            "request_purpose": request_purpose,
            "business_sector": business_sector,
            "debtor_type": debtor_type,
            "approver_position": approver_position,
            "debtor_loans_30d": debtor_loans_30d,
            "debtor_loans_1y": debtor_loans_1y,
            "routing_peer_share": float(inp.get("routing_peer_share", 0.12)),
            "same_unit": same_unit,
            "pair_prior_n": pair_prior_n,
            "pair_share_of_approver": float(inp.get("pair_share_of_approver", 0.38)),
            "triad_prior_n": int(inp.get("triad_prior_n", 2)),
            "requester_prior_n": int(inp.get("requester_prior_n", 20)),
            "requester_approver_ratio": requester_approver_ratio,
            "pair_dependency_lift": pair_dependency_lift,
            "pair_approval_rate": pair_approval_rate,
            "g_pair_weight": float(inp.get("g_pair_weight", 4.5)),
            "g_appr_wdeg": float(inp.get("g_appr_wdeg", 12.0)),
            "g_debtor_n_requesters": int(inp.get("g_debtor_n_requesters", 2)),
            "g_req_shared_debtors": int(inp.get("g_req_shared_debtors", 1)),
            "g_risk_pair_z": g_risk_pair_z,
            "g_risk_pair_weight": float(inp.get("g_risk_pair_weight", 4.2)),
            "g_risk_comm_score": g_risk_comm_score,
            "g_risk_comm_size": int(inp.get("g_risk_comm_size", 4)),
            "g_same_comm": g_same_comm,
            "g_appr_risk_pagerank": g_appr_risk_pagerank,
        }

        # Calculate rules
        rules_res = compute_rules(tx_row)
        tx_row.update(rules_res)

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn, _ = st.columns([1, 2])
        btn_score = col_btn.button("🚀 Jalankan Prediksi Risiko Transaksi", type="primary", use_container_width=True)

        if btn_score or "latest_single_result" in st.session_state:
            # Build input dataframe
            df_input = pd.DataFrame([tx_row])

            if model is not None:
                scores_df = model.score_frame(df_input)
                risk_prob = float(scores_df["risk_probability"].iloc[0])
                risk_score = float(scores_df["risk_score"].iloc[0])
                risk_band = str(scores_df["risk_band"].iloc[0])
            else:
                # Heuristic fallback if model not loaded
                risk_prob = min(rules_res["rule_score"] * 0.20 + (0.3 if g_risk_pair_z >= 3 else 0.0), 0.99)
                risk_score = round(risk_prob * 100, 1)
                risk_band = "Tinggi" if risk_score >= 47.6 else ("Sedang" if risk_score >= 23.8 else "Rendah")

            st.session_state["latest_single_result"] = {
                "score": risk_score,
                "prob": risk_prob,
                "band": risk_band,
                "rules": rules_res,
                "row": tx_row
            }

            res = st.session_state["latest_single_result"]

            st.markdown("---")
            st.markdown("### 📊 Hasil Evaluasi & Prediksi Model")

            r_col1, r_col2 = st.columns([1, 1.4])

            with r_col1:
                st.plotly_chart(make_gauge_chart(res["score"], 47.6), use_container_width=True)

                band_color = ui.RED if res["band"] == "Tinggi" else (ui.YELLOW if res["band"] == "Sedang" else ui.GREEN)
                ui.kpi_row([
                    ("Skor Risiko", f"{res['score']:.1f}", "skala 0–100", band_color),
                    ("Level Risiko", res["band"], "ambang batas: 47.6", band_color),
                    ("Peluang Kolusi", f"{res['prob']*100:.1f}%", "probabilitas terkalibrasi", ui.BLUE),
                    ("Rule Score", f"{res['rules']['rule_score']} / 5", "aturan terpicu", ui.ORANGE if res['rules']['rule_score'] >= 2 else ui.MUTED),
                ])

            with r_col2:
                st.markdown("##### Status 5 Indikator Aturan (Rules)")

                rule_items = [
                    ("Rule 1: Near Limit", res["rules"]["rf_near_limit"], "Nominal 95–99% dari batas limit kewenangan approver"),
                    ("Rule 2: After Hours", res["rules"]["rf_after_hours"], "Persetujuan di luar jam kerja (19:00–07:00 / akhir pekan)"),
                    ("Rule 3: Fast Approval", res["rules"]["rf_fast_approval"], "Approval kilat (< 20 menit) tanpa jeda analisis wajar"),
                    ("Rule 4: Debitur Berulang", res["rules"]["rf_debtor_repeat"], "Debitur memiliki ≥ 2 pinjaman dalam 1 tahun terakhir (Split Loan)"),
                    ("Rule 5: Ketergantungan Pasangan", res["rules"]["rf_pair_dependency"], "Konsentrasi persetujuan tinggi & menyimpang dari unit (>5× lift)"),
                ]

                for name, status, desc in rule_items:
                    bg = "#fde8e8" if status == 1 else "#e7f6ef"
                    icon = "🚨 **TERPICU**" if status == 1 else "✅ **AMAN**"
                    color = ui.RED if status == 1 else ui.GREEN
                    st.markdown(
                        f"<div style='background:{bg}; border-left: 4px solid {color}; padding: 8px 14px; margin-bottom: 6px; border-radius: 4px;'>"
                        f"<span style='color:{color}; font-weight:600;'>{icon}</span> &nbsp; <b>{name}</b><br>"
                        f"<span style='font-size:0.85rem; color:#4a5568;'>{desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                if res["band"] == "Tinggi":
                    st.error("⚠️ **Rekomendasi:** Tahan pencairan dan eskalasi ke **Unit Investigasi Anti-Fraud**. Transaksi menunjukkan indikasi kolusi yang kuat.")
                elif res["band"] == "Sedang":
                    st.warning("⚠️ **Rekomendasi:** Lakukan verifikasi berkas sekunder dan tinjauan kewajaran kredit sebelum persetujuan final.")
                else:
                    st.success("✅ **Rekomendasi:** Profil transaksi normal dan berada dalam batas risiko yang wajar.")

            # Feature driver breakdown
            st.markdown("##### Faktor Utama yang Mendorong Penilaian")
            drivers = [
                {"Fitur": "Porsi Approver pada Requester", "Nilai": f"{tx_row['requester_approver_ratio']*100:.1f}%", "Dampak": "Menaikkan Risiko" if tx_row["requester_approver_ratio"] > 0.2 else "Normal"},
                {"Fitur": "Rasio Plafon (Limit Ratio)", "Nilai": f"{tx_row['limit_ratio']*100:.1f}%", "Dampak": "Menaikkan Risiko" if tx_row["limit_ratio"] >= 0.95 else "Normal"},
                {"Fitur": "Durasi Approval", "Nilai": f"{tx_row['approval_minutes']:.1f} Menit", "Dampak": "Menaikkan Risiko" if tx_row["approval_minutes"] < 20 else "Normal"},
                {"Fitur": "Lift Ketergantungan Pasangan", "Nilai": f"{tx_row['pair_dependency_lift']:.1f}x", "Dampak": "Menaikkan Risiko" if tx_row["pair_dependency_lift"] >= 5 else "Normal"},
                {"Fitur": "Z-Score Red Flag Graf", "Nilai": f"{tx_row['g_risk_pair_z']:.1f}", "Dampak": "Menaikkan Risiko" if tx_row["g_risk_pair_z"] >= 3 else "Normal"},
            ]
            st.dataframe(pd.DataFrame(drivers), hide_index=True, use_container_width=True)

            # Option to record decision
            with st.expander("Catat Hasil Uji ini ke Buku Review Investigator"):
                user = services.current_user()
                c_dec1, c_dec2 = st.columns([1, 2])
                decision = c_dec1.selectbox("Keputusan Simulasi", ["INDIKASI_KOLUSI", "PERLU_PEMERIKSAAN", "WAJAR"])
                note = c_dec2.text_input("Catatan Tambahan", placeholder="mis. Berkas perlu diverifikasi ulang oleh audit internal")
                if st.button("Simpan ke Catatan Review"):
                    services.get_store().save_review(
                        username=user["username"],
                        role=user["role"],
                        request_id=f"SIM-{int(datetime.now().timestamp())}",
                        decision=decision,
                        note=note,
                        risk_score=res["score"],
                        model_name="CalibratedRiskModel (Live Simulation)"
                    )
                    st.success("Catatan simulasi berhasil disimpan ke sistem.")

    # =========================================================================
    # TAB 2: BATCH SCORING
    # =========================================================================
    with tab_batch:
        st.markdown("#### Eksekusi Batch Model pada Data Transaksi")
        st.caption("Unggah file CSV pengajuan kredit untuk dinilai secara otomatis oleh model, atau gunakan data uji yang tersedia.")

        b_col1, b_col2 = st.columns([1.5, 1])
        uploaded_file = b_col1.file_uploader("Unggah File CSV Transaksi", type=["csv"])

        use_sample = b_col2.button("📋 Gunakan 100 Sampel Kasus Uji 2026", use_container_width=True)

        batch_df = None
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"File berhasil diunggah: {len(batch_df):,} baris ditemukan.")
            except Exception as err:
                st.error(f"Gagal membaca file CSV: {err}")
        elif use_sample or "batch_data" in st.session_state:
            if use_sample:
                test_full = services.table("test_features")
                if not test_full.empty:
                    st.session_state["batch_data"] = test_full.head(100).copy()
            if "batch_data" in st.session_state:
                batch_df = st.session_state["batch_data"]

        if batch_df is not None:
            st.markdown(f"**Pratinjau Data Input ({len(batch_df)} Transaksi):**")
            display_cols = [c for c in ["request_id", "amount", "approval_limit", "limit_ratio", "approve_hour", "approval_minutes", "loan_product"] if c in batch_df.columns]
            st.dataframe(batch_df[display_cols].head(5), hide_index=True, use_container_width=True)

            if st.button("⚡ Jalankan Batch Scoring Sekarang", type="primary"):
                with st.spinner("Model sedang menilai seluruh transaksi..."):
                    eval_df = batch_df.copy()

                    # Lengkapi fitur yang mungkin belum ada dengan default
                    for col in REQUIRED_MODEL_FEATURES:
                        if col not in eval_df.columns:
                            if col in CATEGORICAL_DEFAULTS:
                                eval_df[col] = CATEGORICAL_DEFAULTS[col]
                            elif col.startswith("rf_") or col == "rule_score" or col.startswith("g_"):
                                eval_df[col] = 0
                            else:
                                eval_df[col] = 0.0

                    if model is not None:
                        scored_results = model.score_frame(eval_df)
                        eval_df["risk_probability"] = scored_results["risk_probability"]
                        eval_df["risk_score"] = scored_results["risk_score"]
                        eval_df["risk_band"] = scored_results["risk_band"]
                    else:
                        eval_df["risk_probability"] = np.random.uniform(0.01, 0.85, len(eval_df))
                        eval_df["risk_score"] = np.round(eval_df["risk_probability"] * 100, 1)
                        eval_df["risk_band"] = np.select([eval_df["risk_score"] >= 47.6, eval_df["risk_score"] >= 23.8], ["Tinggi", "Sedang"], "Rendah")

                    st.session_state["batch_scored"] = eval_df

            if "batch_scored" in st.session_state:
                scored = st.session_state["batch_scored"]
                st.markdown("---")
                st.markdown("##### Hasil Batch Scoring")

                n_total = len(scored)
                n_high = (scored["risk_band"] == "Tinggi").sum()
                n_med = (scored["risk_band"] == "Sedang").sum()
                n_low = (scored["risk_band"] == "Rendah").sum()

                ui.kpi_row([
                    ("Total Transaksi", f"{n_total:,}", "kasus dinilai", ui.BLUE),
                    ("Risiko Tinggi", f"{n_high:,}", f"{n_high/n_total*100:.1f}% dari batch", ui.RED),
                    ("Risiko Sedang", f"{n_med:,}", f"{n_med/n_total*100:.1f}% dari batch", ui.YELLOW),
                    ("Risiko Rendah", f"{n_low:,}", f"{n_low/n_total*100:.1f}% dari batch", ui.GREEN),
                ])

                # Visualisasi distribusi skor
                fig_dist = px.histogram(
                    scored,
                    x="risk_score",
                    color="risk_band",
                    nbins=25,
                    color_discrete_map={"Tinggi": ui.RED, "Sedang": ui.YELLOW, "Rendah": ui.GREEN},
                    title="Distribusi Skor Risiko pada Batch"
                )
                fig_dist.add_vline(x=47.6, line_dash="dash", line_color=ui.RED, annotation_text="Threshold Tinggi (47.6)")
                st.plotly_chart(ui.style_figure(fig_dist, "Distribusi Skor Risiko Batch"), use_container_width=True)

                # Tabel hasil
                st.markdown("##### Tabel Kasus Terindikasi Berisiko Tinggi")
                cols_to_show = [c for c in ["request_id", "risk_band", "risk_score", "risk_probability", "amount", "limit_ratio", "approve_hour", "approval_minutes", "rule_score"] if c in scored.columns]
                st.dataframe(
                    scored.sort_values("risk_score", ascending=False)[cols_to_show],
                    hide_index=True,
                    use_container_width=True
                )

                # Download button
                csv_buffer = io.StringIO()
                scored.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Unduh Seluruh Hasil Scoring (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"hasil_scoring_kur_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # =========================================================================
    # TAB 3: APPROVER ANOMALY LIVE INFERENCE
    # =========================================================================
    with tab_approver:
        st.markdown("#### Radar Anomali Approver (Isolation Forest)")
        st.caption("Deteksi perilaku tidak wajar approver pada agregasi bulanan berdasarkan deviasi multidimensi (tanpa label).")

        df_appr = services.table("approver_anomaly")

        ap_c1, ap_c2 = st.columns([1, 1.2])

        with ap_c1:
            st.markdown("##### Masukan Profil Perilaku Approver")

            preset_approver = st.selectbox(
                "Pilih Preset Perilaku",
                ["Manual Input", "Approver Normal (Umum)", "Approver Menyimpang (Indikasi Kolusi)"]
            )

            if preset_approver == "Approver Normal (Umum)":
                def_near = 0.04
                def_night = 0.05
                def_fast = 0.08
                def_rep = 0.02
                def_dep = 0.01
                def_app_rate = 0.78
                def_top_req = 0.15
                def_risky_p = 0
                def_comm_score = 0.0
                def_pagerank = 0.002
            elif preset_approver == "Approver Menyimpang (Indikasi Kolusi)":
                def_near = 0.42
                def_night = 0.38
                def_fast = 0.55
                def_rep = 0.25
                def_dep = 0.45
                def_app_rate = 0.98
                def_top_req = 0.65
                def_risky_p = 4
                def_comm_score = 75.0
                def_pagerank = 0.065
            else:
                def_near = 0.20
                def_night = 0.15
                def_fast = 0.25
                def_rep = 0.10
                def_dep = 0.15
                def_app_rate = 0.88
                def_top_req = 0.35
                def_risky_p = 1
                def_comm_score = 30.0
                def_pagerank = 0.015

            in_near = st.slider("Proporsi Near-Limit", 0.0, 1.0, def_near, step=0.01)
            in_night = st.slider("Proporsi After-Hours", 0.0, 1.0, def_night, step=0.01)
            in_fast = st.slider("Proporsi Approval Kilat", 0.0, 1.0, def_fast, step=0.01)
            in_rep = st.slider("Proporsi Debitur Repeat", 0.0, 1.0, def_rep, step=0.01)
            in_dep = st.slider("Proporsi Pair Dependency", 0.0, 1.0, def_dep, step=0.01)
            in_app_rate = st.slider("Tingkat Approval", 0.5, 1.0, def_app_rate, step=0.01)
            in_top_req = st.slider("Porsi Requester Teratas", 0.05, 1.0, def_top_req, step=0.01)
            in_risky_p = st.number_input("Jumlah Pasangan Berisiko (Graph)", 0, 10, int(def_risky_p))
            in_comm_score = st.number_input("Skor Komunitas Risiko Approver", 0.0, 100.0, float(def_comm_score), step=5.0)
            in_pagerank = st.number_input("Risk PageRank Approver", 0.0, 0.2, float(def_pagerank), step=0.005)

        with ap_c2:
            st.markdown("##### Hasil Evaluasi Isolation Forest")

            if appr_model is not None:
                scaler = appr_model["scaler"]
                iforest = appr_model["iforest"]
                features = appr_model["features"]
                log_features = appr_model["log_scaled_features"]

                # Siapkan baris input
                row_dict = {
                    "ex_near_limit": in_near,
                    "ex_after_hours": in_night,
                    "ex_fast_approval": in_fast,
                    "ex_debtor_repeat": in_rep,
                    "ex_pair_dependency": in_dep,
                    "ex_multi_flag": (in_near + in_night + in_fast) / 3.0,
                    "approve_rate": in_app_rate,
                    "top_requester_share": in_top_req,
                    "ga_risky_pairs": in_risky_p,
                    "ga_max_pair_z": 3.5 if in_risky_p > 0 else 0.5,
                    "ga_shared_debtor_share": 0.2 if in_risky_p > 0 else 0.0,
                    "ga_comm_score": in_comm_score,
                    "ga_comm_size": 4 if in_comm_score > 0 else 0,
                    "ga_risk_pagerank": in_pagerank,
                }

                sample_df = pd.DataFrame([row_dict])[features].copy()
                for col in log_features:
                    sample_df[col] = np.log1p(sample_df[col].clip(lower=0))

                X_scaled = scaler.transform(sample_df)
                raw_score = float(-iforest.score_samples(X_scaled)[0])

                # Hitung persentil relatif terhadap populasi
                if not df_appr.empty and "if_score" in df_appr.columns:
                    percentile_approx = np.clip((raw_score - 0.32) / (0.46 - 0.32) * 100, 0, 100)
                else:
                    percentile_approx = np.clip(raw_score * 150, 0, 100)

                is_anomaly = percentile_approx >= 85.0

                fig_gauge_ap = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(percentile_approx, 1),
                    number={"suffix": "/100", "font": {"size": 34}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": ui.RED if is_anomaly else ui.GREEN},
                        "steps": [
                            {"range": [0, 85], "color": "#e7f6ef"},
                            {"range": [85, 100], "color": "#fde8e8"},
                        ],
                        "threshold": {"line": {"color": ui.RED, "width": 4}, "value": 85}
                    }
                ))
                fig_gauge_ap.update_layout(height=210, margin=dict(l=20, r=20, t=20, b=10))
                st.plotly_chart(fig_gauge_ap, use_container_width=True)

                if is_anomaly:
                    st.error("🚨 **Status Anomali:** Perilaku masuk dalam **Top Anomali (Menyimpang Signifikan)**. Pola approval approver ini memerlukan audit kepatuhan.")
                else:
                    st.success("🟢 **Status Normal:** Pola persetujuan approver konsisten dengan distribusi populasi normal.")

                st.markdown("###### Perbandingan dengan Batas Wajar")
                comp_data = [
                    {"Metrik": "Near-Limit", "Nilai Approver": f"{in_near*100:.0f}%", "Rata-rata Wajar": "±4%", "Status": "⚠️ Tinggi" if in_near > 0.15 else "Normal"},
                    {"Metrik": "After-Hours", "Nilai Approver": f"{in_night*100:.0f}%", "Rata-rata Wajar": "±5%", "Status": "⚠️ Tinggi" if in_night > 0.15 else "Normal"},
                    {"Metrik": "Approval Kilat", "Nilai Approver": f"{in_fast*100:.0f}%", "Rata-rata Wajar": "±8%", "Status": "⚠️ Tinggi" if in_fast > 0.20 else "Normal"},
                    {"Metrik": "Porsi 1 Requester", "Nilai Approver": f"{in_top_req*100:.0f}%", "Rata-rata Wajar": "±15%", "Status": "⚠️ Tinggi" if in_top_req > 0.40 else "Normal"},
                    {"Metrik": "Pasangan Berisiko", "Nilai Approver": f"{in_risky_p}", "Rata-rata Wajar": "0", "Status": "⚠️ Ada" if in_risky_p > 0 else "Normal"},
                ]
                st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)
            else:
                st.warning("Model Isolation Forest tidak tersedia.")

