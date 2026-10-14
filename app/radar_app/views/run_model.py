"""Investigator · Cek Berkas Baru (Live Test): Risk Engine 4 pilar menilai satu pengajuan atau satu batch secara langsung.

Tidak ada skor tiruan di halaman ini: semua angka dihitung oleh `MultiViewRiskEngine` yang sama dengan yang dipakai
notebook (file `outputs/models/transaction_risk_model.joblib`). Pengguna mulai dari kasus nyata test 2026, mengubah
input per pilar (what-if), lalu melihat bagaimana skor tiap pilar, kontribusinya, dan band akhir berubah.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import pillars, services

# Ambang 5 rule SOP — sama dengan notebook (Bagian 0.2 & 1.8)
NEAR_LIMIT_MIN_RATIO, NEAR_LIMIT_MAX_RATIO = 0.95, 0.99999
AFTER_HOURS_START, AFTER_HOURS_END = 19, 7
FAST_APPROVAL_MINUTES = 20
DEBTOR_REPEAT_MIN_PRIOR = 2
DEPENDENCY_MIN_PAIR_TX, DEPENDENCY_MIN_RATIO, DEPENDENCY_MIN_LIFT = 5, 0.10, 5.0
RULE_COLUMNS = list(ui.RED_FLAG_LABELS)


def recompute_rules(row: dict, base: pd.Series, weekend: bool) -> dict:
    """Hitung ulang 5 rule dari input. Rule 5 hanya dihitung ulang kalau input relasinya diubah, karena notebook memakai
    jumlah transaksi pasangan 1 tahun (di sini didekati dengan riwayat pasangan)."""
    row["rf_near_limit"] = int(NEAR_LIMIT_MIN_RATIO <= row["limit_ratio"] <= NEAR_LIMIT_MAX_RATIO)
    row["rf_after_hours"] = int(row["approve_hour"] >= AFTER_HOURS_START or row["approve_hour"] < AFTER_HOURS_END or weekend)
    row["rf_fast_approval"] = int(row["approval_minutes"] < FAST_APPROVAL_MINUTES)
    row["rf_debtor_repeat"] = int(row["debtor_loans_1y"] >= DEBTOR_REPEAT_MIN_PRIOR)
    relation_changed = any(not np.isclose(float(row[c]), value_of(base, c))
                           for c in ("requester_approver_ratio", "pair_dependency_lift", "pair_prior_n"))
    if relation_changed:
        row["rf_pair_dependency"] = int(row["pair_prior_n"] >= DEPENDENCY_MIN_PAIR_TX
                                        and (row["requester_approver_ratio"] or 0) >= DEPENDENCY_MIN_RATIO
                                        and (row["pair_dependency_lift"] or 0) >= DEPENDENCY_MIN_LIFT)
    else:
        row["rf_pair_dependency"] = int(base.get("rf_pair_dependency", 0) or 0)
    row["rule_score"] = sum(int(row[c]) for c in RULE_COLUMNS)
    return row


def case_presets(test: pd.DataFrame) -> dict:
    """Label → request_id: contoh nyata dari test 2026 untuk memulai simulasi."""
    presets = {}
    collusion = test[test["fraud_label"] == 1].sort_values("risk_probability", ascending=False)
    if len(collusion):
        presets["🔴 Kolusi terbukti · skor tertinggi"] = collusion.iloc[0]["request_id"]
        presets["🟠 Kolusi terbukti · skor di perbatasan"] = collusion.iloc[len(collusion) * 3 // 4]["request_id"]
    hard_negative = test[test["pattern"].astype(str).str.startswith("Normal -")].sort_values("risk_probability", ascending=False)
    if len(hard_negative):
        presets["🟢 Normal yang mirip kolusi (hard negative)"] = hard_negative.iloc[0]["request_id"]
    normal = test[(test["fraud_label"] == 0) & (test["rule_score"] == 0)].sort_values("risk_probability")
    if len(normal):
        presets["⚪ Normal biasa"] = normal.iloc[len(normal) // 2]["request_id"]
    return presets


def value_of(base: pd.Series, column: str, default: float = 0.0) -> float:
    """Nilai numerik kasus asal (NaN → default) untuk mengisi widget."""
    value = pd.to_numeric(base.get(column), errors="coerce")
    return default if pd.isna(value) else float(value)


def options_of(test: pd.DataFrame, column: str, current) -> tuple[list, int]:
    values = sorted(test[column].dropna().astype(str).unique().tolist())
    current = str(current)
    if current not in values:
        values = [current] + values
    return values, values.index(current)


def score_gauge(score: float, settings: dict) -> go.Figure:
    high, watch = settings.get("final_threshold", 0.5) * 100, settings.get("watch_threshold", 0.25) * 100
    color = ui.ORANGE if score >= high else (ui.YELLOW if score >= watch else ui.GREEN)
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 34, "color": ui.INK}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.28},
               "steps": [{"range": [0, watch], "color": "#e7f6ef"}, {"range": [watch, high], "color": "#fef3d6"},
                         {"range": [high, 100], "color": "#fde8e8"}],
               "threshold": {"line": {"color": ui.ORANGE, "width": 4}, "thickness": 0.8, "value": high}}))
    figure.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=10))
    return figure


def _render_single(model, settings: dict, test: pd.DataFrame) -> None:
    presets = case_presets(test)
    left, right = st.columns([1.3, 1])
    preset = left.selectbox("Mulai dari contoh kasus nyata (test 2026)", list(presets) + ["Pilih nomor berkas lain…"], key="lt_preset")
    if preset in presets:
        request_id = presets[preset]
    else:
        request_id = right.selectbox("Nomor berkas", test["request_id"].tolist(), key="lt_request")
    base = test[test["request_id"] == request_id].iloc[0]
    label = "kolusi (label audit)" if base["fraud_label"] == 1 else "normal (label audit)"
    st.caption(f"Berkas **{request_id}** · pola audit: `{base.get('pattern', '-')}` · {label}. Ubah input di bawah untuk simulasi "
               "*what-if*; rules dihitung ulang otomatis, lalu Risk Engine menilai ulang.")

    key = f"lt_{request_id}"
    flagged_daytime = (value_of(base, "rf_after_hours") == 1
                       and AFTER_HOURS_END <= value_of(base, "approve_hour", 10) < AFTER_HOURS_START)
    col_content, col_relation, col_iforest = st.columns([1.15, 1.15, 0.8])
    with col_content:
        st.markdown(f"**{ui.PILLAR_LABELS['classifier']}**")
        limit = max(value_of(base, "approval_limit", 1.0), 1.0)
        amount = st.number_input("Nominal (Rp)", 0, 500_000_000, int(value_of(base, "amount")), 1_000_000, key=f"{key}_amount")
        st.caption(f"Limit approver {ui.rupiah(limit, short=False)} → rasio **{amount / limit:.1%}**")
        approve_hour = st.slider("Jam approval", 0, 23, int(value_of(base, "approve_hour", 10)), key=f"{key}_hour")
        weekend = st.checkbox("Approval akhir pekan", value=flagged_daytime, key=f"{key}_weekend")
        approval_minutes = st.number_input("Menit dari pengajuan ke approval", 0.0, max(20_000.0, value_of(base, "approval_minutes")),
                                           value_of(base, "approval_minutes", 60.0), 5.0,
                                           key=f"{key}_minutes")
        debtor_loans_1y = st.number_input("Pengajuan lain ke debitur sama (1 thn)", 0, max(30, int(value_of(base, "debtor_loans_1y"))),
                                          int(value_of(base, "debtor_loans_1y")),
                                          key=f"{key}_dl1y")
        debtor_loans_30d = st.number_input("… dalam 30 hari", 0, max(30, int(value_of(base, "debtor_loans_30d"))),
                                           int(value_of(base, "debtor_loans_30d")), key=f"{key}_dl30")
        products, product_index = options_of(test, "loan_product", base["loan_product"])
        loan_product = st.selectbox("Produk", products, index=product_index, key=f"{key}_product")
    with col_relation:
        st.markdown(f"**{ui.PILLAR_LABELS['graph']}**")
        pair_prior_n = st.number_input("Riwayat transaksi pasangan requester–approver", 0,
                                       max(5000, int(value_of(base, "pair_prior_n"))), int(value_of(base, "pair_prior_n")),
                                       key=f"{key}_pair")
        ratio_base = 0.0 if pd.isna(base["requester_approver_ratio"]) else float(base["requester_approver_ratio"])
        requester_approver_ratio = st.slider("% pengajuan requester ke approver ini (1 thn)", 0.0, 1.0, ratio_base, 0.01,
                                             key=f"{key}_ratio")
        lift_base = 0.0 if pd.isna(base["pair_dependency_lift"]) else float(base["pair_dependency_lift"])
        pair_dependency_lift = st.slider("Kelipatan vs kebiasaan rekan unit (lift)", 0.0, 30.0, min(lift_base, 30.0), 0.5,
                                         key=f"{key}_lift")
        same_comm = st.checkbox("Requester & approver satu komunitas risiko", value=bool(value_of(base, "g_same_comm")),
                                key=f"{key}_samecomm")
        comm_score = st.number_input("Skor komunitas risiko requester (mentah)", 0.0, max(1000.0, value_of(base, "g_risk_comm_score")),
                                     value_of(base, "g_risk_comm_score"), 5.0,
                                     key=f"{key}_comm")
        pair_weight = st.number_input("Bobot hubungan di graph risiko", 0.0, max(200.0, value_of(base, "g_risk_pair_weight")),
                                      value_of(base, "g_risk_pair_weight"), 1.0,
                                      key=f"{key}_pairw")
        shared = st.number_input("Debitur yang dipakai bersama RM lain", 0, max(100, int(value_of(base, "g_req_shared_debtors"))),
                                 int(value_of(base, "g_req_shared_debtors")), key=f"{key}_shared")
    with col_iforest:
        st.markdown(f"**{ui.PILLAR_LABELS['iforest']}**")
        if_base = base.get("approver_anomaly_score")
        if_missing = st.checkbox("Approver belum punya skor (< 10 keputusan)", value=pd.isna(if_base), key=f"{key}_ifmiss")
        if_score = st.slider("Persentil anomali approver", 0.0, 100.0, 50.0 if pd.isna(if_base) else float(if_base), 1.0,
                             disabled=if_missing, key=f"{key}_if")
        st.markdown(f"**{ui.PILLAR_LABELS['rule']}**")
        st.caption("Dihitung ulang otomatis dari input di kiri.")

    edited = base.to_dict()
    edited.update(amount=float(amount), limit_ratio=amount / limit, approve_hour=approve_hour, approval_minutes=approval_minutes,
                  debtor_loans_1y=debtor_loans_1y, debtor_loans_30d=debtor_loans_30d, loan_product=loan_product,
                  pair_prior_n=pair_prior_n, requester_approver_ratio=requester_approver_ratio,
                  pair_dependency_lift=pair_dependency_lift, g_same_comm=int(same_comm), g_risk_comm_score=comm_score,
                  g_risk_pair_weight=pair_weight, g_req_shared_debtors=shared,
                  approver_anomaly_score=np.nan if if_missing else if_score)
    edited = recompute_rules(edited, base, weekend)
    with col_iforest:
        active = [ui.RED_FLAG_LABELS[c] for c in RULE_COLUMNS if edited[c] == 1]
        st.markdown(ui.chips(active) if active else "✅ tidak ada rule terpenuhi", unsafe_allow_html=True)

    frames = pd.DataFrame([base.to_dict(), edited])
    scored = model.score_frame(frames)
    original, current = scored.iloc[0], scored.iloc[1]
    st.divider()
    left, right = st.columns([0.9, 1.4])
    with left:
        st.plotly_chart(score_gauge(current["risk_score"], settings), use_container_width=True)
        delta = current["risk_score"] - original["risk_score"]
        ui.kpi_row([("Skor akhir", f"{current['risk_score']:.0f}/100", f"asli {original['risk_score']:.0f} ({delta:+.0f})",
                     ui.BAND_COLORS.get(current["risk_band"], ui.BLUE)),
                    ("Band", current["risk_band"], f"asli: {original['risk_band']}", ui.BAND_COLORS.get(current["risk_band"], ui.BLUE))])
        st.caption(pillars.band_rules_text(settings))
        if current["risk_band"] == "Tinggi":
            st.error("Rekomendasi: tahan pencairan sementara dan periksa (SLA 1×24 jam). Skor adalah indikasi, bukan bukti.")
        elif current["risk_band"] == "Sedang":
            st.warning("Rekomendasi: verifikasi berkas fisik & riwayat pasangan sebelum pencairan.")
        else:
            st.success("Rekomendasi: proses sesuai alur standar.")
    with right:
        pillars.pillar_cards(current)
        st.plotly_chart(pillars.contribution_waterfall(current, height=290), use_container_width=True)

    comparison = pd.DataFrame({"pilar": [ui.PILLAR_SHORT[p] for p in ui.PILLARS] * 2,
                               "versi": ["asli"] * 4 + ["setelah diubah"] * 4,
                               "kontribusi": [original[f"c_{p}"] for p in ui.PILLARS] + [current[f"c_{p}"] for p in ui.PILLARS]})
    figure = px.bar(comparison, x="kontribusi", y="pilar", color="versi", barmode="group", orientation="h",
                    color_discrete_map={"asli": ui.MUTED, "setelah diubah": ui.ORANGE})
    figure.add_vline(x=0, line_color=ui.MUTED)
    st.plotly_chart(ui.style_figure(figure, "Pilar mana yang berubah karena simulasi Anda?",
                                    "kontribusi ke skor akhir (log-odds), sebelum vs sesudah input diubah", height=300),
                    use_container_width=True)
    st.caption("Contoh eksperimen: pada kasus kolusi, pindahkan approval ke jam kantor dan naikkan jeda approval (pelaku menghindari "
               "rules). Skor Pilar 1 & 3 turun, tetapi Pilar 4 (relasi) tetap tinggi — inilah alasan arsitektur berlapis.")


def _render_batch(model, settings: dict, test: pd.DataFrame) -> None:
    st.markdown("Nilai banyak pengajuan sekaligus. Gunakan sampel test 2026 atau unggah CSV dengan kolom input yang sama.")
    left, right = st.columns([1.5, 1])
    uploaded = left.file_uploader("Unggah CSV pengajuan", type=["csv"], key="lt_upload")
    sample_size = right.slider("…atau ambil sampel test 2026", 50, min(2000, len(test)), 300, 50, key="lt_sample")
    frame = pd.read_csv(uploaded) if uploaded is not None else test.sample(sample_size, random_state=7)
    missing = [c for c in model.features if c not in frame.columns and c != "approver_anomaly_score"]
    if missing:
        st.error("Kolom input belum lengkap: " + ", ".join(missing))
        st.download_button("Unduh template kolom", pd.DataFrame(columns=model.features).to_csv(index=False),
                           "template_risk_engine.csv", "text/csv")
        return
    scored = frame.drop(columns=[c for c in ["risk_probability", "risk_score", "risk_band", "escalation", "s_rule", "s_if",
                                             "s_cls", "s_graph", "c_rule", "c_iforest", "c_classifier", "c_graph", "c_base"]
                                 if c in frame.columns]).join(model.score_frame(frame))
    counts = scored["risk_band"].value_counts()
    ui.kpi_row([("Dinilai", ui.number(len(scored)), "pengajuan", ui.BLUE),
                ("Tinggi", ui.number(counts.get("Tinggi", 0)), ui.percent(counts.get("Tinggi", 0) / len(scored), 1), ui.ORANGE),
                ("Sedang", ui.number(counts.get("Sedang", 0)), ui.percent(counts.get("Sedang", 0) / len(scored), 1), ui.YELLOW),
                ("Jaring pengaman", ui.number((scored["escalation"] != "").sum()), "pilar sangat yakin sendirian", ui.PURPLE)])
    left, right = st.columns(2)
    figure = px.histogram(scored, x="risk_score", color="risk_band", nbins=40, color_discrete_map=ui.BAND_COLORS)
    figure.add_vline(x=settings.get("final_threshold", 0.5) * 100, line_dash="dash", line_color=ui.ORANGE, annotation_text="Tinggi")
    figure.add_vline(x=settings.get("watch_threshold", 0.25) * 100, line_dash="dot", line_color=ui.YELLOW, annotation_text="Sedang")
    left.plotly_chart(ui.style_figure(figure, "Distribusi skor akhir", height=320), use_container_width=True)
    high = scored[scored["risk_band"] != "Rendah"]
    if len(high):
        top_pillar = high[[f"c_{p}" for p in ui.PILLARS]].idxmax(axis=1).str[2:].map(ui.PILLAR_SHORT).value_counts()
        figure = px.bar(x=top_pillar.values, y=top_pillar.index, orientation="h", color_discrete_sequence=[ui.ORANGE],
                        labels={"x": "alert", "y": ""})
        right.plotly_chart(ui.style_figure(figure, "Pilar penyebab utama alert", "kontribusi terbesar per alert", height=320),
                           use_container_width=True)
    columns = [c for c in ["request_id", "fraud_label", "pattern", "risk_score", "risk_band", "s_rule", "s_if", "s_cls", "s_graph",
                           "escalation", "amount"] if c in scored.columns]
    st.dataframe(scored.sort_values("risk_score", ascending=False)[columns], hide_index=True, use_container_width=True, height=360,
                 column_config={"risk_score": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f")})
    st.download_button("Unduh hasil (CSV)", scored.to_csv(index=False), "hasil_risk_engine.csv", "text/csv")


def render() -> None:
    ui.page_header("Cek Berkas Baru (Live Test)",
                   "Risk Engine 4 pilar menilai pengajuan secara langsung — ubah input per pilar dan lihat skornya bergerak")
    settings = services.app_settings()
    model = services.get_risk_model()
    test = services.table("test_features")
    if model is None or not hasattr(model, "score_frame") or not hasattr(model, "pillar_scores"):
        st.warning("Risk Engine tidak bisa dimuat di lingkungan ini (versi scikit-learn harus sama dengan saat notebook dijalankan). "
                   f"Detail: {st.session_state.get('model_load_error', 'model lama tanpa 4 pilar')}")
        return
    if test.empty:
        ui.empty_state("Tabel test_features belum ada. Jalankan notebook terlebih dahulu.")
        return
    st.markdown(pillars.pillar_legend(), unsafe_allow_html=True)
    tab_single, tab_batch = st.tabs(["🎯 Uji satu pengajuan (what-if)", "📂 Batch scoring"])
    with tab_single:
        _render_single(model, settings, test)
    with tab_batch:
        _render_batch(model, settings, test)
