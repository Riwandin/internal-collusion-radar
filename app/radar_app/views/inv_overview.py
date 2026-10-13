"""Investigator · Overview: monitoring alert — pencarian & filter → scorecard → grafik → antrean → buka Detail Kasus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import RED_FLAG_COLUMNS, flags_text, latest_decisions


def search_mask(frame: pd.DataFrame, query: str) -> pd.Series:
    """Cocokkan ID transaksi, atau nama / ID pegawai (requester maupun approver)."""
    query = query.strip()
    employees = services.table("employees")
    matched = employees[employees["employee_name"].str.contains(query, case=False, na=False, regex=False)
                        | employees["employee_id"].str.upper().eq(query.upper())]["employee_id"]
    return (frame["request_id"].str.upper().str.contains(query.upper(), regex=False)
            | frame["requester_id"].isin(matched) | frame["approver_id"].isin(matched))


def filter_frame(transactions: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (semua transaksi sesuai periode/wilayah/pencarian, alert sesuai seluruh filter)."""
    regions_map = services.unit_region()
    query = st.text_input("🔎 Cari alert", key="ov_search",
                          placeholder="ID transaksi, nama, atau ID pegawai (mis. R0114175, Gita, E00747)")
    c_year, c_month, c_level, c_region, c_unit, c_status = st.columns([0.8, 1.5, 1.2, 1.2, 1.4, 1.1])
    base = services.period_filter(transactions, "ov", default_year=2026, columns=(c_year, c_month))
    levels = c_level.multiselect("Level risiko", ["Tinggi", "Sedang"], default=["Tinggi", "Sedang"], key="ov_levels")
    regions = c_region.multiselect("Region", sorted(set(regions_map.values())), key="ov_regions")
    unit_options = sorted(u for u in base["requester_unit"].dropna().unique() if not regions or regions_map.get(u) in regions)
    units = c_unit.multiselect("Unit kerja", unit_options, key="ov_units")
    status = c_status.selectbox("Status keputusan", ["semua", "belum diputus", "sudah diputus"], key="ov_status")
    flag_labels = st.multiselect("Red flag (minimal satu terpenuhi)", list(ui.RED_FLAG_LABELS.values()), key="ov_flags")
    if regions:
        base = base[base["requester_unit"].map(regions_map).isin(regions)]
    if units:
        base = base[base["requester_unit"].isin(units)]
    if query.strip():
        base = base[search_mask(base, query)]
    alerts = base[base["risk_band"].isin(levels or ["Tinggi", "Sedang"])]
    if flag_labels:
        columns = [c for c, label in ui.RED_FLAG_LABELS.items() if label in flag_labels]
        alerts = alerts[alerts[columns].eq(1).any(axis=1)]
    if status == "belum diputus":
        alerts = alerts[~alerts["request_id"].isin(decisions.index)]
    elif status == "sudah diputus":
        alerts = alerts[alerts["request_id"].isin(decisions.index)]
    return base, alerts


def render() -> None:
    ui.page_header("Overview Alert", "Alur: saring alert → lihat besaran risikonya → pilih alert di antrean → buka Detail Kasus")
    transactions = services.table("transactions_scored")
    if transactions.empty:
        ui.empty_state("Data belum tersedia. Jalankan notebook lalu muat ke PostgreSQL (lihat README).")
        return
    settings = services.app_settings()
    decisions = latest_decisions()
    base, alerts = filter_frame(transactions, decisions)
    high = alerts[alerts["risk_band"] == "Tinggi"]
    open_alerts = alerts[~alerts["request_id"].isin(decisions.index)]
    lgd = settings.get("lgd", 0.6)
    expected_loss = (alerts["amount"] * alerts["risk_probability"]).sum() * lgd
    ui.kpi_row([
        ("Alert muncul", ui.number(len(alerts)), f"{ui.percent(len(alerts) / max(len(base), 1), 1)} dari {ui.number(len(base))} transaksi", ui.BLUE),
        ("Alert Tinggi", ui.number(len(high)), f"Sedang: {ui.number((alerts['risk_band'] == 'Sedang').sum())}", ui.ORANGE),
        ("Belum diputus", ui.number(len(open_alerts)), f"Tinggi belum diputus: {ui.number((open_alerts['risk_band'] == 'Tinggi').sum())}", ui.RED),
        ("Nominal kredit pada alert", ui.rupiah(alerts["amount"].sum()), f"Tinggi: {ui.rupiah(high['amount'].sum())}", ui.YELLOW),
        ("Estimasi potensi kerugian", ui.rupiah(expected_loss), f"nominal × probabilitas × LGD {ui.percent(lgd)}", ui.PURPLE),
        ("Approver terlibat", ui.number(alerts["approver_id"].nunique()), st.session_state.get("ov_period_label", ""), ui.GREEN)])
    ui.explain({
        "Alert muncul": "transaksi berlevel Tinggi/Sedang sesuai filter. Tinggi = skor ≥ ambang final model; Sedang = zona pantau.",
        "Belum diputus": "alert yang belum punya keputusan investigator (lihat Detail Kasus → Keputusan).",
        "Nominal kredit pada alert": "total plafon kredit yang disetujui pada transaksi alert (eksposur, belum tentu rugi).",
        "Estimasi potensi kerugian": "Σ (nominal × probabilitas kolusi dari model) × LGD. Indikasi besaran, bukan kerugian riil.",
        "Grafik red flag": "batang oranye jauh lebih panjang dari abu-abu → red flag itu khas pada alert, bukan kejadian umum."})
    if alerts.empty:
        st.info("Tidak ada alert yang cocok dengan filter/pencarian.")
        return
    left, right = st.columns([1.4, 1])
    monthly = (alerts.groupby([alerts["event_time"].dt.to_period("M").astype(str).rename("bulan"), "risk_band"])
               .size().rename("jumlah").reset_index())
    figure = px.bar(monthly, x="bulan", y="jumlah", color="risk_band", color_discrete_map=ui.BAND_COLORS,
                    category_orders={"risk_band": ["Tinggi", "Sedang"]}, labels={"risk_band": "level"})
    left.plotly_chart(ui.style_figure(figure, "Tren alert per bulan", "jumlah alert Tinggi & Sedang"), use_container_width=True)
    status = alerts["request_id"].map(decisions["decision"]).fillna("Belum diputus").value_counts()
    figure = px.pie(status.rename_axis("status").reset_index(name="jumlah"), names="status", values="jumlah", hole=.55,
                    color="status", color_discrete_map={"Belum diputus": ui.RED, "Tindak lanjut / eskalasi": ui.ORANGE,
                                                        "Bukan kolusi (alarm palsu)": ui.GREEN, "Perlu data tambahan": ui.YELLOW})
    right.plotly_chart(ui.style_figure(figure, "Status keputusan alert", "progres penanganan"), use_container_width=True)
    left, right = st.columns(2)
    flags = pd.DataFrame({"rule": [ui.RED_FLAG_LABELS[c] for c in RED_FLAG_COLUMNS],
                          "semua transaksi": [base[c].mean() * 100 for c in RED_FLAG_COLUMNS],
                          "alert": [alerts[c].mean() * 100 for c in RED_FLAG_COLUMNS]})
    figure = go.Figure()
    figure.add_bar(y=flags["rule"], x=flags["semua transaksi"], orientation="h", name="semua transaksi", marker_color=ui.MUTED)
    figure.add_bar(y=flags["rule"], x=flags["alert"], orientation="h", name="alert", marker_color=ui.ORANGE)
    figure.update_layout(barmode="group", xaxis_title="% transaksi yang memenuhi rule")
    left.plotly_chart(ui.style_figure(figure, "Red flag pada alert", "dibanding seluruh transaksi periode ini"), use_container_width=True)
    by_unit = (alerts.groupby("requester_unit").agg(alert=("request_id", "size"), nominal=("amount", "sum"))
               .sort_values("alert").tail(10).reset_index())
    figure = px.bar(by_unit, x="alert", y="requester_unit", orientation="h", color_discrete_sequence=[ui.ORANGE],
                    hover_data={"nominal": ":,.0f"}, labels={"requester_unit": "", "alert": "jumlah alert"})
    right.plotly_chart(ui.style_figure(figure, "10 unit dengan alert terbanyak", "unit requester"), use_container_width=True)

    st.markdown("#### Antrean alert")
    names = services.name_map()
    table = alerts.sort_values(["risk_score", "event_time"], ascending=False).head(500).copy()
    table["requester"] = table["requester_id"].map(names)
    table["approver"] = table["approver_id"].map(names)
    table["red flag"] = table.apply(flags_text, axis=1)
    table["status"] = table["request_id"].map(decisions["decision"]).fillna("belum diputus")
    table["nominal"] = table["amount"].map(ui.rupiah)
    table["alasan utama"] = table["top_reasons"].fillna("").str.split(" | ").str[0]
    columns = ["request_id", "event_time", "requester", "approver", "requester_unit", "nominal", "risk_score", "risk_band",
               "red flag", "status", "alasan utama"]
    st.caption(f"Menampilkan {len(table):,} dari {len(alerts):,} alert (urut skor tertinggi). Pilih satu baris lalu buka Detail Kasus.")
    selection = st.dataframe(
        table[columns], hide_index=True, use_container_width=True, height=380, on_select="rerun",
        selection_mode="single-row", key="ov_table",
        column_config={"risk_score": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f"),
                       "event_time": st.column_config.DatetimeColumn("waktu", format="DD MMM YYYY HH:mm"),
                       "requester_unit": "unit", "risk_band": "level"})
    rows = getattr(getattr(selection, "selection", None), "rows", []) or []
    open_column, download_column = st.columns(2)
    if rows:
        request_id = table.iloc[rows[0]]["request_id"]
        if open_column.button(f"Buka Detail Kasus {request_id} →", type="primary", use_container_width=True):
            st.session_state["case_id"] = request_id
            st.switch_page(services.PAGES["detail_kasus"])
    else:
        open_column.info("Pilih satu baris di tabel untuk membuka Detail Kasus.")
    download_column.download_button("Unduh antrean (CSV)", table[columns].assign(amount=table["amount"]).to_csv(index=False),
                                    "alert_terfilter.csv", "text/csv", use_container_width=True)
