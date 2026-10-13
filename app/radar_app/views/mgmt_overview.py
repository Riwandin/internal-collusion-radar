"""Manajemen · Command Center Eksekutif & Peta Risiko Wilayah/Cabang."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import latest_decisions, red_flag_summary


def render() -> None:
    ui.page_header(
        "Command Center Eksekutif & Peta Cabang",
        "Pantau eksposur rupiah berisiko, efektivitas model dibanding rules, konsentrasi risiko per cabang, dan progres tim investigator.",
    )
    settings = services.app_settings()
    transactions = services.table("transactions_scored")
    unit_risk = services.table("unit_risk")
    year = st.sidebar.selectbox(
        "Tahun Evaluasi",
        sorted(transactions["event_time"].dt.year.unique().astype(int), reverse=True),
        key="exec_year",
    )
    frame = transactions[transactions["event_time"].dt.year == year]
    months = max(frame["event_time"].dt.to_period("M").nunique(), 1)
    high = frame[frame["risk_band"] == "Tinggi"]
    reviewed = high[high["reviewed"].astype(str).str.lower().eq("true")]
    true_reviewed = int((reviewed["label_audit"] == 1).sum())
    reviewed_precision = true_reviewed / max(len(reviewed), 1)
    lower = true_reviewed / max(len(high), 1)
    upper = (true_reviewed + reviewed_precision * (len(high) - len(reviewed))) / max(len(high), 1)

    impact = services.table("impact")
    model_impact = impact[impact["skenario"].str.startswith("Model")].iloc[0] if len(impact) else None
    decisions = latest_decisions()
    backlog = int((~high["request_id"].isin(decisions.index)).sum())

    ui.kpi_row([
        ("Transaksi Dinilai", ui.number(len(frame)), f"{year} · {months} bln · {ui.rupiah(frame['amount'].sum())}", ui.BLUE),
        ("Alert Tinggi / Bulan", ui.number(len(high) / months, 0), f"Rules: ±{ui.number((frame['rule_score'] >= settings.get('rules_threshold', 1)).sum() / months)}/bln", ui.ORANGE),
        ("Eksposur Berisiko", ui.rupiah(high["amount"].sum()), "Total nominal pada alert Tinggi", ui.RED),
        ("Ketepatan (Precision)", f"{ui.percent(lower)}–{ui.percent(upper)}", "Rentang pada populasi", ui.GREEN),
        ("Manfaat Bersih / Thn", ui.rupiah(model_impact["manfaat bersih/tahun"]) if model_impact is not None else "-", "Estimasi penghematan", ui.PURPLE),
        ("Backlog Investigasi", ui.number(backlog), "Alert Tinggi belum diputus", ui.YELLOW),
    ])

    st.write("")

    tab_exec, tab_branch = st.tabs([
        "📊 Ringkasan Eksekutif & Tren Nasional",
        "🗺️ Peta Risiko Wilayah & Kantor Cabang (Drill-Down)",
    ])

    with tab_exec:
        with st.container(border=True):
            st.markdown("**📌 Ringkasan Masalah Bisnis (Executive Insight):** " + settings.get("problem_statement", "-"))

        kpis = services.table("monthly_kpis")
        left, right = st.columns([1.45, 1])
        figure = go.Figure()
        figure.add_bar(x=kpis["month"], y=kpis["rule_flagged"], name="Ditandai Rules (Banjir Alarm)", marker_color="#d8d7d2")
        figure.add_bar(x=kpis["month"], y=kpis["alert_tinggi"], name="Alert Tinggi (Model ML)", marker_color=ui.ORANGE)
        figure.add_scatter(
            x=kpis["month"],
            y=kpis["kolusi_terkonfirmasi"],
            name="Kolusi Terkonfirmasi Audit",
            mode="lines",
            line=dict(color=ui.INK, width=2),
        )
        figure.add_vrect(x0="2025-01", x1="2025-12", fillcolor=ui.YELLOW, opacity=0.08, line_width=0, annotation_text="validasi")
        figure.add_vrect(x0="2026-01", x1=kpis["month"].max(), fillcolor=ui.GREEN, opacity=0.08, line_width=0, annotation_text="test (2026)")
        figure.update_layout(barmode="overlay")
        left.plotly_chart(
            ui.style_figure(
                figure,
                "Model ML Memangkas Alarm Palsu −81% Dibanding Rules",
                "Perbandingan jumlah alert bulanan vs kasus kolusi nyata (2021–2026)",
            ),
            use_container_width=True,
        )

        region = (
            unit_risk[unit_risk["year"] == year]
            .groupby("region")
            .agg(transaksi=("transaksi", "sum"), alert=("alert_tinggi", "sum"))
            .reset_index()
        )
        region["alert_rate"] = region["alert"] / region["transaksi"]
        figure = px.bar(
            region.sort_values("alert_rate"),
            x="alert_rate",
            y="region",
            orientation="h",
            color="alert_rate",
            color_continuous_scale=["#dbe9fb", ui.ORANGE],
            labels={"alert_rate": "% Transaksi Alert Tinggi", "region": ""},
        )
        figure.update_layout(coloraxis_showscale=False, xaxis_tickformat=".1%")
        right.plotly_chart(
            ui.style_figure(figure, "Wilayah dengan Konsentrasi Alert Tertinggi", f"Tahun {year}"),
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)
        communities = services.table("communities").head(6)
        with c1:
            st.markdown("**🕸️ 6 Komunitas Risiko Terbesar**")
            st.dataframe(
                communities[["community_id", "skor_komunitas", "anggota", "unit", "aktif_bulan"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "community_id": "ID",
                    "skor_komunitas": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
                    "aktif_bulan": "Aktif (Bln)",
                },
            )
        with c2:
            metrics = services.table("model_metrics")
            test = metrics[metrics["periode"] == "test 2026"].set_index("model")
            st.markdown("**🎯 Perbandingan Akurasi Model (Data Uji 2026)**")
            st.dataframe(
                test[["precision", "recall", "PR-AUC", "FP"]].style.format(
                    {"precision": "{:.0%}", "recall": "{:.0%}", "PR-AUC": "{:.3f}"}
                ),
                use_container_width=True,
            )
        with c3:
            st.markdown("**💰 Ringkasan Dampak Finansial Tahunan**")
            if len(impact):
                display = impact.set_index("skenario")[
                    ["kerugian dicegah/tahun", "biaya pemeriksaan/tahun", "manfaat bersih/tahun"]
                ].map(ui.rupiah)
                st.dataframe(display, use_container_width=True)

    with tab_branch:
        u_frame = unit_risk[unit_risk["year"] == year].copy()
        u_frame["region"] = u_frame["region"].fillna("Lainnya")
        left, right = st.columns([1.3, 1])
        figure = px.treemap(
            u_frame,
            path=["region", "requester_unit"],
            values="transaksi",
            color="alert_rate",
            color_continuous_scale=["#eef4fc", "#f3c35a", ui.ORANGE],
            hover_data={"alert_tinggi": True, "skor_rata": ":.1f"},
        )
        figure.update_layout(coloraxis_colorbar=dict(title="% Alert", tickformat=".0%"))
        left.plotly_chart(
            ui.style_figure(
                figure,
                "Peta Sebaran Risiko: Wilayah → Kantor Cabang",
                "Ukuran kotak = volume pengajuan kredit · Warna = % alert risiko Tinggi",
                height=480,
            ),
            use_container_width=True,
        )

        top = u_frame.sort_values("alert_rate", ascending=False).head(15)
        figure = px.bar(
            top.iloc[::-1],
            x="alert_rate",
            y="requester_unit",
            orientation="h",
            color_discrete_sequence=[ui.ORANGE],
            hover_data=["transaksi", "alert_tinggi", "region"],
            labels={"alert_rate": "% Alert Tinggi", "requester_unit": ""},
        )
        figure.update_layout(xaxis_tickformat=".1%")
        right.plotly_chart(
            ui.style_figure(figure, "15 Kantor Cabang dengan % Alert Tertinggi", f"Tahun {year}", height=480),
            use_container_width=True,
        )

        st.divider()
        unit = st.selectbox(
            "🔎 Pilih Kantor Cabang untuk Bedah Detail Unit:",
            top["requester_unit"].tolist() + sorted(set(u_frame["requester_unit"]) - set(top["requester_unit"])),
            key="mgmt_unit_pick",
        )
        unit_tx = transactions[(transactions["requester_unit"] == unit) & (transactions["event_time"].dt.year == year)]
        ui.kpi_row([
            ("Total Transaksi Unit", ui.number(len(unit_tx)), unit, ui.BLUE),
            ("Alert Tinggi", ui.number((unit_tx["risk_band"] == "Tinggi").sum()), ui.percent((unit_tx["risk_band"] == "Tinggi").mean(), 1), ui.ORANGE),
            ("Nominal Berisiko", ui.rupiah(unit_tx.loc[unit_tx["risk_band"] == "Tinggi", "amount"].sum()), "Pada cabang ini", ui.RED),
            ("Pejabat Pemutus Terlibat", ui.number(unit_tx.loc[unit_tx["risk_band"] == "Tinggi", "approver_id"].nunique()), "Approver unik", ui.PURPLE),
        ])
        l_col, r_col = st.columns(2)
        monthly = (
            unit_tx.groupby(unit_tx["event_time"].dt.to_period("M").astype(str))["risk_band"]
            .apply(lambda s: (s == "Tinggi").mean())
            .reset_index()
        )
        fig_m = px.line(monthly, x="event_time", y="risk_band", markers=True, labels={"event_time": "Bulan", "risk_band": "% Alert Tinggi"})
        fig_m.update_layout(yaxis_tickformat=".0%")
        l_col.plotly_chart(ui.style_figure(fig_m, f"Tren % Alert Tinggi · Cabang {unit}", height=320), use_container_width=True)

        flags = red_flag_summary(unit_tx)
        fig_f = px.bar(flags, x="% alert Tinggi", y="rule", orientation="h", color_discrete_sequence=[ui.ORANGE])
        fig_f.update_layout(xaxis_tickformat=".0%")
        r_col.plotly_chart(
            ui.style_figure(fig_f, f"Modus Pelanggaran Dominan di Cabang {unit}", "Membantu menentukan fokus audit cabang", height=320),
            use_container_width=True,
        )
