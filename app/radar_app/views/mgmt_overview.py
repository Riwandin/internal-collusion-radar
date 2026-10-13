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
    regions_map = services.unit_region()
    year_column, region_column = st.columns([1, 2])
    year = year_column.selectbox("Tahun Evaluasi", sorted(transactions["event_time"].dt.year.unique().astype(int), reverse=True), key="exec_year")
    regions = region_column.multiselect("Filter Wilayah / Region", sorted(set(regions_map.values())), key="exec_regions", placeholder="Semua wilayah")
    frame = transactions[transactions["event_time"].dt.year == year]
    if regions:
        frame = frame[frame["requester_unit"].map(regions_map).isin(regions)]
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
        ("Akurasi Alert Terbukti", f"{ui.percent(lower)}–{ui.percent(upper)}", "Rentang hasil audit fisik", ui.GREEN),
        ("Manfaat Bersih / Thn", ui.rupiah(model_impact["manfaat bersih/tahun"]) if model_impact is not None else "-", "Estimasi penghematan", ui.PURPLE),
        ("Backlog Investigasi", ui.number(backlog), "Alert Tinggi belum diputus", ui.YELLOW),
    ])
    ui.explain({
        "Transaksi Dinilai": f"Seluruh transaksi pengajuan kredit yang dievaluasi sistem pada tahun {year}.",
        "Alert Tinggi / Bulan": "Rata-rata alert prioritas Tinggi per bulan dibanding ekspektasi rules konvensional.",
        "Eksposur Berisiko": "Total nilai nominal kredit yang terdeteksi berisiko Tinggi.",
        "Akurasi Alert Terbukti": "Perkiraan proporsi alert yang terbukti kolusi saat diaudit secara langsung.",
        "Manfaat Bersih / Thn": "Penghematan kerugian kredit fraud setelah dikurangi biaya investigasi & sistem.",
        "Backlog Investigasi": "Jumlah alert kategori Tinggi yang masih menunggu disposisi/keputusan investigator.",
    })

    st.write("")

    tab_exec, tab_branch = st.tabs([
        "📊 Ringkasan Eksekutif & Tren Nasional",
        "🗺️ Peta Risiko Wilayah & Kantor Cabang (Drill-Down)",
    ])

    with tab_exec:
        with st.container(border=True):
            st.markdown("**📌 Ringkasan Masalah Bisnis (Executive Insight):** " + settings.get("problem_statement", "-"))

        kpis = services.table("monthly_kpis").copy()
        kpis["year"] = kpis["month"].str[:4].astype(int)
        latest_month = kpis["month"].max()
        latest_year = int(latest_month[:4])
        latest_month_dt = pd.to_datetime(latest_month + "-01")
        latest_month_label = latest_month_dt.strftime("%b %Y")

        yearly = (
            kpis.groupby("year")
            .agg(
                total_transaksi=("transaksi", "sum"),
                kasus_kolusi=("kolusi_terkonfirmasi", "sum"),
                bulan=("month", "count"),
            )
            .reset_index()
        )
        yearly["rate_per_1000"] = (yearly["kasus_kolusi"] / yearly["total_transaksi"]) * 1000

        def get_year_label(row: pd.Series) -> str:
            y = int(row["year"])
            if y == latest_year and row["bulan"] < 12:
                return f"{y}* (s.d. {latest_month_label})"
            return str(y)

        yearly["label"] = yearly.apply(get_year_label, axis=1)

        left, right = st.columns([1.45, 1])

        with left:
            t_hdr_col, t_sel_col = st.columns([1.3, 1.2])
            t_hdr_col.markdown("##### 📈 Tren Kasus Kolusi & Rasio per 1.000 Kredit")
            trend_view = t_sel_col.radio(
                "Pilihan Tampilan Tren",
                ["Semua Tahun (2021–2026)", f"Detail Bulanan ({year})"],
                horizontal=True,
                key="mgmt_trend_view",
                label_visibility="collapsed",
            )

            if trend_view == "Semua Tahun (2021–2026)":
                yearly = (
                    kpis.groupby("year")
                    .agg(
                        total_transaksi=("transaksi", "sum"),
                        kasus_kolusi=("kolusi_terkonfirmasi", "sum"),
                        bulan=("month", "count"),
                    )
                    .reset_index()
                )
                yearly["rate_per_1000"] = (yearly["kasus_kolusi"] / yearly["total_transaksi"]) * 1000

                def get_year_label(row: pd.Series) -> str:
                    y = int(row["year"])
                    if y == latest_year and row["bulan"] < 12:
                        return f"{y}* (Jan–Sep)"
                    return str(y)

                yearly["label"] = yearly.apply(get_year_label, axis=1)

                fig_trend = go.Figure()
                bar_colors = ["#3b82f6" if int(y) == int(year) else "#1d4ed8" for y in yearly["year"]]
                fig_trend.add_bar(
                    x=yearly["label"],
                    y=yearly["kasus_kolusi"],
                    name="Kasus Kolusi Terbukti (Audit)",
                    marker=dict(color=bar_colors, line=dict(color="#1e3a8a", width=1)),
                    text=[f"<b>{k:,}</b>" for k in yearly["kasus_kolusi"]],
                    textposition="outside",
                    hovertemplate="<b>Tahun %{x}</b><br>Kasus Terbukti: <b>%{y:,} kasus</b><extra></extra>",
                )
                fig_trend.add_scatter(
                    x=yearly["label"],
                    y=yearly["rate_per_1000"],
                    name="Tingkat per 1.000 Pengajuan",
                    mode="lines+markers+text",
                    yaxis="y2",
                    line=dict(color="#ea580c", width=3.5),
                    marker=dict(size=9, color="#ea580c", line=dict(color="white", width=2)),
                    text=[f"<b>{r:.1f}</b>" for r in yearly["rate_per_1000"]],
                    textposition="top center",
                    textfont=dict(color="#c2410c", size=11),
                    hovertemplate="<b>Tahun %{x}</b><br>Tingkat: <b>%{y:.2f} per 1.000 kredit</b><extra></extra>",
                )
                fig_trend.update_layout(
                    yaxis=dict(
                        title="Jumlah Kasus Kolusi (Kasus)",
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.06)",
                        range=[0, max(yearly["kasus_kolusi"]) * 1.25],
                    ),
                    yaxis2=dict(
                        title="Tingkat per 1.000 Pengajuan",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        range=[0, max(yearly["rate_per_1000"]) * 1.35],
                    ),
                    xaxis=dict(title="", tickmode="array", tickvals=yearly["label"]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    margin=dict(t=35, b=20, l=10, r=10),
                    height=360,
                )
                st.plotly_chart(
                    ui.style_figure(
                        fig_trend,
                        "",
                        "Batang: total kasus terbukti · Garis: normalisasi per 1.000 pengajuan agar adil terhadap perubahan volume kredit",
                        height=360,
                    ),
                    use_container_width=True,
                )
                st.caption(
                    f"ℹ️ *Tahun 2026 merupakan data berjalan sementara (YTD s.d. {latest_month_label}, 9 bulan). Garis rasio per 1.000 pengajuan menjaga perbandingan tetap adil."
                )
            else:
                m_year = kpis[kpis["year"] == year].copy()
                m_year["month_num"] = m_year["month"].str[5:7].astype(int)
                months_id = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}
                m_year["label_bulan"] = m_year["month_num"].map(months_id)
                m_year["rate_per_1000"] = (m_year["kolusi_terkonfirmasi"] / m_year["transaksi"].replace(0, np.nan)) * 1000

                fig_month = go.Figure()
                fig_month.add_bar(
                    x=m_year["label_bulan"],
                    y=m_year["kolusi_terkonfirmasi"],
                    name="Kasus Kolusi Bulanan",
                    marker=dict(color="#2563eb", line=dict(color="#1e40af", width=1)),
                    text=[f"<b>{k:,}</b>" for k in m_year["kolusi_terkonfirmasi"]],
                    textposition="outside",
                    hovertemplate="<b>Bulan %{x}</b><br>Kasus Terbukti: <b>%{y:,} kasus</b><extra></extra>",
                )
                fig_month.add_scatter(
                    x=m_year["label_bulan"],
                    y=m_year["rate_per_1000"],
                    name="Tingkat per 1.000 Kredit",
                    mode="lines+markers+text",
                    yaxis="y2",
                    line=dict(color="#ea580c", width=3.5),
                    marker=dict(size=8, color="#ea580c", line=dict(color="white", width=2)),
                    text=[f"<b>{r:.1f}</b>" for r in m_year["rate_per_1000"]],
                    textposition="top center",
                    textfont=dict(color="#c2410c", size=11),
                    hovertemplate="<b>Bulan %{x}</b><br>Tingkat: <b>%{y:.2f} per 1.000 kredit</b><extra></extra>",
                )
                fig_month.update_layout(
                    yaxis=dict(
                        title="Kasus Kolusi (Kasus)",
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.06)",
                        range=[0, max(m_year["kolusi_terkonfirmasi"]) * 1.25],
                    ),
                    yaxis2=dict(
                        title="Tingkat per 1.000 Kredit",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        range=[0, max(m_year["rate_per_1000"]) * 1.35],
                    ),
                    xaxis=dict(title=""),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    margin=dict(t=35, b=20, l=10, r=10),
                    height=360,
                )
                st.plotly_chart(
                    ui.style_figure(
                        fig_month,
                        "",
                        f"Dinamika bulanan tahun {year}: pergerakan kasus terbukti audit dan rasionya per 1.000 transaksi",
                        height=360,
                    ),
                    use_container_width=True,
                )
                st.caption(f"ℹ️ Menampilkan rincian performa bulanan untuk tahun evaluasi terpilih ({year}).")

        u_yr = unit_risk[unit_risk["year"] == year]
        if regions:
            u_yr = u_yr[u_yr["region"].isin(regions)]
            grp_col = "requester_unit"
            chart_title = "Cabang dengan Konsentrasi Alert Tertinggi"
            chart_sub = f"Tahun {year} · Wilayah: {', '.join(regions)}"
        else:
            grp_col = "region"
            chart_title = "Wilayah dengan Konsentrasi Alert Tertinggi"
            chart_sub = f"Tahun {year} · Seluruh Wilayah Nasional"

        reg_summary = (
            u_yr.groupby(grp_col)
            .agg(transaksi=("transaksi", "sum"), alert=("alert_tinggi", "sum"))
            .reset_index()
        )
        reg_summary["alert_rate"] = reg_summary["alert"] / reg_summary["transaksi"].replace(0, np.nan)
        reg_top = reg_summary.dropna(subset=["alert_rate"]).sort_values("alert_rate").tail(12)
        fig_reg = px.bar(
            reg_top,
            x="alert_rate",
            y=grp_col,
            orientation="h",
            color="alert_rate",
            color_continuous_scale=["#dbe9fb", ui.ORANGE],
            labels={"alert_rate": "% Transaksi Alert Tinggi", grp_col: ""},
        )
        fig_reg.update_layout(coloraxis_showscale=False, xaxis_tickformat=".1%", height=420)
        right.plotly_chart(
            ui.style_figure(fig_reg, chart_title, chart_sub, height=420),
            use_container_width=True,
        )

        st.divider()

        c_seg, c_comm = st.columns([1.25, 1])
        with c_seg:
            st.markdown("##### 📦 Konsentrasi Risiko Portofolio Kredit & Sektor Usaha")
            st.markdown(
                f"<div class='small-note' style='margin-bottom: 8px;'>"
                f"Sebaran pengajuan kredit berisiko Tinggi pada tahun {year} berdasarkan skema pinjaman KUR dan sektor debitur."
                f"</div>",
                unsafe_allow_html=True,
            )
            tab_prod, tab_sekt = st.tabs(["🏷️ Produk Kredit KUR", "🏭 Sektor Usaha Debitur"])
            with tab_prod:
                prod_summary = (
                    high.groupby("loan_product")
                    .agg(
                        alert=("request_id", "count"),
                        nominal=("amount", "sum"),
                        plafon_rata=("amount", "mean"),
                    )
                    .reset_index()
                    .sort_values("nominal", ascending=False)
                )
                prod_summary["porsi_nominal"] = (
                    prod_summary["nominal"] / max(prod_summary["nominal"].sum(), 1)
                ) * 100
                st.dataframe(
                    prod_summary,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "loan_product": "Skema KUR",
                        "alert": st.column_config.NumberColumn("Jumlah Alert", format="%d"),
                        "nominal": st.column_config.NumberColumn("Total Nominal Berisiko (Rp)", format="Rp %,.0f"),
                        "plafon_rata": st.column_config.NumberColumn("Rata-rata Plafon (Rp)", format="Rp %,.0f"),
                        "porsi_nominal": st.column_config.ProgressColumn("Porsi Nominal (%)", min_value=0, max_value=100, format="%.1f%%"),
                    },
                )
                st.caption("Porsi nominal tertinggi membutuhkan pengetatan batas kewenangan memutus kredit.")

            with tab_sekt:
                sekt_summary = (
                    high.groupby("business_sector")
                    .agg(
                        alert=("request_id", "count"),
                        nominal=("amount", "sum"),
                    )
                    .reset_index()
                    .sort_values("nominal", ascending=False)
                )
                sekt_summary["porsi_nominal"] = (
                    sekt_summary["nominal"] / max(sekt_summary["nominal"].sum(), 1)
                ) * 100
                st.dataframe(
                    sekt_summary,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "business_sector": "Sektor Usaha",
                        "alert": st.column_config.NumberColumn("Jumlah Alert", format="%d"),
                        "nominal": st.column_config.NumberColumn("Total Nominal Berisiko (Rp)", format="Rp %,.0f"),
                        "porsi_nominal": st.column_config.ProgressColumn("Porsi Nominal (%)", min_value=0, max_value=100, format="%.1f%%"),
                    },
                )
                st.caption("Gunakan data konsentrasi sektor untuk memandu sampling pemeriksaan audit kepatuhan.")

        with c_comm:
            st.markdown("##### 🕸️ 6 Komunitas Risiko Terbesar (Jejaring Kolusi)")
            st.markdown(
                "<div class='small-note' style='margin-bottom: 8px;'>"
                "Kelompok pegawai dengan intensitas persetujuan kredit mencurigakan dan hubungan berulang yang tidak wajar."
                "</div>",
                unsafe_allow_html=True,
            )
            communities = services.table("communities").head(6)
            st.dataframe(
                communities[["community_id", "skor_komunitas", "anggota", "unit", "aktif_bulan"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "community_id": "ID Klaster",
                    "skor_komunitas": st.column_config.ProgressColumn("Skor Risiko", min_value=0, max_value=100, format="%.0f"),
                    "anggota": "Jml Pegawai",
                    "unit": "Unit Utama",
                    "aktif_bulan": "Bulan Aktif",
                },
            )
            st.caption("Detail interaktif visualisasi jejaring antar-pegawai dapat dilihat pada menu **Peta Jaringan & Komplotan**.")

    with tab_branch:
        u_frame = unit_risk[unit_risk["year"] == year].copy()
        if regions:
            u_frame = u_frame[u_frame["region"].isin(regions)]
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
