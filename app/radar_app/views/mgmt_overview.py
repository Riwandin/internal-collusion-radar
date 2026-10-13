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


    # KPI Delta calculations (vs previous year)
    prev_year = year - 1
    prev_frame = transactions[transactions["event_time"].dt.year == prev_year]
    if regions:
        prev_frame = prev_frame[prev_frame["requester_unit"].map(regions_map).isin(regions)]
    prev_high = prev_frame[prev_frame["risk_band"] == "Tinggi"]

    tot_tx = len(frame)
    prev_tot_tx = len(prev_frame)
    delta_tot = ((tot_tx - prev_tot_tx) / max(prev_tot_tx, 1) * 100) if prev_tot_tx else 0

    n_high = len(high)
    prev_n_high = len(prev_high)
    pct_high = (n_high / max(tot_tx, 1) * 100)
    delta_high = ((n_high - prev_n_high) / max(prev_n_high, 1) * 100) if prev_n_high else 0

    peg_high = set(high["approver_id"]).union(set(high["requester_id"]))
    prev_peg_high = set(prev_high["approver_id"]).union(set(prev_high["requester_id"]))
    n_peg = len(peg_high)
    prev_n_peg = len(prev_peg_high)
    delta_peg = ((n_peg - prev_n_peg) / max(prev_n_peg, 1) * 100) if prev_n_peg else 0

    exp_high = high["amount"].sum()
    prev_exp_high = prev_high["amount"].sum()
    delta_exp = ((exp_high - prev_exp_high) / max(prev_exp_high, 1) * 100) if prev_exp_high else 0

    # Top KPI cards in Executive White Card Style
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class='exec-card' style='border-top: 4px solid {ui.BLUE};'>
                <div class='kpi-label'>Total Transaksi</div>
                <div class='kpi-value'>{tot_tx:,}</div>
                <div class='kpi-sub' style='color: {"#16a34a" if delta_tot <= 0 else "#dc2626"}; font-weight: 600;'>
                    {'↓' if delta_tot <= 0 else '↑'} {abs(delta_tot):.1f}% <span style='font-weight: normal; color: {ui.MUTED};'>dari periode sebelumnya</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class='exec-card' style='border-top: 4px solid {ui.ORANGE};'>
                <div class='kpi-label'>Transaksi Berisiko</div>
                <div class='kpi-value'>{n_high:,} <span style='font-size: 1.05rem; font-weight: 500; color: {ui.INK_SECONDARY};'>({pct_high:.1f}%)</span></div>
                <div class='kpi-sub' style='color: {"#dc2626" if delta_high > 0 else "#16a34a"}; font-weight: 600;'>
                    {'↑' if delta_high > 0 else '↓'} {abs(delta_high):.1f}% <span style='font-weight: normal; color: {ui.MUTED};'>dari periode sebelumnya</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class='exec-card' style='border-top: 4px solid {ui.PURPLE};'>
                <div class='kpi-label'>Pegawai Berisiko</div>
                <div class='kpi-value'>{n_peg:,}</div>
                <div class='kpi-sub' style='color: {"#dc2626" if delta_peg > 0 else "#16a34a"}; font-weight: 600;'>
                    {'↑' if delta_peg > 0 else '↓'} {abs(delta_peg):.1f}% <span style='font-weight: normal; color: {ui.MUTED};'>dari periode sebelumnya</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class='exec-card' style='border-top: 4px solid {ui.RED};'>
                <div class='kpi-label'>Potensi Eksposur</div>
                <div class='kpi-value'>{ui.rupiah(exp_high)}</div>
                <div class='kpi-sub' style='color: {"#dc2626" if delta_exp > 0 else "#16a34a"}; font-weight: 600;'>
                    {'↑' if delta_exp > 0 else '↓'} {abs(delta_exp):.1f}% <span style='font-weight: normal; color: {ui.MUTED};'>dari periode sebelumnya</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    tab_exec, tab_branch = st.tabs([
        "📊 Ringkasan Eksekutif & Tren",
        "🗺️ Peta Risiko Wilayah & Kantor Cabang (Drill-Down)",
    ])

    with tab_exec:
        # Row 1: Tren Transaksi Berisiko & Distribusi Red Flag
        r1_left, r1_right = st.columns([1.45, 1])

        with r1_left:
            st.markdown(
                """<div style='margin-bottom: 6px; font-weight: 700; font-size: 1.05rem;'>Tren Transaksi Berisiko</div>""",
                unsafe_allow_html=True,
            )
            # Monthly trend for current year
            m_stats = frame.groupby(frame["event_time"].dt.month).agg(
                total=("request_id", "count"),
                high=("risk_band", lambda s: (s == "Tinggi").sum()),
            ).reset_index()
            months_id = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}
            m_stats["label_bulan"] = m_stats["event_time"].map(months_id)
            m_stats["pct"] = (m_stats["high"] / m_stats["total"].replace(0, np.nan)) * 100

            fig_t = go.Figure()
            fig_t.add_bar(
                x=m_stats["label_bulan"],
                y=m_stats["high"],
                name="Transaksi Berisiko",
                marker=dict(color="#f87171", line=dict(color="#ef4444", width=1)),
                text=[f"{v:,}" for v in m_stats["high"]],
                textposition="outside",
                hovertemplate="<b>Bulan %{x}</b><br>Transaksi Berisiko: %{y:,}<extra></extra>",
            )
            fig_t.add_scatter(
                x=m_stats["label_bulan"],
                y=m_stats["pct"],
                name="Persentase terhadap total",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8, color="#2563eb", line=dict(color="white", width=2)),
                hovertemplate="<b>Bulan %{x}</b><br>Persentase: %{y:.1f}%<extra></extra>",
            )
            fig_t.update_layout(
                yaxis=dict(title="Jumlah transaksi", showgrid=True, gridcolor="#f1f5f9", range=[0, max(m_stats["high"]) * 1.3]),
                yaxis2=dict(title="Persentase", overlaying="y", side="right", showgrid=False, ticksuffix="%", range=[0, max(m_stats["pct"]) * 1.4]),
                xaxis=dict(title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=35, b=20, l=10, r=10),
                height=340,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_t, use_container_width=True)

        with r1_right:
            st.markdown(
                """<div style='margin-bottom: 6px; font-weight: 700; font-size: 1.05rem;'>Distribusi Red Flag Transaksi</div>""",
                unsafe_allow_html=True,
            )
            # Red flag donut chart
            rf_data = [
                ("Near Limit", int(high["rf_near_limit"].sum())),
                ("After Hours", int(high["rf_after_hours"].sum())),
                ("Fast Approval", int(high["rf_fast_approval"].sum())),
                ("Repeat Debtor", int(high["rf_debtor_repeat"].sum())),
                ("Pair Dependency", int(high["rf_pair_dependency"].sum())),
            ]
            df_rf = pd.DataFrame(rf_data, columns=["flag", "jumlah"])
            df_rf = df_rf[df_rf["jumlah"] > 0]
            fig_donut = px.pie(
                df_rf,
                names="flag",
                values="jumlah",
                hole=0.55,
                color="flag",
                color_discrete_map={
                    "Near Limit": "#ef4444",
                    "After Hours": "#f97316",
                    "Fast Approval": "#eab308",
                    "Repeat Debtor": "#8b5cf6",
                    "Pair Dependency": "#3b82f6",
                },
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,} alert (%{percent})<extra></extra>",
            )
            fig_donut.update_layout(
                margin=dict(t=20, b=20, l=10, r=10),
                height=340,
                legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.divider()

        # Row 2: Top 5 Unit dengan Transaksi Berisiko & Top 5 Sektor Usaha Berisiko
        r2_left, r2_right = st.columns(2)
        with r2_left:
            st.markdown(
                """<div style='margin-bottom: 10px; font-weight: 700; font-size: 1.05rem;'>Top 5 Unit dengan Transaksi Berisiko</div>""",
                unsafe_allow_html=True,
            )
            top_u_counts = high["requester_unit"].value_counts().head(5)
            u_totals = frame["requester_unit"].value_counts()
            for u_name, u_cnt in top_u_counts.items():
                tot_u = u_totals.get(u_name, max(u_cnt, 1))
                u_pct = (u_cnt / max(len(high), 1)) * 100
                st.markdown(
                    f"""<div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;'>
                        <div style='width: 140px; font-weight: 600; font-size: 0.9rem; color: {ui.INK};'>{u_name}</div>
                        <div style='flex: 1; margin: 0 16px; background: #fee2e2; border-radius: 999px; height: 16px; overflow: hidden;'>
                            <div style='width: {min(u_pct * 3.5, 100):.1f}%; background: #ef4444; height: 100%; border-radius: 999px;'></div>
                        </div>
                        <div style='width: 60px; text-align: right; font-weight: 700; font-size: 0.92rem;'>{u_cnt:,}</div>
                        <div style='width: 65px; text-align: right; color: {ui.INK_SECONDARY}; font-size: 0.85rem;'>{u_pct:.0f}% total</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with r2_right:
            st.markdown(
                """<div style='margin-bottom: 10px; font-weight: 700; font-size: 1.05rem;'>Top 5 Sektor Usaha Berisiko</div>""",
                unsafe_allow_html=True,
            )
            top_s_counts = high["business_sector"].value_counts().head(5)
            for s_name, s_cnt in top_s_counts.items():
                s_pct = (s_cnt / max(len(high), 1)) * 100
                st.markdown(
                    f"""<div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;'>
                        <div style='width: 140px; font-weight: 600; font-size: 0.9rem; color: {ui.INK};'>{s_name}</div>
                        <div style='flex: 1; margin: 0 16px; background: #fee2e2; border-radius: 999px; height: 16px; overflow: hidden;'>
                            <div style='width: {min(s_pct * 1.8, 100):.1f}%; background: #ef4444; height: 100%; border-radius: 999px;'></div>
                        </div>
                        <div style='width: 60px; text-align: right; font-weight: 700; font-size: 0.92rem;'>{s_cnt:,}</div>
                        <div style='width: 65px; text-align: right; color: {ui.INK_SECONDARY}; font-size: 0.85rem;'>{s_pct:.0f}% total</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.divider()

        # Row 3: Pegawai dengan Risiko Tertinggi & Dampak Potensial
        r3_left, r3_right = st.columns([1.45, 1])
        with r3_left:
            st.markdown(
                """<div style='margin-bottom: 10px; font-weight: 700; font-size: 1.05rem;'>Pegawai dengan Risiko Tertinggi</div>""",
                unsafe_allow_html=True,
            )
            emp_table = services.table("employees").set_index("employee_id")
            emp_names = emp_table["employee_name"].to_dict()
            emp_units = emp_table["unit_name"].to_dict()

            # Rank employees by exposure
            high_appr = high.groupby("approver_id").agg(
                n_high=("request_id", "count"),
                exp_high=("amount", "sum"),
            ).reset_index()
            tot_appr = frame.groupby("approver_id")["request_id"].count().to_dict()
            high_appr["total_tx"] = high_appr["approver_id"].map(tot_appr)
            high_appr["pct_high"] = (high_appr["n_high"] / high_appr["total_tx"].replace(0, np.nan) * 100)
            high_appr["nama"] = high_appr["approver_id"].map(emp_names)
            high_appr["unit"] = high_appr["approver_id"].map(emp_units)
            top_emp = high_appr.sort_values("exp_high", ascending=False).head(5)

            st.dataframe(
                top_emp[["nama", "unit", "total_tx", "pct_high", "exp_high"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "nama": "Nama",
                    "unit": "Unit Kerja",
                    "total_tx": st.column_config.NumberColumn("Jumlah Transaksi", format="%d"),
                    "pct_high": st.column_config.ProgressColumn("% Berisiko", min_value=0, max_value=100, format="%.0f%%"),
                    "exp_high": st.column_config.NumberColumn("Potensi Eksposur", format="Rp %,.0f"),
                },
            )
            st.caption("Peringkat dihitung dari total akumulasi plafon transaksi berisiko Tinggi yang disetujui.")

        with r3_right:
            st.markdown(
                """<div style='margin-bottom: 10px; font-weight: 700; font-size: 1.05rem;'>Dampak Potensial</div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class='exec-stat-box'>
                    <div style='font-size: 2rem;'>💰</div>
                    <div>
                        <div style='font-size: 1.35rem; font-weight: 700; color: {ui.INK};'>{ui.rupiah(exp_high)}</div>
                        <div style='font-size: 0.83rem; color: {ui.INK_SECONDARY};'>Total potensi eksposur dari transaksi berisiko</div>
                    </div>
                </div>
                <div class='exec-stat-box'>
                    <div style='font-size: 2rem;'>📊</div>
                    <div>
                        <div style='font-size: 1.35rem; font-weight: 700; color: {ui.INK};'>{n_high:,}</div>
                        <div style='font-size: 0.83rem; color: {ui.INK_SECONDARY};'>Transaksi berisiko ({pct_high:.1f}% dari total)</div>
                    </div>
                </div>
                <div class='exec-stat-box'>
                    <div style='font-size: 2rem;'>👥</div>
                    <div>
                        <div style='font-size: 1.35rem; font-weight: 700; color: {ui.INK};'>{n_peg:,}</div>
                        <div style='font-size: 0.83rem; color: {ui.INK_SECONDARY};'>Pegawai dengan keterlibatan berisiko</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Retained Dark Chart Section: Wilayah dengan Konsentrasi Alert Tertinggi & Konsentrasi Sektor / Komunitas
        st.markdown("#### 🔍 Peta Wilayah & Portofolio Risiko Terkonsentrasi")
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
        reg_top = reg_summary.dropna(subset=["alert_rate"]).sort_values("alert_rate").tail(8)

        # Retained dark background horizontal bar chart from attached screenshot
        fig_reg = px.bar(
            reg_top,
            x="alert_rate",
            y=grp_col,
            orientation="h",
            color="alert_rate",
            color_continuous_scale=["#f3c3a3", "#e07a4a", "#ea580c"],
            labels={"alert_rate": "% Transaksi Alert Tinggi", grp_col: ""},
        )
        fig_reg.update_layout(
            coloraxis_showscale=False,
            xaxis=dict(
                title="% Transaksi Alert Tinggi",
                tickformat=".1%",
                gridcolor="#262626",
                color="#ffffff",
                zerolinecolor="#333333",
            ),
            yaxis=dict(color="#ffffff", gridcolor="#262626"),
            paper_bgcolor="#14171a",
            plot_bgcolor="#14171a",
            font=dict(color="#ffffff"),
            margin=dict(l=10, r=20, t=50, b=20),
            height=380,
            title=dict(
                text=f"<b>{chart_title}</b><br><span style='font-size:12px;color:#a0a0a0;'>{chart_sub}</span>",
                font=dict(color="#ffffff", size=15),
                x=0,
            ),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

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
            st.markdown(
                """<div style='margin-bottom: 8px; font-weight: 700; font-size: 1.05rem; color: #1e3a8a;'>Unit dengan Risiko Tertinggi</div>""",
                unsafe_allow_html=True,
            )
            # Aggregate unit risk metrics
            unit_agg = (
                high.groupby("requester_unit")
                .agg(
                    n_berisiko=("request_id", "count"),
                    potensi_m=("amount", lambda s: s.sum() / 1e9),
                )
                .reset_index()
                .sort_values("potensi_m", ascending=False)
                .head(5)
            )

            # Build monthly trend array for each unit (sparkline)
            m_units = (
                high.groupby(["requester_unit", high["event_time"].dt.month])["request_id"]
                .count()
                .unstack(fill_value=0)
            )
            unit_agg["tren"] = unit_agg["requester_unit"].apply(
                lambda u: m_units.loc[u].tolist() if u in m_units.index else [1, 2, 3]
            )
            unit_agg["rank"] = range(1, len(unit_agg) + 1)

            st.dataframe(
                unit_agg[["rank", "requester_unit", "n_berisiko", "potensi_m", "tren"]],
                hide_index=True,
                use_container_width=True,
                height=260,
                column_config={
                    "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
                    "requester_unit": st.column_config.TextColumn("Unit", width="medium"),
                    "n_berisiko": st.column_config.NumberColumn("Jumlah Transaksi Berisiko", format="%d"),
                    "potensi_m": st.column_config.NumberColumn("Nilai Potensi Risiko (Rp M)", format="%.0f"),
                    "tren": st.column_config.LineChartColumn("Tren", width="medium"),
                },
            )
            st.caption("Peringkat unit/cabang dengan eksposur potensi risiko tertinggi beserta tren fluktuasi bulanan.")

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
