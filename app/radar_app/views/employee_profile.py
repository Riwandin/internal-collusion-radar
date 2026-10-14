"""Investigator · Profil Pegawai: filter & cari pegawai → scorecard jendela waktu → tren anomali, red flag, rekan, alert."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import RED_FLAG_COLUMNS

FEATURE_LABELS = {"ex_near_limit": "near-limit vs rekan", "ex_after_hours": "luar jam kerja vs rekan",
                  "ex_fast_approval": "approval kilat vs rekan", "ex_debtor_repeat": "debitur berulang vs rekan",
                  "ex_pair_dependency": "ketergantungan requester vs rekan", "ex_multi_flag": "multi rule vs rekan",
                  "approve_rate": "tingkat approve", "top_requester_share": "konsentrasi 1 requester",
                  "ga_risky_pairs": "[graph] pasangan berisiko", "ga_max_pair_z": "[graph] z pasangan terburuk",
                  "ga_shared_debtor_share": "[graph] % debitur bersama", "ga_comm_score": "[graph] skor komunitas",
                  "ga_comm_size": "[graph] ukuran komunitas", "ga_risk_pagerank": "[graph] sentralitas"}
ACCESS_LABELS = {"ax_offhours_login": "login luar jam kerja", "ax_offhours_activity": "view/export luar jam kerja",
                 "ax_export_per_login": "export per login", "ax_view_per_login": "view per login"}


WINDOWS = {"30 hari": 30, "90 hari": 90, "180 hari": 180, "365 hari": 365}


def employee_table() -> pd.DataFrame:
    employees = services.table("employees").copy()
    anomaly = services.table("approver_anomaly")
    latest = anomaly[anomaly["month"] == anomaly["month"].max()].set_index("approver_id")
    employees["skor anomali"] = employees["employee_id"].map(latest["if_score"])
    employees["region"] = employees["region"].fillna("Lainnya")
    return employees.sort_values("skor anomali", ascending=False, na_position="last")


def select_employee(employees: pd.DataFrame) -> tuple[pd.Series | None, int]:
    search_slot = st.container()
    c_role, c_region, c_unit, c_window = st.columns(4)
    role = c_role.selectbox("Role", ["Semua", "Approver", "Requester"], key="emp_role")
    frame = employees if role == "Semua" else employees[employees["role"] == role]
    regions = c_region.multiselect("Region", sorted(frame["region"].unique()), key="emp_regions")
    if regions:
        frame = frame[frame["region"].isin(regions)]
    units = c_unit.multiselect("Unit kerja", sorted(frame["unit_name"].unique()), key="emp_units")
    if units:
        frame = frame[frame["unit_name"].isin(units)]
    window = WINDOWS[c_window.selectbox("Jendela waktu", list(WINDOWS), index=1, key="emp_window")]
    if frame.empty:
        search_slot.selectbox("🔎 Cari pegawai (nama / ID)", ["(tidak ada yang cocok)"], disabled=True)
        return None, window
    ids = frame["employee_id"].tolist()
    info = frame.set_index("employee_id")
    employee_id = search_slot.selectbox(
        "🔎 Cari pegawai (nama / ID)", ids, key="emp_pick",
        format_func=lambda e: f"{info.at[e, 'employee_name']} ({e}) · {info.at[e, 'role']} · {info.at[e, 'unit_name']}"
                              + (f" · skor anomali {info.at[e, 'skor anomali']:.0f}" if pd.notna(info.at[e, 'skor anomali']) else ""))
    return info.loc[employee_id].rename(employee_id), window


def render() -> None:
    ui.page_header("Profil Pegawai", "Alur: saring & cari pegawai → baca scorecard dalam jendela waktu → lihat tren, red flag, dan rekan kerjanya")
    employees = employee_table()
    employee, window = select_employee(employees)
    if employee is None:
        st.info("Tidak ada pegawai yang cocok dengan filter.")
        return
    employee_id, is_approver = employee.name, employee["role"] == "Approver"
    transactions = services.table("transactions_scored")
    end = transactions["event_time"].max()
    start = end - pd.Timedelta(days=window)
    own_column = "approver_id" if is_approver else "requester_id"
    own = transactions[transactions[own_column] == employee_id]
    recent = own[own["event_time"] > start]
    alerts = recent[recent["risk_band"] != "Rendah"]
    anomaly = services.table("approver_anomaly")
    history = anomaly[anomaly["approver_id"] == employee_id].sort_values("month")
    latest = history.iloc[-1] if len(history) else None
    approved = int((recent["decision"] == "Approved").sum())
    top = recent.loc[recent["risk_score"].idxmax()] if len(recent) else None

    st.markdown(f"### {employee['employee_name']} ({employee_id})")
    st.caption(f"{employee['role']} · {employee['position']} · {employee['unit_name']} ({employee['region']}) · bergabung "
               f"{pd.to_datetime(employee['employment_start_date']).strftime('%b %Y') if pd.notna(employee['employment_start_date']) else '-'}"
               f" · jendela {start:%d %b %Y} – {end:%d %b %Y}")
    ui.kpi_row([
        ("Skor anomali", f"{latest['if_score']:.0f}/100" if latest is not None else "-",
         f"{latest['anomaly_band']} · {latest['month']}" if latest is not None else "hanya approver yang dinilai",
         ui.ANOMALY_COLORS.get(latest["anomaly_band"], ui.BLUE) if latest is not None else ui.MUTED),
        ("Tingkat approve", ui.percent(approved / len(recent)) if len(recent) else "-",
         f"{ui.number(approved)} dari {ui.number(len(recent))} {'keputusan' if is_approver else 'pengajuan'} · {window} hari", ui.BLUE),
        ("Alert dalam jendela", ui.number(len(alerts)), f"Tinggi: {ui.number((alerts['risk_band'] == 'Tinggi').sum())}", ui.ORANGE),
        ("Alert tertinggi", f"{top['risk_score']:.0f}/100" if top is not None else "-",
         f"{top['request_id']} · {top['risk_band']}" if top is not None else "", ui.RED),
        ("Nominal pada alert", ui.rupiah(alerts["amount"].sum()), f"sebagai {employee['role'].lower()}", ui.YELLOW)])
    ui.explain({
        "Skor anomali": "persentil Isolation Forest: seberapa menyimpang pola approval dibanding seluruh approver-bulan "
                        "(≥ 95 perlu dilihat, ≥ 80 pantau). Requester tidak dinilai model ini.",
        "Tingkat approve": "persentase transaksi berstatus Approved dalam jendela waktu terpilih (sebagai approver: yang ia setujui; "
                           "sebagai requester: pengajuannya yang disetujui).",
        "Alert tertinggi": "transaksi dengan skor risiko kolusi tertinggi milik pegawai ini dalam jendela waktu.",
        "Red flag pegawai": "% transaksi pegawai yang memenuhi tiap rule dibanding seluruh transaksi bank pada jendela yang sama."})
    if latest is not None:
        st.markdown(f"**Alasan anomali (SHAP Isolation Forest):** {latest['if_reason'] or 'skor di bawah 80, tidak diurai'}")

    left, right = st.columns(2)
    if len(history):
        figure = px.line(history.tail(24), x="month", y="if_score", markers=True, labels={"month": "bulan", "if_score": "skor anomali"})
        figure.add_hline(y=95, line_dash="dash", line_color=ui.ORANGE, annotation_text="perlu dilihat")
        figure.add_hline(y=80, line_dash="dot", line_color=ui.YELLOW, annotation_text="pantau")
        left.plotly_chart(ui.style_figure(figure, "Tren skor anomali (24 bulan)", "persentil terhadap semua approver-bulan"),
                          use_container_width=True)
    else:
        monthly = own.groupby(own["event_time"].dt.to_period("M").astype(str))["risk_score"].agg(["mean", "max"]).reset_index()
        figure = px.line(monthly, x="event_time", y=["mean", "max"], markers=True,
                         labels={"event_time": "bulan", "value": "skor risiko", "variable": ""})
        left.plotly_chart(ui.style_figure(figure, "Tren skor risiko transaksi", "rata-rata & maksimum per bulan (requester)"),
                          use_container_width=True)
    peers = transactions[(transactions["event_time"] > start)]
    flags = pd.DataFrame({"rule": [ui.RED_FLAG_LABELS[c] for c in RED_FLAG_COLUMNS],
                          "seluruh transaksi": [peers[c].mean() * 100 for c in RED_FLAG_COLUMNS],
                          "pegawai ini": [recent[c].mean() * 100 if len(recent) else 0 for c in RED_FLAG_COLUMNS]})
    figure = go.Figure()
    figure.add_bar(y=flags["rule"], x=flags["seluruh transaksi"], orientation="h", name="seluruh transaksi", marker_color=ui.MUTED)
    figure.add_bar(y=flags["rule"], x=flags["pegawai ini"], orientation="h", name="pegawai ini", marker_color=ui.ORANGE)
    figure.update_layout(barmode="group", xaxis_title="% transaksi yang memenuhi rule")
    right.plotly_chart(ui.style_figure(figure, "Red flag pegawai ini", f"{window} hari terakhir"), use_container_width=True)

    if latest is not None:
        peers_anomaly = anomaly[(anomaly["month"] == latest["month"]) & (anomaly["position"] == latest["position"])]
        features = [f for f in FEATURE_LABELS if f in history.columns]
        comparison = pd.DataFrame({"fitur": [FEATURE_LABELS[f] for f in features],
                                   "pegawai ini": [latest[f] for f in features],
                                   "median rekan sejabatan": [peers_anomaly[f].median() for f in features]})
        figure = go.Figure()
        figure.add_bar(y=comparison["fitur"], x=comparison["median rekan sejabatan"], orientation="h", name="median rekan", marker_color=ui.MUTED)
        figure.add_bar(y=comparison["fitur"], x=comparison["pegawai ini"], orientation="h", name="pegawai ini", marker_color=ui.ORANGE)
        figure.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
        st.plotly_chart(ui.style_figure(figure, "Dibanding rekan sejabatan (bulan terakhir)", "ex_* = kelebihan dalam SD", height=460),
                        use_container_width=True)
        render_access_context(latest, peers)
    else:
        st.info("Pegawai ini bukan approver (atau keputusannya < 10 dalam 90 hari), sehingga tidak dinilai Isolation Forest.")
    render_npl_monitoring(employee["employee_id"])
    pairs = services.table("pair_network")
    partner_column = "requester_id" if is_approver else "approver_id"
    partners = pairs[pairs[own_column] == employee_id].copy()
    partners["pasangan"] = partners[partner_column].map(services.name_map()) + " (" + partners[partner_column] + ")"
    partners = partners.sort_values(["risky", "n"], ascending=False).head(15)
    left, right = st.columns(2)
    with left:
        st.markdown("**Pasangan kerja (12 bulan, snapshot graph terbaru)**")
        st.dataframe(partners[["pasangan", "n", "k", "z", "risky", "dependency_lift"]], hide_index=True, use_container_width=True,
                     column_config={"n": "transaksi", "k": "ber-red-flag", "z": st.column_config.NumberColumn("z red flag", format="%.1f"),
                                    "risky": "berisiko", "dependency_lift": st.column_config.NumberColumn("lift ketergantungan", format="%.1f")})
    with right:
        st.markdown(f"**Alert terkait ({window} hari)**")
        st.dataframe(alerts.sort_values("risk_score", ascending=False)[["request_id", "event_time", "risk_score", "risk_band", "amount", "top_reasons"]],
                     hide_index=True, use_container_width=True,
                     column_config={"risk_score": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f"),
                                    "event_time": st.column_config.DatetimeColumn("waktu", format="DD MMM YYYY")})
    export = recent.drop(columns=["evidence"], errors="ignore").sort_values("event_time", ascending=False)
    st.download_button(f"Unduh transaksi pegawai ({window} hari, CSV)", export.to_csv(index=False),
                       f"transaksi_{employee_id}_{window}hari.csv", "text/csv")


def render_access_context(latest: pd.Series, peers: pd.DataFrame) -> None:
    """Perilaku akses sistem 90 hari terakhir (Login/View/Export) dibanding median rekan sejabatan."""
    columns = [c for c in ACCESS_LABELS if c in latest.index and pd.notna(latest[c])]
    if not columns:
        return
    in_model = bool(services.app_settings().get("graph_hetero", {}).get("use_access_features"))
    st.markdown("**Perilaku akses sistem (90 hari, selisih terhadap median rekan sejabatan)** · "
                + ("ikut menjadi fitur Pilar 2" if in_model else "konteks saja, tidak masuk skor (tidak lolos uji validation)"))
    values = pd.DataFrame({"perilaku": [ACCESS_LABELS[c] for c in columns], "selisih vs rekan": [latest[c] for c in columns],
                           "persentil di antara rekan": [(peers[c] <= latest[c]).mean() * 100 if c in peers else np.nan
                                                         for c in columns]})
    st.dataframe(values, hide_index=True, use_container_width=True,
                 column_config={"selisih vs rekan": st.column_config.NumberColumn(format="%+.2f"),
                                "persentil di antara rekan": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
    st.caption("Log akses tidak punya request_id, jadi perilaku ini tidak dikaitkan ke transaksi tertentu. Login malam bisa "
               "sah (lembur closing); baca bersama alasan anomali lain.")


def render_npl_monitoring(employee_id: str) -> None:
    """Monitoring pascapencairan: porsi kredit yang sudah NPL. BUKAN fitur skor."""
    monitoring = services.table("npl_monitoring")
    if monitoring.empty:
        return
    own = monitoring[monitoring["employee_id"] == employee_id]
    if own.empty:
        return
    with st.expander("Monitoring pascapencairan: kredit yang sudah NPL (bukan bagian skor)"):
        all_roles = monitoring.groupby("peran")["porsi_npl"].median()
        view = own.assign(**{"median pegawai lain": own["peran"].map(all_roles)})
        st.dataframe(view[["peran", "kredit_cair", "npl_diketahui", "porsi_npl", "median pegawai lain"]], hide_index=True,
                     use_container_width=True,
                     column_config={"kredit_cair": "kredit cair", "npl_diketahui": "sudah NPL",
                                    "porsi_npl": st.column_config.NumberColumn("porsi NPL", format="%.3f"),
                                    "median pegawai lain": st.column_config.NumberColumn(format="%.3f")})
        st.caption(str(own["catatan"].iloc[0]))
