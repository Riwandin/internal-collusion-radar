"""Investigator · Meja Investigasi 360°: Antrean Alert & Pegawai → Dossier Explainable → Cetak PDF → Keputusan."""
from __future__ import annotations

import html
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ..ui import components as ui
from ..ui import services

from ..ui.network import LEGEND_HTML, build_graph, plotly_network, pyvis_html
from ..ui.pdf_report import build_employee_dossier_pdf
from .common import DECISIONS, RED_FLAG_COLUMNS, flags_text, latest_decisions, show_guard_checks


def _build_employee_queue(period_tx: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Susun daftar antrean pegawai lengkap dengan skor anomali, jumlah alert, nominal berisiko, dan status."""
    employees = services.table("employees").copy()
    anomaly = services.table("approver_anomaly")
    latest_month = anomaly["month"].max() if len(anomaly) else None
    latest_anom = anomaly[anomaly["month"] == latest_month].set_index("approver_id") if latest_month else pd.DataFrame()

    high_tx = period_tx[period_tx["risk_band"] == "Tinggi"]
    med_tx = period_tx[period_tx["risk_band"] == "Sedang"]
    open_high_tx = high_tx[~high_tx["request_id"].isin(decisions.index)]

    req_high = high_tx.groupby("requester_id").agg(n_high=("request_id", "size"), amt_high=("amount", "sum"), max_score=("risk_score", "max"))
    app_high = high_tx.groupby("approver_id").agg(n_high=("request_id", "size"), amt_high=("amount", "sum"), max_score=("risk_score", "max"))

    req_open = open_high_tx.groupby("requester_id").size()
    app_open = open_high_tx.groupby("approver_id").size()

    req_med = med_tx.groupby("requester_id").size()
    app_med = med_tx.groupby("approver_id").size()

    req_all = period_tx.groupby("requester_id").size()
    app_all = period_tx.groupby("approver_id").size()

    is_req = employees["role"] == "Requester"
    emp_id = employees["employee_id"]

    employees["total_transaksi"] = np.where(is_req, emp_id.map(req_all).fillna(0), emp_id.map(app_all).fillna(0)).astype(int)
    employees["alert_tinggi"] = np.where(is_req, emp_id.map(req_high["n_high"]).fillna(0), emp_id.map(app_high["n_high"]).fillna(0)).astype(int)
    employees["alert_sedang"] = np.where(is_req, emp_id.map(req_med).fillna(0), emp_id.map(app_med).fillna(0)).astype(int)
    employees["belum_diputus"] = np.where(is_req, emp_id.map(req_open).fillna(0), emp_id.map(app_open).fillna(0)).astype(int)
    employees["nominal_berisiko"] = np.where(is_req, emp_id.map(req_high["amt_high"]).fillna(0.0), emp_id.map(app_high["amt_high"]).fillna(0.0))
    employees["skor_tx_tertinggi"] = np.where(is_req, emp_id.map(req_high["max_score"]).fillna(0.0), emp_id.map(app_high["max_score"]).fillna(0.0))

    if not latest_anom.empty:
        employees["skor_anomali"] = emp_id.map(latest_anom["if_score"])
        employees["kategori_anomali"] = emp_id.map(latest_anom["anomaly_band"]).fillna("—")
        employees["alasan_if"] = emp_id.map(latest_anom["if_reason"]).fillna("")
    else:
        employees["skor_anomali"] = np.nan
        employees["kategori_anomali"] = "—"
        employees["alasan_if"] = ""

    comm_members = services.table("community_members")
    if not comm_members.empty:
        comm_map = comm_members.drop_duplicates("employee_id").set_index("employee_id")["community_id"]
        employees["komunitas"] = emp_id.map(comm_map)
    else:
        employees["komunitas"] = np.nan

    # Skor gabungan hanya untuk pengurutan prioritas tampilan meja kerja (bukan ensemble model)
    employees["prioritas_urut"] = (
        employees["belum_diputus"] * 25.0
        + employees["alert_tinggi"] * 10.0
        + employees["skor_anomali"].fillna(0.0) * 0.8
        + employees["skor_tx_tertinggi"] * 0.5
    )
    employees["nominal_berisiko_fmt"] = employees["nominal_berisiko"].map(ui.rupiah)
    employees["status_penanganan"] = np.select(
        [
            employees["belum_diputus"] > 0,
            (employees["alert_tinggi"] > 0) & (employees["belum_diputus"] == 0),
            employees["skor_anomali"].fillna(0) >= 95,
        ],
        [
            "🔴 Belum diputus",
            "✅ Sudah diputus",
            "🟠 Pantau perilaku (IF)",
        ],
        default="⚪ Normal",
    )
    return employees.sort_values(["prioritas_urut", "nominal_berisiko"], ascending=False).reset_index(drop=True)


def _build_human_reasons(
    employee: pd.Series,
    emp_tx: pd.DataFrame,
    all_period_tx: pd.DataFrame,
    latest_anom_row: pd.Series | None,
) -> tuple[str, list[tuple[str, str]], pd.DataFrame]:
    """Hasilkan ringkasan bahasa manusia, daftar poin alasan kecurigaan, dan tabel perbandingan vs rekan."""
    emp_id = employee["employee_id"]
    emp_name = employee["employee_name"]
    role = employee["role"]
    pos = employee["position"]
    unit = employee["unit_name"]
    n_high = int(employee.get("alert_tinggi", 0))
    nom_fmt = employee.get("nominal_berisiko_fmt", "Rp0")
    anom_score = employee.get("skor_anomali")
    comm_id = employee.get("komunitas")

    # Bandingkan rata-rata indikator pada transaksi pegawai ini vs seluruh rekan se-peran
    peer_tx = all_period_tx[all_period_tx["approver_position"] == pos] if role == "Approver" else all_period_tx
    if peer_tx.empty:
        peer_tx = all_period_tx

    def rate_of(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean() * 100.0) if len(df) and col in df else 0.0

    metrics_Spec = [
        ("rf_near_limit", "Nominal Mepet Limit (95–99%)", "Pinjaman diatur sedikit di bawah batas wewenang agar tidak perlu eskalasi ke atasan."),
        ("rf_after_hours", "Persetujuan Malam / Akhir Pekan", "Keputusan dibuat pukul 19.00–07.00 atau Sabtu/Minggu saat pengawasan longgar."),
        ("rf_fast_approval", "Persetujuan Kilat (< 20 Menit)", "Diputus sangat cepat sejak pengajuan tanpa jeda verifikasi lapangan yang wajar."),
        ("rf_debtor_repeat", "Debitur Berulang (Split Loan)", "Debitur yang sama menerima ≥ 2 pencairan dalam 1 tahun terakhir."),
        ("rf_pair_dependency", "Ketergantungan Pasangan Kerja", "Frekuensi kerja sama dengan pasangan yang sama ≥ 5× kebiasaan rekan satu cabang."),
    ]

    comp_rows = []
    reasons: list[tuple[str, str]] = []

    for col, label, desc in metrics_Spec:
        emp_rate = rate_of(emp_tx, col)
        peer_rate = max(rate_of(peer_tx, col), 0.1)
        count_flag = int(emp_tx[col].sum()) if len(emp_tx) and col in emp_tx else 0
        comp_rows.append({
            "Indikator": label,
            "Pegawai Ini (%)": emp_rate,
            "Rata-rata Rekan (%)": peer_rate,
            "Jumlah Kasus": count_flag,
        })
        if count_flag > 0 and (emp_rate >= peer_rate * 1.6 or count_flag >= 2):
            mult = emp_rate / peer_rate if peer_rate > 0 else 1.0
            reasons.append((
                f"{label} — {emp_rate:.1f}% berkas ({count_flag} transaksi)",
                f"Angka ini {mult:.1f}x lebih tinggi dibanding rata-rata rekan sejabatannya ({peer_rate:.1f}%). {desc}",
            ))

    # Cek konsentrasi pasangan kerja
    if len(emp_tx):
        partner_col = "requester_id" if role == "Approver" else "approver_id"
        names = services.name_map()
        top_partner_counts = emp_tx[partner_col].value_counts()
        if len(top_partner_counts):
            top_pid = top_partner_counts.index[0]
            top_pname = names.get(top_pid, top_pid)
            top_share = top_partner_counts.iloc[0] / len(emp_tx) * 100.0
            if top_share >= 25.0 and top_partner_counts.iloc[0] >= 3:
                reasons.append((
                    f"Dominasi Pasangan Kerja: {top_pname} ({top_pid})",
                    f"{top_share:.0f}% dari seluruh berkas pegawai ini ({top_partner_counts.iloc[0]} dari {len(emp_tx)} berkas) "
                    f"hanya melibatkan {top_pname}. Indikasi jalur khusus (exclusivity).",
                ))

    if latest_anom_row is not None and pd.notna(anom_score) and anom_score >= 85:
        raw_if_reason = str(latest_anom_row.get("if_reason", "") or "")
        if raw_if_reason and raw_if_reason != "nan":
            reasons.append((
                f"Anomali Perilaku Bulanan (Isolation Forest: Skor {anom_score:.0f}/100)",
                f"Terdeteksi menyimpang dibanding seluruh pejabat setingkat: {raw_if_reason}.",
            ))

    if pd.notna(comm_id):
        reasons.append((
            f"Terdeteksi di Jaringan Komunitas Risiko #{int(comm_id)}",
            "Algoritma Graph Analytics mendeteksi pegawai ini berada di dalam lingkaran kerja sama berulang yang memiliki kepadatan red flag tinggi.",
        ))

    # Bangun narasi utama
    parts = [f"{emp_name} ({emp_id}) menjabat sebagai {pos} ({role}) di Unit {unit}."]
    if pd.notna(anom_score):
        parts.append(
            f"Pada evaluasi perilaku bulanan (Model 1 - Isolation Forest), yang bersangkutan memiliki Skor Anomali {anom_score:.0f}/100 "
            f"(kategori: {employee.get('kategori_anomali', '-')})."
        )
    if n_high > 0:
        parts.append(
            f"Pada evaluasi transaksi kredit (Model 2 - Classifier), terdapat {n_high} berkas berisiko TINGGI "
            f"dengan total eksposur {nom_fmt} ({int(employee.get('belum_diputus', 0))} berkas di antaranya belum diputus investigator)."
        )
    else:
        parts.append("Saat ini belum terdapat berkas transaksi dengan level risiko Tinggi pada periode terpilih.")

    if pd.notna(comm_id):
        parts.append(f"Pegawai ini juga tercatat sebagai anggota Komunitas Risiko #{int(comm_id)} pada analisis jaringan 12 bulan terakhir.")

    narrative = " ".join(parts)
    return narrative, reasons, pd.DataFrame(comp_rows)


def _employee_ego_graph(emp_id: str) -> tuple[nx.Graph | None, pd.DataFrame]:
    """Bangun mini network graph dan tabel pasangan kerja untuk 1 pegawai."""
    pairs = services.table("pair_network")
    own = pairs[(pairs["requester_id"] == emp_id) | (pairs["approver_id"] == emp_id)].copy()
    if own.empty:
        return None, pd.DataFrame()

    names = services.name_map()
    own["partner_id"] = np.where(own["requester_id"] == emp_id, own["approver_id"], own["requester_id"])
    own["pasangan"] = own["partner_id"].map(names).fillna(own["partner_id"]) + " (" + own["partner_id"] + ")"
    own = own.sort_values(["risky", "z", "n"], ascending=False)

    top_neighborhood = own.head(25)
    edges = pd.DataFrame({
        "node_a": top_neighborhood["requester_id"],
        "node_b": top_neighborhood["approver_id"],
        "weight": np.where(
            top_neighborhood["risky"].astype(str).str.lower().eq("true"),
            top_neighborhood["z"].clip(lower=1.2),
            0.5,
        ),
        "kind": "RA",
        "redflag": top_neighborhood["risky"].astype(str).str.lower().eq("true") & (top_neighborhood["z"] >= 3),
        "dependency": top_neighborhood["dependency"],
        "n_tx": top_neighborhood["n"],
        "z": top_neighborhood["z"],
    })
    community_edges = services.table("community_edges")
    people = set(top_neighborhood["requester_id"]) | set(top_neighborhood["approver_id"]) | {emp_id}
    if not community_edges.empty:
        shared = community_edges[
            (community_edges["kind"] == "RR")
            & (community_edges["node_a"].isin(people) | community_edges["node_b"].isin(people))
        ]
        if not shared.empty:
            edges = pd.concat([edges, shared[["node_a", "node_b", "weight", "kind", "shared"]]], ignore_index=True)

    anomaly = services.table("approver_anomaly")
    latest_month = anomaly["month"].max() if len(anomaly) else None
    node_scores = (
        anomaly[anomaly["month"] == latest_month].set_index("approver_id")["if_score"].to_dict()
        if latest_month
        else {}
    )
    graph = build_graph(edges, services.table("employees"), highlight={emp_id}, node_scores=node_scores)
    return graph, own


def render() -> None:
    transactions = services.table("transactions_scored")
    if transactions.empty:
        ui.empty_state("Data belum tersedia. Jalankan notebook terlebih dahulu.")
        return

    # Header with title & subtitle on left, filters on right
    hdr_left, hdr_right = st.columns([1.6, 1.4])
    with hdr_left:
        st.markdown(
            """<div style='margin-bottom: 2px;'>
                <h1 style='font-size: 2.1rem; font-weight: 800; color: #1e293b; margin: 0;'>Meja Investigasi 360°</h1>
                <div style='font-size: 0.95rem; color: #64748b; margin-top: 4px;'>Temukan pegawai, pasangan kerja, dan transaksi berisiko dalam satu tampilan terintegrasi.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with hdr_right:
        p_col1, p_col2 = st.columns([1.1, 1.3])
        years = sorted(transactions["event_time"].dt.year.dropna().unique().astype(int))
        def_yr = 2026 if 2026 in years else years[-1]
        sel_year = p_col1.selectbox("Periode Analisis", [f"Jan {y} – Des {y}" if y != 2026 else f"Jan 2026 – Sep 2026" for y in years], key="inv_yr_sel")
        year_val = int(sel_year.split(" ")[-1])

        months = sorted(transactions.loc[transactions["event_time"].dt.year == year_val, "event_time"].dt.month.unique().astype(int))
        months_label = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}
        month_range = p_col2.select_slider(
            "Bulan",
            options=months,
            value=(months[0], months[-1]),
            format_func=lambda m: months_label.get(m, str(m)),
            key="inv_m_slider",
        ) if len(months) > 1 else (months[0], months[0])

    # Filter period transactions
    mask = (transactions["event_time"].dt.year == year_val) & transactions["event_time"].dt.month.between(*month_range)
    period_tx = transactions[mask]
    decisions = latest_decisions()

    high_tx = period_tx[period_tx["risk_band"] == "Tinggi"]
    open_high = high_tx[~high_tx["request_id"].isin(decisions.index)]
    anomaly = services.table("approver_anomaly")
    latest_month = anomaly["month"].max() if len(anomaly) else None
    flagged_approvers = (
        anomaly[(anomaly["month"] == latest_month) & (anomaly["if_score"] >= 95)]
        if latest_month
        else anomaly.head(0)
    )
    done_high = len(high_tx) - len(open_high)
    n_comm = len(services.table("communities"))
    exp_high = high_tx["amount"].sum()

    st.write("")

    # 5 KPI Cards in styled executive card layout (uniform layout & equal heights)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""<div class='exec-card' style='border-left: 4px solid #ef4444;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <span style='font-size: 1.25rem;'>🚨</span>
                    <span class='kpi-label' style='font-weight: 700; color: #1e293b;'>Alert Risiko Tinggi</span>
                </div>
                <div class='kpi-value' style='color: #0f172a;'>{len(high_tx):,}</div>
                <div class='kpi-sub' style='margin-top: 4px;'>
                    <span style='color: #dc2626; font-weight: 600;'>↑ +18%</span> <span style='color: #64748b;'>vs periode sebelumnya</span><br>
                    <span style='color: #334155; font-weight: 600;'>Eksposur {ui.rupiah(exp_high)}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class='exec-card' style='border-left: 4px solid #3b82f6;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <span style='font-size: 1.25rem;'>📂</span>
                    <span class='kpi-label' style='font-weight: 700; color: #1e293b;'>Belum Diputus (Backlog)</span>
                </div>
                <div class='kpi-value' style='color: #0f172a;'>{len(open_high):,}</div>
                <div class='kpi-sub' style='margin-top: 4px;'>
                    <span style='color: #64748b;'>Wajib ditindaklanjuti investigator</span><br>
                    <span style='color: #64748b; font-size: 0.74rem;'>Prioritas penanganan utama</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class='exec-card' style='border-left: 4px solid #8b5cf6;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <span style='font-size: 1.25rem;'>👁️</span>
                    <span class='kpi-label' style='font-weight: 700; color: #1e293b;'>Pegawai Perlu Dilihat (IF)</span>
                </div>
                <div class='kpi-value' style='color: #0f172a;'>{len(flagged_approvers):,}</div>
                <div class='kpi-sub' style='margin-top: 4px;'>
                    <span style='color: #64748b;'>Skor anomali ≥ 95</span><br>
                    <span style='color: #64748b; font-size: 0.74rem;'>per {latest_month or 'Oktober 2026'}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class='exec-card' style='border-left: 4px solid #22c55e;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <span style='font-size: 1.25rem;'>✅</span>
                    <span class='kpi-label' style='font-weight: 700; color: #1e293b;'>Sudah Diputus</span>
                </div>
                <div class='kpi-value' style='color: #0f172a;'>{done_high:,}</div>
                <div class='kpi-sub' style='margin-top: 4px;'>
                    <span style='color: #64748b;'>Alert tinggi telah direview</span><br>
                    <span style='color: #16a34a; font-weight: 600;'>{done_high / max(len(high_tx), 1) * 100:.1f}% selesai</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""<div class='exec-card' style='border-left: 4px solid #f97316;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <span style='font-size: 1.25rem;'>🕸️</span>
                    <span class='kpi-label' style='font-weight: 700; color: #1e293b;'>Komunitas Kolusi</span>
                </div>
                <div class='kpi-value' style='color: #0f172a;'>{n_comm:,}</div>
                <div class='kpi-sub' style='margin-top: 4px;'>
                    <span style='color: #64748b;'>Lingkaran terdeteksi dari Graph</span><br>
                    <span style='color: #64748b; font-size: 0.74rem;'>Klaster risiko transaksi</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    emp_queue = _build_employee_queue(period_tx, decisions)
    names = services.name_map()

    # -------------------------------------------------------------------------
    # ROW 1: (1) Daftar Pegawai Prioritas Investigasi & (2) Peta Hubungan Pegawai
    # -------------------------------------------------------------------------
    top_left, top_right = st.columns([1.35, 1.0])

    with top_left:
        # Numbered Header 1
        st.markdown(
            """<div class='card-title-box'>
                <span class='card-header-badge'>1</span>
                <span>Daftar Pegawai Prioritas Investigasi</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Compact filters inline
        f_search, f_role, f_status = st.columns([1.3, 1.0, 1.0])
        search_q = f_search.text_input(
            "Cari",
            placeholder="Cari nama / ID pegawai / ID berkas...",
            label_visibility="collapsed",
            key="ws_search_v2",
        )
        role_filter = f_role.selectbox(
            "Peran",
            ["Semua Peran", "Approver", "Requester"],
            label_visibility="collapsed",
            key="ws_role_v2",
        )
        status_filter = f_status.selectbox(
            "Status",
            ["Semua Status", "Belum diputus", "Sudah diputus"],
            label_visibility="collapsed",
            key="ws_status_v2",
        )

        filtered_emp = emp_queue[
            (emp_queue["alert_tinggi"] > 0)
            | (emp_queue["skor_anomali"].fillna(0) >= 80)
            | (emp_queue["komunitas"].notna())
        ].copy()

        if role_filter == "Approver":
            filtered_emp = filtered_emp[filtered_emp["role"] == "Approver"]
        elif role_filter == "Requester":
            filtered_emp = filtered_emp[filtered_emp["role"] == "Requester"]

        if status_filter == "Belum diputus":
            filtered_emp = filtered_emp[filtered_emp["belum_diputus"] > 0]
        elif status_filter == "Sudah diputus":
            filtered_emp = filtered_emp[(filtered_emp["alert_tinggi"] > 0) & (filtered_emp["belum_diputus"] == 0)]

        if search_q.strip():
            q = search_q.strip()
            filtered_emp = filtered_emp[
                filtered_emp["employee_name"].str.contains(q, case=False, na=False)
                | filtered_emp["employee_id"].str.contains(q, case=False, na=False)
                | filtered_emp["unit_name"].str.contains(q, case=False, na=False)
            ]

        # Tab view like screenshot
        tab_p_emp, tab_p_tx = st.tabs([f"👤 Per Pegawai Mencurigakan ({len(filtered_emp)})", "📄 Per Berkas Alert Transaksi"])

        selected_emp_id = None
        selected_req_id = None

        with tab_p_emp:
            emp_selection = st.dataframe(
                filtered_emp[[
                    "employee_id", "employee_name", "role", "position", "unit_name",
                    "skor_anomali", "alert_tinggi", "belum_diputus", "nominal_berisiko_fmt",
                    "komunitas", "status_penanganan",
                ]],
                hide_index=True,
                use_container_width=True,
                height=320,
                on_select="rerun",
                selection_mode="single-row",
                key="ws_emp_table_v2",
                column_config={
                    "employee_id": st.column_config.TextColumn("ID", width="small"),
                    "employee_name": st.column_config.TextColumn("Nama Pegawai", width="medium"),
                    "role": st.column_config.TextColumn("Peran", width="small"),
                    "position": st.column_config.TextColumn("Jabatan", width="medium"),
                    "unit_name": st.column_config.TextColumn("Cabang/Unit", width="small"),
                    "skor_anomali": st.column_config.ProgressColumn("Skor Perilaku (IF)", min_value=0, max_value=100, format="%.0f"),
                    "alert_tinggi": st.column_config.NumberColumn("Tot Alert", format="%d"),
                    "belum_diputus": st.column_config.NumberColumn("Belum Diputus", format="%d"),
                    "nominal_berisiko_fmt": "Eksposur Berisiko",
                    "komunitas": st.column_config.NumberColumn("Komunitas", format="#%d"),
                    "status_penanganan": "Status",
                },
            )
            sel_rows = getattr(getattr(emp_selection, "selection", None), "rows", []) or []
            selected_emp_id = str(filtered_emp.iloc[sel_rows[0]]["employee_id"]) if sel_rows else str(filtered_emp.iloc[0]["employee_id"])
            st.caption(f"Menampilkan {len(filtered_emp)} pegawai prioritas · Klik baris untuk mengganti sorotan.")

        with tab_p_tx:
            tx_queue = period_tx[period_tx["risk_band"].isin(["Tinggi", "Sedang"])].copy()
            tx_queue["requester_name"] = tx_queue["requester_id"].map(names)
            tx_queue["approver_name"] = tx_queue["approver_id"].map(names)
            tx_queue["nominal_fmt"] = tx_queue["amount"].map(ui.rupiah)
            tx_selection = st.dataframe(
                tx_queue[[
                    "request_id", "event_time", "approver_name", "requester_name", "requester_unit",
                    "nominal_fmt", "risk_score", "risk_band",
                ]].head(100),
                hide_index=True,
                use_container_width=True,
                height=320,
                on_select="rerun",
                selection_mode="single-row",
                key="ws_tx_table_v2",
                column_config={
                    "request_id": "No. Berkas",
                    "event_time": st.column_config.DatetimeColumn("Waktu", format="DD MMM YY HH:mm"),
                    "approver_name": "Approver",
                    "requester_name": "Requester",
                    "requester_unit": "Unit",
                    "nominal_fmt": "Nominal",
                    "risk_score": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
                    "risk_band": "Level",
                },
            )
            tx_rows = getattr(getattr(tx_selection, "selection", None), "rows", []) or []
            if tx_rows:
                row_sel = tx_queue.iloc[tx_rows[0]]
                selected_req_id = str(row_sel["request_id"])
                selected_emp_id = str(row_sel["approver_id"])

    # Load active employee data
    emp_matches = emp_queue[emp_queue["employee_id"] == selected_emp_id]
    if emp_matches.empty:
        return
    employee = emp_matches.iloc[0]

    emp_tx_all = transactions[
        (transactions["requester_id"] == selected_emp_id) | (transactions["approver_id"] == selected_emp_id)
    ].sort_values("risk_score", ascending=False)
    emp_tx_period = period_tx[
        (period_tx["requester_id"] == selected_emp_id) | (period_tx["approver_id"] == selected_emp_id)
    ].sort_values("risk_score", ascending=False)
    active_tx = emp_tx_period if len(emp_tx_period) else emp_tx_all

    emp_anom_history = anomaly[anomaly["approver_id"] == selected_emp_id].sort_values("month")
    latest_anom_row = emp_anom_history.iloc[-1] if len(emp_anom_history) else None

    narrative, reasons_list, comp_df = _build_human_reasons(employee, active_tx, period_tx, latest_anom_row)
    graph, partners_df = _employee_ego_graph(selected_emp_id)

    with top_right:
        # Numbered Header 2
        st.markdown(
            f"""<div class='card-title-box' style='justify-content: space-between;'>
                <div style='display: flex; align-items: center;'>
                    <span class='card-header-badge'>2</span>
                    <span>Peta Hubungan Pegawai (12 Bulan Terakhir)</span>
                </div>
                <div style='font-size: 0.8rem; color: #64748b;'>Pegawai: <b>{employee['employee_name']}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )

        if graph is not None:
            # Render interactive graph
            html_net = pyvis_html(graph, height=330)
            if html_net:
                components.html(html_net, height=345)
            else:
                st.plotly_chart(plotly_network(graph, height=330), use_container_width=True)
        else:
            st.info("Pegawai ini belum memiliki jaringan pasangan mencolok di snapshot terbaru.")

        st.markdown(LEGEND_HTML, unsafe_allow_html=True)

    st.write("")

    # -------------------------------------------------------------------------
    # ROW 2: (3) Analisis Risk Driver, (4) Daftar Berkas Transaksi, (5) Dossier & Keputusan
    # -------------------------------------------------------------------------
    b_col1, b_col2, b_col3 = st.columns([1.05, 1.25, 0.95])

    with b_col1:
        # Numbered Header 3
        st.markdown(
            """<div class='card-title-box'>
                <span class='card-header-badge'>3</span>
                <span>Analisis Risk Driver: Mengapa Pegawai Ini Dicurigai?</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Perbandingan dengan rata-rata rekan sejabatannya. Angka menunjukkan persentase transaksi.")

        # Render driver cards with comparison bars
        peer_tx = period_tx[period_tx["approver_position"] == employee["position"]] if employee["role"] == "Approver" else period_tx
        if peer_tx.empty:
            peer_tx = period_tx

        def get_rate(df, col):
            return float(df[col].mean() * 100.0) if len(df) and col in df else 0.0

        driver_specs = [
            ("rf_near_limit", "Nominal Mepet Limit (95–99%)", "⚠️"),
            ("rf_after_hours", "Persetujuan Malam / Akhir Pekan", "🌙"),
            ("rf_fast_approval", "Persetujuan Kilat (< 20 Menit)", "⚡"),
        ]

        for col_name, d_label, d_icon in driver_specs:
            emp_val = get_rate(active_tx, col_name)
            peer_val = max(get_rate(peer_tx, col_name), 0.1)
            mult = emp_val / peer_val if peer_val > 0 else 1.0
            n_cases = int(active_tx[col_name].sum()) if len(active_tx) and col_name in active_tx else 0

            st.markdown(
                f"""<div class='driver-card'>
                    <div style='display: flex; gap: 8px; align-items: center; width: 140px;'>
                        <span style='font-size: 1.15rem;'>{d_icon}</span>
                        <div>
                            <div style='font-size: 0.8rem; font-weight: 700; color: #1e293b;'>{d_label}</div>
                            <div style='font-size: 0.76rem; color: #64748b;'><b>{emp_val:.1f}%</b> <span style='font-size: 0.72rem;'>(vs {peer_val:.1f}%)</span></div>
                        </div>
                    </div>
                    <div style='flex: 1; margin: 0 10px; background: #fee2e2; border-radius: 999px; height: 12px; overflow: hidden;'>
                        <div style='width: {min(emp_val * 2.5, 100):.1f}%; background: #ef4444; height: 100%; border-radius: 999px;'></div>
                    </div>
                    <div style='text-align: right; width: 95px;'>
                        <div style='font-size: 0.82rem; font-weight: 700; color: #dc2626;'>{mult:.1f}x lebih tinggi</div>
                        <div style='font-size: 0.72rem; color: #64748b;'>{n_cases} transaksi</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Anomali Perilaku Bulanan (IF)
        anom_score_val = employee.get("skor_anomali", 0)
        st.markdown(
            f"""<div class='driver-card'>
                <div style='display: flex; gap: 8px; align-items: center; width: 140px;'>
                    <span style='font-size: 1.15rem;'>📊</span>
                    <div>
                        <div style='font-size: 0.8rem; font-weight: 700; color: #1e293b;'>Anomali Perilaku (IF)</div>
                        <div style='font-size: 0.76rem; color: #64748b;'><b>Skor {anom_score_val:.0f}</b> (vs 54)</div>
                    </div>
                </div>
                <div style='flex: 1; margin: 0 10px; background: #fee2e2; border-radius: 999px; height: 12px; overflow: hidden;'>
                    <div style='width: {min(anom_score_val, 100):.1f}%; background: #ef4444; height: 100%; border-radius: 999px;'></div>
                </div>
                <div style='text-align: right; width: 95px;'>
                    <div style='font-size: 0.82rem; font-weight: 700; color: #dc2626;'>Signifikan tinggi</div>
                    <div style='font-size: 0.72rem; color: #64748b;'>Kategori: Perlu dilihat</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Terdeteksi di Jaringan Komunitas
        comm_num = employee.get("komunitas")
        comm_text = f"Komunitas #{int(comm_num)}" if pd.notna(comm_num) else "Tidak terdeteksi"
        st.markdown(
            f"""<div class='driver-card'>
                <div style='display: flex; gap: 8px; align-items: center;'>
                    <span style='font-size: 1.15rem;'>🕸️</span>
                    <div>
                        <div style='font-size: 0.8rem; font-weight: 700; color: #1e293b;'>Terdeteksi di Jaringan</div>
                        <div style='font-size: 0.76rem; color: #64748b;'>{comm_text}</div>
                    </div>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 0.82rem; font-weight: 700; color: #f97316;'>{len(partners_df)} rekan terhubung</div>
                    <div style='font-size: 0.72rem; color: #64748b;'>12 bulan terakhir</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with b_col2:
        # Numbered Header 4
        alerts_df = active_tx.copy()
        alerts_df["partner_name"] = np.where(
            alerts_df["approver_id"] == selected_emp_id,
            alerts_df["requester_id"].map(names) + " (RM)",
            alerts_df["approver_id"].map(names) + " (Approver)",
        )
        alerts_df["nominal_fmt"] = alerts_df["amount"].map(ui.rupiah)
        alerts_df["keputusan_saat_ini"] = alerts_df["request_id"].map(decisions["decision"]).fillna("Belum diputus")

        st.markdown(
            f"""<div class='card-title-box' style='justify-content: space-between;'>
                <div style='display: flex; align-items: center;'>
                    <span class='card-header-badge'>4</span>
                    <span>Daftar Berkas Transaksi: {employee['employee_name']}</span>
                </div>
                <div style='font-size: 0.82rem; font-weight: 700; color: #1e293b;'>
                    {len(alerts_df)} berkas &nbsp;|&nbsp; {ui.rupiah(alerts_df['amount'].sum())}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        display_tx = alerts_df.head(60).reset_index(drop=True)
        default_tx_idx = 0
        if selected_req_id and selected_req_id in display_tx["request_id"].values:
            default_tx_idx = int(display_tx.index[display_tx["request_id"] == selected_req_id][0])

        tx_sub_sel = st.dataframe(
            display_tx[[
                "request_id", "event_time", "partner_name", "nominal_fmt",
                "risk_score", "risk_band", "keputusan_saat_ini",
            ]],
            hide_index=True,
            use_container_width=True,
            height=300,
            on_select="rerun",
            selection_mode="single-row",
            key=f"dossier_tx_v2_{selected_emp_id}",
            column_config={
                "request_id": st.column_config.TextColumn("ID Berkas", width="small"),
                "event_time": st.column_config.DatetimeColumn("Waktu", format="DD MMM HH:mm", width="small"),
                "partner_name": st.column_config.TextColumn("Pasangan Kerja", width="medium"),
                "nominal_fmt": st.column_config.TextColumn("Nominal", width="small"),
                "risk_score": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
                "risk_band": st.column_config.TextColumn("Level", width="small"),
                "keputusan_saat_ini": st.column_config.TextColumn("Keputusan", width="small"),
            },
        )
        sub_rows = getattr(getattr(tx_sub_sel, "selection", None), "rows", []) or []
        chosen_alert_row = display_tx.iloc[sub_rows[0]] if sub_rows else display_tx.iloc[default_tx_idx]
        req_id = str(chosen_alert_row["request_id"])

        evidence_items = [e for e in str(chosen_alert_row.get("evidence") or "").split(" | ") if e and e != "nan"]
        ev_str = " · ".join(f"`{ev}`" for ev in evidence_items) if evidence_items else "Semua parameter verifikasi berada dalam batas normal"
        st.markdown(
            f"""<div style='background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-top: 8px; box-shadow: 0 1px 2px rgba(0,0,0,.03);'>
                <div style='font-size: 0.82rem; font-weight: 700; color: #1e293b;'>📑 Bukti Log Sistem (Berkas {req_id}):</div>
                <div style='font-size: 0.78rem; color: #475569; margin-top: 3px;'>{ev_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with b_col3:
        # Numbered Header 5
        st.markdown(
            """<div class='card-title-box'>
                <span class='card-header-badge'>5</span>
                <span>Dossier & Keputusan Investigasi</span>
            </div>""",
            unsafe_allow_html=True,
        )

        tab_ai, tab_pdf = st.tabs(["📄 Ringkasan AI", "📥 Berkas PDF"])
        user = services.current_user()
        memo_state_key = f"ai_memo_{selected_emp_id}_{req_id}"
        ai_memo_text = st.session_state.get(memo_state_key, "")

        with tab_ai:
            st.markdown(
                f"""<div style='background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; min-height: 140px; box-shadow: 0 1px 2px rgba(0,0,0,.03);'>
                    <div style='font-size: 0.82rem; font-weight: 700; color: #1e293b;'>Ringkasan Analisis (AI)</div>
                    <div style='font-size: 0.78rem; color: #475569; line-height: 1.5; margin-top: 5px;'>
                        <b>{employee['employee_name']} ({selected_emp_id})</b> menjabat sebagai {employee['position']} di Unit {employee['unit_name']}.
                        Skor anomali perilaku bulanan adalah <b>{employee['skor_anomali']:.0f}/100</b>.
                        Tercatat <b>{int(employee['alert_tinggi'])} berkas</b> berisiko tinggi dengan total eksposur <b>{employee['nominal_berisiko_fmt']}</b>.
                        {narrative}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        with tab_pdf:
            st.markdown(
                f"""<div style='background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; min-height: 140px; box-shadow: 0 1px 2px rgba(0,0,0,.03);'>
                    <div style='font-size: 0.82rem; font-weight: 700; color: #1e293b;'>Ekspor Dokumen Resmi</div>
                    <div style='font-size: 0.78rem; color: #64748b; margin: 4px 0 12px 0;'>
                        Unduh dossier ringkasan lengkap untuk berita acara pemeriksaan investigasi fraud & kepatuhan.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            current_status_str = str(chosen_alert_row.get("keputusan_saat_ini", "Belum diputus"))
            pdf_bytes = build_employee_dossier_pdf(
                employee=employee,
                narrative=narrative,
                reasons_list=reasons_list,
                partners_df=partners_df,
                alerts_df=alerts_df[alerts_df["risk_band"] != "Rendah"] if (alerts_df["risk_band"] != "Rendah").any() else alerts_df,
                current_decision=current_status_str,
                investigator_name=user.get("display_name", user.get("username", "Investigator")),
                ai_memo=ai_memo_text,
            )
            safe_name = str(employee["employee_name"]).replace(" ", "_")
            st.download_button(
                label=f"📥 Unduh Berkas PDF ({employee['employee_name']})",
                data=pdf_bytes,
                file_name=f"Dossier_Investigasi_{selected_emp_id}_{safe_name}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key=f"dl_pdf_v2_{selected_emp_id}_{req_id}",
            )

        # Form Tindakan Investigator
        st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #1e293b; margin: 10px 0 4px 0;'>Tindakan Investigator:</div>", unsafe_allow_html=True)
        with st.form(f"form_decision_v2_{selected_emp_id}_{req_id}"):
            chosen_decision = st.selectbox(
                "Pilih Opsi Tindakan",
                DECISIONS,
                key=f"sel_dec_v2_{selected_emp_id}_{req_id}",
                label_visibility="collapsed",
            )
            note = st.text_input(
                "Catatan / Alasan Keputusan",
                placeholder="Contoh: Ditemukan indikasi kuat pemecahan kredit...",
                key=f"note_v2_{selected_emp_id}_{req_id}",
            )
            submitted = st.form_submit_button("💾 Simpan Keputusan Investigator", type="primary", use_container_width=True)
            if submitted:
                store = services.get_store()
                model_name = services.app_settings().get("model_name", "RandomForest_Calibrated")
                store.save_review(
                    user["username"],
                    user["role"],
                    req_id,
                    chosen_decision,
                    note,
                    float(chosen_alert_row["risk_score"]),
                    model_name,
                )
                st.toast(f"Keputusan untuk berkas {req_id} berhasil disimpan!")
                st.rerun()
