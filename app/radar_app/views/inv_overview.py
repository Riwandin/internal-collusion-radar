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
from ..ui import pillars, services
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

    # Urutan prioritas meja kerja (bukan skor model): backlog dulu, lalu alert Tinggi, radar IF, dan skor transaksi tertinggi
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
            f"Pada radar perilaku bulanan (Pilar 2 · Isolation Forest), yang bersangkutan memiliki Skor Anomali {anom_score:.0f}/100 "
            f"(kategori: {employee.get('kategori_anomali', '-')})."
        )
    if n_high > 0:
        parts.append(
            f"Pada penilaian transaksi (Risk Engine 4 pilar), terdapat {n_high} berkas berisiko TINGGI "
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
    ui.page_header(
        "Meja Investigasi 360°",
        "Pilih 1 pegawai atau 1 alert transaksi untuk melihat analisis kejanggalan bahasa manusia, jaringan komplotan, unduh PDF, dan simpan keputusan.",
    )
    transactions = services.table("transactions_scored")
    if transactions.empty:
        ui.empty_state("Data belum tersedia. Jalankan notebook terlebih dahulu.")
        return

    settings = services.app_settings()
    period_tx = services.period_filter(transactions, "inv_workspace", default_year=2026)
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

    # 1. BARIS KPI UTAMA
    ui.kpi_row([
        ("Alert Risiko Tinggi", ui.number(len(high_tx)), f"Eksposur {ui.rupiah(high_tx['amount'].sum())}", ui.ORANGE),
        ("Belum Diputus (Backlog)", ui.number(len(open_high)), "Wajib ditindaklanjuti investigator", ui.RED),
        ("Pejabat Perlu Dilihat (IF)", ui.number(len(flagged_approvers)), f"Skor anomali ≥ 95 · {latest_month}", ui.PURPLE),
        ("Sudah Diputus", ui.number(len(high_tx) - len(open_high)), "Alert Tinggi telah direview", ui.GREEN),
        ("Komunitas Kolusi", ui.number(len(services.table("communities"))), "Lingkaran terdeteksi Graph", ui.BLUE),
    ])

    st.write("")

    # 2. PILIHAN MODE ANTREAN & FILTER CEPAT
    emp_queue = _build_employee_queue(period_tx, decisions)
    names = services.name_map()

    with st.container(border=True):
        col_mode, col_search, col_role, col_status = st.columns([1.4, 1.2, 0.9, 1.0])
        queue_mode = col_mode.radio(
            "Tampilkan Antrean Berdasarkan",
            ["👤 Per Pegawai Mencurigakan", "📄 Per Berkas Alert Transaksi"],
            horizontal=True,
            key="ws_queue_mode",
        )
        search_q = col_search.text_input(
            "Cari Nama / ID Pegawai / ID Berkas",
            placeholder="Ketik mis. Nadia, E00391, atau R011...",
            key="ws_search",
        )
        role_filter = col_role.selectbox(
            "Peran Pegawai",
            ["Semua Peran", "Approver (Pemutus)", "Requester (RM/Pengaju)"],
            key="ws_role",
        )
        status_filter = col_status.selectbox(
            "Status Keputusan",
            ["Semua Status", "Belum Diputus Saja", "Sudah Diputus Saja"],
            key="ws_status",
        )

        selected_emp_id: str | None = None
        selected_req_id: str | None = None

        if queue_mode.startswith("👤"):
            filtered_emp = emp_queue[
                (emp_queue["alert_tinggi"] > 0)
                | (emp_queue["skor_anomali"].fillna(0) >= 80)
                | (emp_queue["komunitas"].notna())
            ].copy()
            if role_filter.startswith("Approver"):
                filtered_emp = filtered_emp[filtered_emp["role"] == "Approver"]
            elif role_filter.startswith("Requester"):
                filtered_emp = filtered_emp[filtered_emp["role"] == "Requester"]

            if status_filter == "Belum Diputus Saja":
                filtered_emp = filtered_emp[filtered_emp["belum_diputus"] > 0]
            elif status_filter == "Sudah Diputus Saja":
                filtered_emp = filtered_emp[(filtered_emp["alert_tinggi"] > 0) & (filtered_emp["belum_diputus"] == 0)]

            if search_q.strip():
                q = search_q.strip()
                filtered_emp = filtered_emp[
                    filtered_emp["employee_name"].str.contains(q, case=False, na=False)
                    | filtered_emp["employee_id"].str.contains(q, case=False, na=False)
                    | filtered_emp["unit_name"].str.contains(q, case=False, na=False)
                ]

            st.caption(
                f"Menampilkan **{len(filtered_emp):,} pegawai prioritas**. "
                "👉 **Klik satu baris pegawai di tabel bawah** untuk membuka Dossier Investigasi 360° & cetak PDF."
            )
            emp_selection = st.dataframe(
                filtered_emp[[
                    "employee_id", "employee_name", "role", "position", "unit_name",
                    "skor_anomali", "alert_tinggi", "belum_diputus", "nominal_berisiko_fmt",
                    "komunitas", "status_penanganan",
                ]],
                hide_index=True,
                use_container_width=True,
                height=290,
                on_select="rerun",
                selection_mode="single-row",
                key="ws_emp_table",
                column_config={
                    "employee_id": "ID",
                    "employee_name": "Nama Pegawai",
                    "role": "Peran",
                    "position": "Jabatan",
                    "unit_name": "Cabang/Unit",
                    "skor_anomali": st.column_config.ProgressColumn("Skor Perilaku (IF)", min_value=0, max_value=100, format="%.0f"),
                    "alert_tinggi": "Alert Tinggi",
                    "belum_diputus": "Belum Diputus",
                    "nominal_berisiko_fmt": "Eksposur Berisiko",
                    "komunitas": st.column_config.NumberColumn("ID Komunitas", format="#%d"),
                    "status_penanganan": "Status",
                },
            )
            if filtered_emp.empty:
                st.info("Tidak ada pegawai yang cocok dengan filter di atas.")
                return
            sel_rows = getattr(getattr(emp_selection, "selection", None), "rows", []) or []
            selected_emp_id = str(filtered_emp.iloc[sel_rows[0]]["employee_id"]) if sel_rows else str(filtered_emp.iloc[0]["employee_id"])

        else:
            # Mode per berkas transaksi
            tx_queue = period_tx[period_tx["risk_band"].isin(["Tinggi", "Sedang"])].copy()
            tx_queue["requester_name"] = tx_queue["requester_id"].map(names)
            tx_queue["approver_name"] = tx_queue["approver_id"].map(names)
            tx_queue["sudah_diputus"] = tx_queue["request_id"].isin(decisions.index)
            tx_queue["status_label"] = np.where(
                tx_queue["sudah_diputus"],
                "✅ " + tx_queue["request_id"].map(decisions["decision"]).fillna("Diputus"),
                "🔴 Belum diputus",
            )
            tx_queue["nominal_fmt"] = tx_queue["amount"].map(ui.rupiah)
            tx_queue["alasan_utama"] = tx_queue["top_reasons"].fillna("").str.split(" | ").str[0]
            contribution_columns = [f"c_{p}" for p in ui.PILLARS]
            tx_queue["pilar_utama"] = (tx_queue[contribution_columns].idxmax(axis=1).str[2:].map(ui.PILLAR_SHORT)
                                       if set(contribution_columns) <= set(tx_queue.columns) else "-")

            if status_filter == "Belum Diputus Saja":
                tx_queue = tx_queue[~tx_queue["sudah_diputus"]]
            elif status_filter == "Sudah Diputus Saja":
                tx_queue = tx_queue[tx_queue["sudah_diputus"]]

            if search_q.strip():
                q = search_q.strip()
                tx_queue = tx_queue[
                    tx_queue["request_id"].str.contains(q, case=False, na=False)
                    | tx_queue["requester_name"].str.contains(q, case=False, na=False)
                    | tx_queue["approver_name"].str.contains(q, case=False, na=False)
                    | tx_queue["requester_unit"].str.contains(q, case=False, na=False)
                ]

            tx_queue = tx_queue.sort_values(["risk_score", "event_time"], ascending=False).head(400)
            st.caption(
                f"Menampilkan **{len(tx_queue):,} berkas alert**. "
                "👉 **Klik satu baris transaksi** untuk melihat analisis pegawai & berkasnya di bawah."
            )
            tx_selection = st.dataframe(
                tx_queue[[
                    "request_id", "event_time", "approver_name", "requester_name", "requester_unit",
                    "nominal_fmt", "risk_score", "risk_band", "pilar_utama", "alasan_utama", "status_label",
                ]],
                hide_index=True,
                use_container_width=True,
                height=290,
                on_select="rerun",
                selection_mode="single-row",
                key="ws_tx_table",
                column_config={
                    "request_id": "No. Berkas",
                    "event_time": st.column_config.DatetimeColumn("Waktu Approval", format="DD MMM YYYY HH:mm"),
                    "approver_name": "Approver (Pemutus)",
                    "requester_name": "Requester (RM)",
                    "requester_unit": "Unit",
                    "nominal_fmt": "Nominal",
                    "risk_score": st.column_config.ProgressColumn("Skor Risiko", min_value=0, max_value=100, format="%.0f"),
                    "risk_band": "Level",
                    "pilar_utama": "Pilar Penyebab Utama",
                    "alasan_utama": "Alasan Utama",
                    "status_label": "Status Keputusan",
                },
            )
            if tx_queue.empty:
                st.info("Tidak ada berkas transaksi yang cocok dengan filter.")
                return
            sel_rows = getattr(getattr(tx_selection, "selection", None), "rows", []) or []
            chosen_tx = tx_queue.iloc[sel_rows[0]] if sel_rows else tx_queue.iloc[0]
            selected_req_id = str(chosen_tx["request_id"])
            focus_role = st.radio(
                "Sorot Profil Pegawai pada Transaksi Ini:",
                [
                    f"Approver: {chosen_tx['approver_name']} ({chosen_tx['approver_id']})",
                    f"Requester: {chosen_tx['requester_name']} ({chosen_tx['requester_id']})",
                ],
                horizontal=True,
                key="ws_tx_focus_emp",
            )
            selected_emp_id = str(chosen_tx["approver_id"]) if focus_role.startswith("Approver") else str(chosen_tx["requester_id"])

    # =========================================================================
    # 3. DOSSIER INVESTIGASI 360° UNTUK PEGAWAI TERPILIH
    # =========================================================================
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

    # Banner Dossier 360°
    anom_badge = (
        ui.badge(f"Skor Perilaku (IF): {employee['skor_anomali']:.0f}/100 ({employee['kategori_anomali']})", ui.ORANGE)
        if pd.notna(employee["skor_anomali"])
        else ui.badge("Peran: Requester (RM/CA)", ui.BLUE)
    )
    alert_badge = ui.badge(f"{int(employee['alert_tinggi'])} Alert Transaksi Tinggi", ui.RED if employee["alert_tinggi"] > 0 else ui.GREEN)
    comm_badge = (
        ui.badge(f"Anggota Komunitas Risiko #{int(employee['komunitas'])}", ui.PURPLE)
        if pd.notna(employee["komunitas"])
        else ""
    )

    st.markdown(
        f"""
        <div class='dossier-banner'>
            <div class='dossier-title'>
                📂 Dossier Investigasi 360°: {html.escape(str(employee['employee_name']))} ({html.escape(str(selected_emp_id))})
                &nbsp; {anom_badge} {alert_badge} {comm_badge}
            </div>
            <div class='dossier-summary'>
                <b>Ringkasan Analisis (Bahasa Manusia):</b> {html.escape(narrative)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2 Kolom Inti: Kiri = Kenapa Dicurigai (Explainable), Kanan = Jaringan Komplotan
    col_why, col_net = st.columns([1.15, 1.15])

    with col_why:
        st.markdown("#### 🔍 1. Mengapa Pegawai Ini Dicurigai?")
        st.caption("Terjemahan langsung dari sinyal model ML & Rules dibandingkan dengan rata-rata rekan sejabatannya.")
        if reasons_list:
            for r_title, r_detail in reasons_list:
                st.markdown(
                    f"<div class='reason-card'><div class='reason-title'>⚠️ {html.escape(r_title)}</div>"
                    f"<div class='reason-detail'>{html.escape(r_detail)}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ Tidak ditemukan indikator perilaku yang menyimpang secara mencolok dibanding rekan sejabatan.")

        if not comp_df.empty:
            fig_comp = go.Figure()
            fig_comp.add_bar(
                y=comp_df["Indikator"],
                x=comp_df["Rata-rata Rekan (%)"],
                orientation="h",
                name="Rata-rata Rekan Sejabatan",
                marker_color=ui.MUTED,
            )
            fig_comp.add_bar(
                y=comp_df["Indikator"],
                x=comp_df["Pegawai Ini (%)"],
                orientation="h",
                name=str(employee["employee_name"]),
                marker_color=ui.ORANGE,
            )
            fig_comp.update_layout(
                barmode="group",
                xaxis_title="% dari total transaksi",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(
                ui.style_figure(
                    fig_comp,
                    "Perbandingan Nyata: Pegawai Ini vs Rata-rata Rekan",
                    "Semakin panjang bar oranye dibanding abu-abu, semakin menyimpang polanya",
                    height=300,
                ),
                use_container_width=True,
            )

    with col_net:
        st.markdown("#### 🕸️ 2. Siapa Lingkaran / Komplotan Kerjanya?")
        st.caption("Peta hubungan transaksi 12 bulan terakhir. Titik bertepi oranye adalah pegawai yang sedang diperiksa.")
        if graph is not None:
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)
            html_net = pyvis_html(graph, height=340)
            if html_net:
                components.html(html_net, height=355)
            else:
                st.plotly_chart(plotly_network(graph, height=350), use_container_width=True)
        else:
            st.info("Pegawai ini belum memiliki jaringan pasangan mencolok di snapshot terbaru.")

        if not partners_df.empty:
            st.markdown("**Pasangan Kerja Paling Sering / Berisiko (12 Bulan Terakhir)**")
            st.dataframe(
                partners_df[["pasangan", "n", "k", "dependency_lift", "risky"]].head(5),
                hide_index=True,
                use_container_width=True,
                height=185,
                column_config={
                    "pasangan": "Nama Pasangan (RM/Approver)",
                    "n": "Total Tx Bersama",
                    "k": "Tx Red Flag",
                    "dependency_lift": st.column_config.NumberColumn("Kelipatan vs Unit (Lift)", format="%.1fx"),
                    "risky": "Relasi Berisiko?",
                },
            )

    # =========================================================================
    # 4. DAFTAR TRANSAKSI TERKAIT & PENJELASAN SHAP PER BERKAS
    # =========================================================================
    st.divider()
    st.markdown(f"#### 📋 3. Daftar Berkas Transaksi yang Melibatkan {employee['employee_name']}")

    alerts_df = active_tx.copy()
    alerts_df["partner_name"] = np.where(
        alerts_df["approver_id"] == selected_emp_id,
        alerts_df["requester_id"].map(names) + " (RM)",
        alerts_df["approver_id"].map(names) + " (Approver)",
    )
    alerts_df["nominal_fmt"] = alerts_df["amount"].map(ui.rupiah)
    alerts_df["alasan_singkat"] = alerts_df["top_reasons"].fillna("").str.replace(" | ", "; ", regex=False)
    alerts_df["keputusan_saat_ini"] = alerts_df["request_id"].map(decisions["decision"]).fillna("⏳ Belum diputus")

    col_tx_list, col_tx_shap = st.columns([1.25, 1.0])
    with col_tx_list:
        st.caption("Klik satu berkas di tabel bawah untuk membedah skor 4 pilar, alasan, dan bukti detailnya di sebelah kanan:")
        display_tx = alerts_df.head(50).reset_index(drop=True)
        default_tx_idx = 0
        if selected_req_id and selected_req_id in display_tx["request_id"].values:
            default_tx_idx = int(display_tx.index[display_tx["request_id"] == selected_req_id][0])

        tx_sub_sel = st.dataframe(
            display_tx[[
                "request_id", "event_time", "partner_name", "nominal_fmt",
                "risk_score", "risk_band", "keputusan_saat_ini", "alasan_singkat",
            ]],
            hide_index=True,
            use_container_width=True,
            height=270,
            on_select="rerun",
            selection_mode="single-row",
            key=f"dossier_tx_{selected_emp_id}",
            column_config={
                "request_id": "ID Berkas",
                "event_time": st.column_config.DatetimeColumn("Waktu", format="DD MMM YY HH:mm"),
                "partner_name": "Pasangan Kerja",
                "nominal_fmt": "Nominal",
                "risk_score": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
                "risk_band": "Level",
                "keputusan_saat_ini": "Keputusan",
                "alasan_singkat": "Alasan",
            },
        )
        sub_rows = getattr(getattr(tx_sub_sel, "selection", None), "rows", []) or []
        chosen_alert_row = display_tx.iloc[sub_rows[0]] if sub_rows else display_tx.iloc[default_tx_idx]

    with col_tx_shap:
        req_id = str(chosen_alert_row["request_id"])
        st.markdown(
            f"**Bedah Berkas `{req_id}`** &nbsp; {ui.band_badge(str(chosen_alert_row['risk_band']))} "
            f"&nbsp; **Skor: {chosen_alert_row['risk_score']:.0f}/100**",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Debitur: {chosen_alert_row.get('debtor_name_masked', '-')} ({chosen_alert_row['debtor_id']}) · "
            f"{chosen_alert_row['loan_product']} · {ui.rupiah(chosen_alert_row['amount'], short=False)} "
            f"({ui.percent(chosen_alert_row['limit_ratio'])} dari limit wewenang)"
        )
        if pillars.has_pillars(chosen_alert_row):
            pillars.pillar_cards(chosen_alert_row)
            st.plotly_chart(pillars.contribution_waterfall(chosen_alert_row, f"Dari mana skor {chosen_alert_row['risk_score']:.0f} berasal?",
                                                           height=270), use_container_width=True)

        explanations = services.table("alert_explanations")
        reasons_frame = explanations[explanations["request_id"] == req_id]
        if not reasons_frame.empty and "contribution" in reasons_frame:
            st.plotly_chart(pillars.reasons_chart(reasons_frame, f"Alasan di balik skor berkas {req_id}"), use_container_width=True)
        else:
            st.info("Berkas ini berisiko rendah sehingga tidak memiliki uraian alasan.")

        evidence_items = [e for e in str(chosen_alert_row.get("evidence") or "").split(" | ") if e and e != "nan"]
        if evidence_items:
            st.markdown(
                "**Bukti Log Sistem:** " + " · ".join(f"`{ev}`" for ev in evidence_items),
            )
        pillars.graph_evidence_block(req_id, services.table("graph_evidence"))

    # =========================================================================
    # 5. PANEL TINDAKAN: CETAK PDF RINGKASAN & FORM KEPUTUSAN INVESTIGATOR
    # =========================================================================
    st.divider()
    col_pdf, col_decide = st.columns([1.1, 1.2])

    user = services.current_user()
    memo_state_key = f"ai_memo_{selected_emp_id}_{req_id}"
    ai_memo_text = st.session_state.get(memo_state_key, "")

    with col_pdf:
        with st.container(border=True):
            st.markdown("#### 📄 4. Cetak Ringkasan PDF & Bantuan AI")
            st.caption(
                "Unduh laporan ringkasan lengkap (Dossier PDF) berisi profil pegawai, alasan kejanggalan vs rekan, "
                "tabel pasangan komplotan, dan daftar transaksi untuk lampiran berita acara pemeriksaan."
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
                label=f"📥 Unduh PDF Ringkasan Analisis ({employee['employee_name']})",
                data=pdf_bytes,
                file_name=f"Dossier_Investigasi_{selected_emp_id}_{safe_name}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key=f"dl_pdf_{selected_emp_id}_{req_id}",
            )

            if st.button(
                f"🤖 Minta AI Susun Draf Memo & Pertanyaan Klarifikasi ({req_id})",
                use_container_width=True,
                key=f"btn_ai_{selected_emp_id}_{req_id}",
            ):
                with st.spinner("Agen AI sedang merangkum bukti transaksi, jaringan, dan pedoman SOP…"):
                    run_res = services.get_agent().run(
                        f"Siapkan draf memo investigasi untuk transaksi {req_id}",
                        "investigator",
                        user["username"],
                    )
                    memo_draft = (
                        run_res.get("pending_action", {}).get("memo")
                        or run_res.get("final_answer", "")
                    )
                    st.session_state[memo_state_key] = memo_draft
                    st.rerun()

            if ai_memo_text:
                edited_memo = st.text_area(
                    "Draf Memo AI (otomatis ikut tercetak di PDF di atas):",
                    value=ai_memo_text,
                    height=180,
                    key=f"edit_memo_{selected_emp_id}_{req_id}",
                )
                st.session_state[memo_state_key] = edited_memo
                c_ok, c_del = st.columns(2)
                if c_ok.button("✅ Setujui & Arsipkan Memo", key=f"approve_memo_{req_id}", use_container_width=True):
                    services.get_store().approve_agent_action(req_id, "memo_investigasi", edited_memo, user["username"])
                    st.success("Memo resmi disetujui dan tercatat di jejak audit.")
                if c_del.button("🗑️ Tutup Draf", key=f"clear_memo_{req_id}", use_container_width=True):
                    st.session_state.pop(memo_state_key, None)
                    st.rerun()

    with col_decide:
        with st.container(border=True):
            st.markdown("#### ⚖️ 5. Keputusan Investigator")
            st.caption("Tentukan status tindak lanjut untuk kasus/pegawai ini. Setiap keputusan tercatat di log audit dan menjadi umpan balik model.")

            if req_id in decisions.index:
                last_rev = decisions.loc[req_id]
                st.info(
                    f"Status terakhir berkas **{req_id}**: **{last_rev['decision']}** "
                    f"(oleh `{last_rev['username']}` pada `{last_rev['reviewed_at']}`). Catatan: *{last_rev['note'] or '-'}*"
                )

            open_emp_req_ids = alerts_df[
                (alerts_df["risk_band"] == "Tinggi") & (~alerts_df["request_id"].isin(decisions.index))
            ]["request_id"].tolist()

            with st.form(f"form_decision_{selected_emp_id}_{req_id}"):
                chosen_decision = st.radio(
                    "Pilih Opsi Tindakan:",
                    DECISIONS,
                    key=f"rad_dec_{selected_emp_id}_{req_id}",
                )
                apply_scope = st.radio(
                    "Terapkan Keputusan Pada:",
                    [
                        f"Hanya berkas terpilih ({req_id})",
                        f"Semua alert Tinggi milik {employee['employee_name']} yang belum diputus ({len(open_emp_req_ids)} berkas)",
                    ],
                    horizontal=True,
                    key=f"scope_dec_{selected_emp_id}_{req_id}",
                )
                note = st.text_area(
                    "Catatan Bukti / Alasan Keputusan (misal: nomor surat lembur, hasil cek agunan, atau jadwal pemanggilan):",
                    placeholder="Contoh: Ditemukan indikasi kuat pemecahan kredit dan ketergantungan pasangan, eskalasi ke SKAI...",
                    height=85,
                )
                submitted = st.form_submit_button("💾 Simpan Keputusan Investigator", type="primary", use_container_width=True)
                if submitted:
                    store = services.get_store()
                    model_name = settings.get("model_name", "RandomForest_Calibrated")
                    if apply_scope.startswith("Semua") and len(open_emp_req_ids) > 0:
                        for target_rid in open_emp_req_ids:
                            row_score = float(alerts_df.loc[alerts_df["request_id"] == target_rid, "risk_score"].iloc[0])
                            store.save_review(user["username"], user["role"], str(target_rid), chosen_decision, note, row_score, model_name)
                        st.toast(f"Keputusan disimpan untuk {len(open_emp_req_ids)} berkas milik {employee['employee_name']}!")
                    else:
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

    # =========================================================================
    # 6. RIWAYAT KEPUTUSAN & TREN (DI DALAM EXPANDER AGAR RAPI)
    # =========================================================================
    with st.expander("📜 Lihat Riwayat Seluruh Keputusan Investigator & Tren Bulanan"):
        all_reviews = services.get_store().reviews()
        if all_reviews.empty:
            st.caption("Belum ada keputusan investigator yang disimpan.")
        else:
            st.dataframe(
                all_reviews.sort_values("reviewed_at", ascending=False),
                hide_index=True,
                use_container_width=True,
                height=240,
            )
            st.download_button(
                "Unduh Log Keputusan (CSV)",
                all_reviews.to_csv(index=False),
                "riwayat_keputusan_investigator.csv",
                "text/csv",
            )
