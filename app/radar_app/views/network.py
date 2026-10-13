"""Jaringan Risiko: komunitas dari graph analytics, anggota (nama, jabatan, unit), dan graph interaktif."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from ..ui import components as ui
from ..ui import services
from ..ui.network import LEGEND_HTML, build_graph, plotly_network, pyvis_html


def latest_anomaly_scores() -> pd.Series:
    anomaly = services.table("approver_anomaly")
    if anomaly.empty:
        return pd.Series(dtype=float)
    return anomaly[anomaly["month"] == anomaly["month"].max()].set_index("approver_id")["if_score"]


def show_graph(graph, title: str, height: int = 540) -> None:
    """Graph interaktif PyVis (cadangan: Plotly) beserta legenda."""
    st.markdown(LEGEND_HTML, unsafe_allow_html=True)
    html = pyvis_html(graph, height=height)
    if html:
        components.html(html, height=height + 20)
    else:
        st.plotly_chart(plotly_network(graph, title), use_container_width=True)


def community_graph(community_id: int, include_context: bool = False):
    """Graph satu komunitas risiko; opsional ditambah hubungan normal anggotanya (konteks)."""
    edges, members = services.table("community_edges"), services.table("community_members")
    people = set(members.loc[members["community_id"] == community_id, "employee_id"])
    community_edges = edges[(edges["community_a"] == community_id) & (edges["community_b"] == community_id)]
    frame_edges = community_edges[["node_a", "node_b", "weight", "kind", "redflag", "dependency", "n_tx", "z", "shared"]]
    if include_context:
        pairs = services.table("pair_network")
        normal = pairs[(pairs["requester_id"].isin(people) | pairs["approver_id"].isin(people))
                       & ~pairs["risky"].astype(str).str.lower().eq("true")].nlargest(25, "n")
        frame_edges = pd.concat([frame_edges, pd.DataFrame({"node_a": normal["requester_id"], "node_b": normal["approver_id"],
                                                             "weight": 0.5, "kind": "RA", "redflag": False, "dependency": False,
                                                             "n_tx": normal["n"], "z": normal["z"]})], ignore_index=True)
    return build_graph(frame_edges, services.table("employees"), highlight=people, node_scores=latest_anomaly_scores().to_dict())


def render() -> None:
    user = services.current_user()
    ui.page_header("Jaringan Risiko", "Lingkaran pegawai yang hubungannya menyimpang (graph analytics, tanpa label, 12 bulan terakhir)")
    communities, members = services.table("communities"), services.table("community_members")
    history = services.table("community_history")
    if communities.empty:
        ui.empty_state("Belum ada komunitas risiko.")
        return
    search_column, priority_column, members_column = st.columns([2, 1.5, 1.2])
    unit_query = search_column.text_input("🔎 Cari unit / approver utama", key="net_unit", placeholder="mis. Bekasi atau E00292")
    priorities = priority_column.multiselect("Prioritas", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang", "Rendah"], key="net_prio")
    min_members = members_column.slider("Anggota minimal", 3, int(max(communities["anggota"].max(), 3)), 3, key="net_min")
    frame = communities[communities["prioritas"].isin(priorities) & (communities["anggota"] >= min_members)]
    if unit_query:
        frame = frame[frame["unit"].str.contains(unit_query, case=False, na=False, regex=False)
                      | frame["approver_utama"].str.contains(unit_query, case=False, na=False, regex=False)]
    if frame.empty:
        st.info("Tidak ada komunitas yang cocok dengan filter.")
        return
    avg_months_float = frame["aktif_bulan"].mean() if len(frame) else 0
    m_int = int(avg_months_float)
    d_int = int(round((avg_months_float - m_int) * 30))
    if m_int > 0 and d_int > 0:
        val_aktif = f"{m_int} bulan {d_int} hari"
    elif m_int > 0:
        val_aktif = f"{m_int} bulan"
    elif d_int > 0:
        val_aktif = f"{d_int} hari"
    else:
        val_aktif = "-"

    ui.kpi_row([("Komunitas risiko", ui.number(len(frame)), "≥ 3 orang, snapshot terbaru", ui.ORANGE),
                ("Pegawai terlibat", ui.number(frame["anggota"].sum()), "total seluruh klaster terpilih", ui.BLUE),
                ("Prioritas Tinggi", ui.number((frame["prioritas"] == "Tinggi").sum()), "kuartil skor teratas", ui.RED),
                ("Rata-rata aktif", val_aktif, "dari 12 snapshot riwayat terakhir", ui.GREEN)])
    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Daftar Komunitas Terdeteksi**")
        selection = st.dataframe(
            frame[["community_id", "skor_komunitas", "prioritas", "anggota", "approver_utama", "unit", "edge_red_flag",
                   "edge_ketergantungan", "edge_debitur_bersama", "aktif_bulan"]],
            hide_index=True, use_container_width=True, height=400, on_select="rerun", selection_mode="single-row", key="net_table",
            column_config={"skor_komunitas": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
                           "community_id": "ID", "prioritas": "Prioritas", "anggota": "Anggota",
                           "approver_utama": "Approver Utama", "unit": "Unit Kerja",
                           "edge_red_flag": "Red Flag", "edge_ketergantungan": "Ketergantungan",
                           "edge_debitur_bersama": "Debitur Bersama", "aktif_bulan": "Aktif (Bln)"})
        st.caption("💡 *Klik salah satu baris di atas untuk memuat rincian jaringan secara otomatis.*")
    with right:
        figure = px.scatter(
            frame,
            x="aktif_bulan",
            y="skor_komunitas",
            size="anggota",
            color="prioritas",
            color_discrete_map={"Tinggi": ui.RED, "Sedang": ui.ORANGE, "Rendah": ui.BLUE},
            hover_name="unit",
            hover_data={
                "community_id": True,
                "skor_komunitas": ":.1f",
                "aktif_bulan": ":.1f",
                "anggota": True,
                "approver_utama": True,
                "prioritas": True,
            },
            labels={
                "aktif_bulan": "Durasi Persistensi (Bulan Aktif dari 12 Snapshot)",
                "skor_komunitas": "Skor Anomali Komunitas (0 - 100)",
                "prioritas": "Derajat Risiko",
                "anggota": "Jumlah Anggota",
            },
        )
        figure.update_traces(
            marker=dict(line=dict(width=1, color="white"), opacity=0.88),
            hovertemplate=(
                "<b>Komunitas %{customdata[0]}</b><br>"
                "Unit Kerja: %{hovertext}<br>"
                "Approver Utama: %{customdata[4]}<br>"
                "Skor Komunitas: %{y:.1f}<br>"
                "Durasi Persistensi: %{x:.1f} bulan<br>"
                "Anggota Terlibat: %{customdata[3]} orang<br>"
                "Prioritas: %{customdata[5]}<extra></extra>"
            ),
        )
        figure.update_layout(
            height=400,
            xaxis=dict(range=[0, 13], dtick=2),
            yaxis=dict(range=[0, max(frame["skor_komunitas"].max() * 1.15, 60)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        # Quadrant reference lines
        figure.add_hline(y=30, line_dash="dot", line_color="#d0d0d0", annotation_text="Ambang Skor Tinggi", annotation_position="top left")
        figure.add_vline(x=6, line_dash="dot", line_color="#d0d0d0", annotation_text="Persisten (≥6 bln)", annotation_position="top right")

        st.plotly_chart(
            ui.style_figure(
                figure,
                "Matriks Prioritas Investigasi Komunitas",
                "Kuadran kanan-atas: Skor risiko tinggi & paling persisten (target investigasi utama)",
                height=400,
            ),
            use_container_width=True,
        )

    rows = getattr(getattr(selection, "selection", None), "rows", []) or []

    community_id_list = frame["community_id"].tolist()
    if rows and rows[0] < len(frame):
        default_cid = int(frame.iloc[rows[0]]["community_id"])
        selected_index = community_id_list.index(default_cid) if default_cid in community_id_list else 0
    else:
        selected_index = 0

    st.markdown("#### 🎯 Analisis Rinci Jaringan & Anggota")
    st.markdown(
        "<div class='small-note' style='margin-bottom: 6px;'>"
        "Pilih ID komunitas yang ingin dianalisis secara mendalam melalui pilihan di bawah atau dengan mengklik baris pada tabel di atas."
        "</div>",
        unsafe_allow_html=True,
    )

    def format_community_option(cid: int) -> str:
        row = frame[frame["community_id"] == cid]
        if not row.empty:
            r = row.iloc[0]
            unit_name = str(r.get("unit", "-"))
            score = float(r.get("skor_komunitas", 0))
            prio = str(r.get("prioritas", "-"))
            return f"Komunitas #{cid} — {unit_name} | Skor {score:.0f} | Prioritas {prio}"
        return f"Komunitas #{cid}"

    community_id = st.selectbox(
        "Pilih Komunitas untuk Ditinjau:",
        options=community_id_list,
        index=selected_index,
        format_func=format_community_option,
        key="net_pick",
    )
    info = communities[communities["community_id"] == community_id].iloc[0]
    group = members[members["community_id"] == community_id].copy()
    group["skor anomali (IF)"] = group["employee_id"].map(latest_anomaly_scores())
    st.divider()
    st.markdown(f"### Komunitas {community_id} &nbsp; {ui.badge('prioritas ' + info['prioritas'], ui.ORANGE if info['prioritas'] == 'Tinggi' else ui.YELLOW)}",
                unsafe_allow_html=True)
    persis_val = float(info['aktif_bulan'])
    persis_m = int(persis_val)
    persis_d = int(round((persis_val - persis_m) * 30))
    if persis_m > 0 and persis_d > 0:
        persis_str = f"{persis_m} bulan {persis_d} hari"
    elif persis_m > 0:
        persis_str = f"{persis_m} bulan"
    else:
        persis_str = f"{persis_d} hari"

    ui.kpi_row([("Skor komunitas", f"{info['skor_komunitas']:.0f}/100", "relatif thd komunitas terkuat 12 bln", ui.ORANGE),
                ("Anggota", ui.number(info["anggota"]), f"{info['n_approver']} approver · {info['n_requester']} requester", ui.BLUE),
                ("Hubungan berisiko", ui.number(info["edge_red_flag"] + info["edge_ketergantungan"]),
                 f"{info['edge_red_flag']} red flag · {info['edge_ketergantungan']} ketergantungan", ui.PURPLE),
                ("Debitur bersama", ui.number(info["edge_debitur_bersama"]), "pasangan requester berbagi debitur", ui.GREEN),
                ("Persistensi", persis_str, "dari 12 snapshot riwayat terakhir", ui.YELLOW)])
    graph_column, members_column = st.columns([1.5, 1])
    with graph_column:
        include_context = st.toggle("Tampilkan juga hubungan normal anggota (12 bulan)", value=False, key="net_context")
        show_graph(community_graph(community_id, include_context), f"Komunitas {community_id}")
    with members_column:
        st.markdown("**Anggota komunitas**")
        st.dataframe(group[["employee_name", "employee_id", "role", "position", "unit", "pagerank", "skor anomali (IF)", "aktif_bulan"]],
                     hide_index=True, use_container_width=True, height=300,
                     column_config={"employee_name": "nama", "pagerank": st.column_config.NumberColumn("sentralitas", format="%.2f"),
                                    "skor anomali (IF)": st.column_config.NumberColumn(format="%.0f"), "aktif_bulan": "aktif (bln)"})
        presence = history[history["employee_id"].isin(group["employee_id"])].groupby("month")["employee_id"].nunique().reset_index()
        figure = px.line(presence, x="month", y="employee_id", markers=True, labels={"employee_id": "anggota terdeteksi", "month": "bulan"})
        st.plotly_chart(ui.style_figure(figure, "Berapa anggota muncul tiap bulan", "12 snapshot terakhir", height=250),
                        use_container_width=True)
    if user["role"] == "investigator" and st.button("Minta agen merangkum komunitas ini", key=f"net_agent_{community_id}"):
        with st.spinner("Agen membaca komunitas…"):
            result = services.get_agent().run(f"Ringkas komunitas {community_id} beserta anggotanya", "investigator", user["username"])
        st.markdown(result["final_answer"])
    st.caption("Komunitas adalah pola statistik dari hubungan transaksi, bukan bukti kolusi. Gunakan sebagai prioritas pemeriksaan.")
