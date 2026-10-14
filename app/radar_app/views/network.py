"""Jaringan Risiko: komunitas dari graph analytics, graph heterogen multi-entitas, dan bukti jaringan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from .. import hetero_graph as hg
from ..ui import components as ui
from ..ui import services
from ..ui.d3_network import d3_network_html, hetero_payload, risk_payload
from ..ui.network import (HETERO_LEGEND_HTML, LEGEND_HTML, build_graph, collapse_hetero, hetero_html, plotly_network,
                          pyvis_html)

RELATION_MEANING = {
    "SUBMITS": "pegawai mengajukan pengajuan", "REQUESTED_BY_DEBTOR": "pengajuan untuk debitur",
    "HAS_APPROVAL": "pengajuan punya kejadian approval", "DECIDED_BY": "approval diputus oleh pegawai",
    "BELONGS_TO": "pegawai anggota unit (statis)", "ACCESSES": "pegawai mengakses sistem (Login/View/Export, agregat)",
    "HAS_PERFORMANCE": "pengajuan punya kinerja kredit (monitoring, bukan fitur)"}
EVIDENCE_LABELS = {"risky_pair": "Pasangan berisiko", "persistent_pair": "Hubungan berisiko bertahan",
                   "same_community": "Satu komunitas risiko", "shared_debtor_motif": "Motif debitur bersama",
                   "central_approver": "Approver sentral"}


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
    ui.page_header("Jaringan Risiko", "Lingkaran pegawai yang hubungannya menyimpang dan graph multi-entitas "
                                      "(pegawai · pengajuan · debitur · unit · sistem), tanpa label, 12 bulan terakhir")
    tabs = st.tabs(["🕸️ Komunitas risiko", "🧩 Graph heterogen & bukti jaringan"])
    with tabs[0]:
        render_communities(user)
    with tabs[1]:
        render_hetero(user)


def render_communities(user: dict) -> None:
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
    edges = services.table("community_edges")
    latest_scores = latest_anomaly_scores().to_dict()
    view = st.radio("Tampilan graph", ["Interaktif + panel detail", "Hubungan pegawai (graph risiko)",
                                       "Multi-entitas (pegawai · debitur · unit · sistem)"],
                    horizontal=True, key="net_view")
    if view.startswith("Interaktif"):
        render_d3(edges, group, info, community_id)
        graph_column, members_column = None, st.container()
    else:
        graph_column, members_column = st.columns([1.5, 1])
        with graph_column:
            if view.startswith("Hubungan"):
                render_risk_graph(edges, group, latest_scores, community_id)
            else:
                render_multi_entity(group, community_id)
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


def render_d3(edges: pd.DataFrame, group: pd.DataFrame, info: pd.Series, community_id: int) -> None:
    """Graph D3: dua tampilan dalam satu komponen + panel detail node/garis."""
    community_edges = edges[(edges["community_a"] == community_id) & (edges["community_b"] == community_id)]
    nodes, hetero_edges = services.table("graph_nodes"), services.table("graph_edges")
    hetero = (hetero_payload(collapse_hetero(nodes, hetero_edges, set(group["employee_id"]), services.table("employees")))
              if not nodes.empty and not hetero_edges.empty else {"nodes": [], "edges": []})
    summary = {"id": int(community_id), "unit": str(info["unit"]), "score": float(info["skor_komunitas"]),
               "prio": str(info["prioritas"]), "members": int(info["anggota"]), "active": float(info["aktif_bulan"])}
    components.html(d3_network_html(risk_payload(community_edges, group), hetero, summary, height=540), height=560)
    st.caption("Tombol di panel kanan mengganti tampilan. Klik atau arahkan kursor ke node/garis untuk detail; seret node "
               "untuk menata ulang. Butuh koneksi internet untuk memuat pustaka D3.")


def render_risk_graph(edges: pd.DataFrame, group: pd.DataFrame, latest_scores: pd.Series, community_id: int) -> None:
    include_context = st.toggle("Tampilkan juga hubungan normal anggota (12 bulan)", value=False, key="net_context")
    community_edges = edges[(edges["community_a"] == community_id) & (edges["community_b"] == community_id)]
    frame_edges = community_edges[["node_a", "node_b", "weight", "kind", "redflag", "dependency", "n_tx", "z", "shared"]]
    if include_context:
        pairs = services.table("pair_network")
        people = set(group["employee_id"])
        normal = pairs[(pairs["requester_id"].isin(people) | pairs["approver_id"].isin(people))
                       & ~pairs["risky"].astype(str).str.lower().eq("true")].nlargest(25, "n")
        frame_edges = pd.concat([frame_edges, pd.DataFrame({"node_a": normal["requester_id"], "node_b": normal["approver_id"],
                                                             "weight": 0.5, "kind": "RA", "redflag": False, "dependency": False,
                                                             "n_tx": normal["n"], "z": normal["z"]})], ignore_index=True)
    graph = build_graph(frame_edges, services.table("employees"), highlight=set(group["employee_id"]),
                        node_scores=latest_scores.to_dict())
    st.markdown(LEGEND_HTML, unsafe_allow_html=True)
    html = pyvis_html(graph, height=540)
    if html:
        components.html(html, height=560)
    else:
        st.plotly_chart(plotly_network(graph, f"Komunitas {community_id}"), use_container_width=True)


def render_multi_entity(group: pd.DataFrame, community_id: int) -> None:
    nodes, edges = services.table("graph_nodes"), services.table("graph_edges")
    if nodes.empty or edges.empty:
        ui.empty_state("Tabel graph_nodes/graph_edges belum ada. Jalankan notebook v10.1.")
        return
    shared_only = st.toggle("Hanya debitur yang dipakai ≥ 2 requester atau ≥ 2 approver berbeda", value=True, key="net_shared_only")
    graph = collapse_hetero(nodes, edges, set(group["employee_id"]), services.table("employees"), shared_only=shared_only)
    kinds = pd.Series([d["kind"] for _, d in graph.nodes(data=True)]).value_counts()
    st.caption(f"{kinds.get('Debtor', 0)} debitur · {kinds.get('Unit', 0)} unit · {kinds.get('System', 0)} sistem · "
               f"{sum(1 for *_, d in graph.edges(data=True) if d['relation'] == 'MENGAJUKAN_KE')} pasangan pengaju→pemutus antar anggota. "
               "Node pengajuan & approval dilipat menjadi garis supaya graph terbaca.")
    st.markdown(HETERO_LEGEND_HTML, unsafe_allow_html=True)
    html = hetero_html(graph, height=540)
    if html:
        components.html(html, height=560)
    else:
        st.info("PyVis tidak tersedia; tampilan multi-entitas membutuhkan PyVis.")


def render_hetero(user: dict) -> None:
    settings = services.app_settings().get("graph_hetero", {})
    schema, validation = services.table("graph_schema"), services.table("graph_validation")
    comparison, evidence = services.table("graph_config_comparison"), services.table("graph_evidence")
    if schema.empty:
        ui.empty_state("Tabel graph heterogen belum ada. Jalankan notebook v10.1.")
        return
    checks_ok, checks_total = settings.get("checks_passed", int(validation["lulus"].astype(str).str.lower().eq("true").sum())), \
        settings.get("checks_total", len(validation))
    final = comparison[comparison["final"].astype(str).str.lower().eq("true")].iloc[0] if not comparison.empty else None
    baseline = comparison[comparison["konfigurasi"].str.startswith("A")].iloc[0] if not comparison.empty else None
    ui.kpi_row([("Node", ui.number(settings.get("nodes", schema.loc[schema["jenis"] == "node", "jumlah"].sum())),
                 f"7 tipe · snapshot {settings.get('snapshot_cutoff', '-')}", ui.BLUE),
                ("Edge", ui.number(settings.get("edges", schema.loc[schema["jenis"] == "edge", "jumlah"].sum())), "7 relasi bertipe", ui.PURPLE),
                ("Uji integritas", f"{checks_ok}/{checks_total}", "namespace, skema, waktu, duplikat, PII", ui.GREEN),
                ("Alarm palsu vs tanpa graph", f"{int(baseline['FP']):,} → {int(final['FP']):,}" if final is not None else "-",
                 "test 2026, konfigurasi final", ui.ORANGE),
                ("Kolusi lolos vs tanpa graph", f"{int(baseline['FN']):,} → {int(final['FN']):,}" if final is not None else "-",
                 "test 2026", ui.RED)])
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Skema graph** (jumlah di snapshot terbaru)")
        node_count = schema[schema["jenis"] == "node"].set_index("nama")["jumlah"]
        edge_rows = schema[schema["jenis"] == "edge"]
        lines = ['digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Inter", fontsize=10];',
                 'edge [fontname="Inter", fontsize=9, color="#7a7a7a"];']
        lines += [f'"{name}" [label="{name}\\n{int(count):,}"];' for name, count in node_count.items()]
        relation_ends = {"SUBMITS": ("Employee", "Request"), "REQUESTED_BY_DEBTOR": ("Request", "Debtor"),
                         "HAS_APPROVAL": ("Request", "Approval"), "DECIDED_BY": ("Approval", "Employee"),
                         "BELONGS_TO": ("Employee", "Unit"), "ACCESSES": ("Employee", "System"),
                         "HAS_PERFORMANCE": ("Request", "LoanPerformance")}
        for row in edge_rows.itertuples(index=False):
            source, target = relation_ends.get(row.nama, ("?", "?"))
            lines.append(f'"{source}" -> "{target}" [label="{row.nama}\\n{int(row.jumlah):,}"];')
        st.graphviz_chart("\n".join(lines) + "}", use_container_width=True)
    with right:
        st.markdown("**Arti setiap relasi**")
        st.dataframe(pd.DataFrame({"relasi": list(RELATION_MEANING), "arti": list(RELATION_MEANING.values())}),
                     hide_index=True, use_container_width=True, height=290)
        st.caption("Tidak dibuat: relasi akses→transaksi (log tidak punya request_id), rekening/agunan/alamat (kolom tidak ada), "
                   "dan kolektibilitas (snapshot akhir data = informasi masa depan).")
    st.divider()
    st.markdown("#### Apakah graph membantu? Perbandingan pada data test 2026")
    if not comparison.empty:
        view = comparison.copy()
        view["konfigurasi"] = np.where(view["final"].astype(str).str.lower().eq("true"), view["konfigurasi"] + "  ✅ dipakai",
                                       view["konfigurasi"])
        columns = ["konfigurasi", "PR-AUC test", "recall", "precision", "FP", "FN", "precision@10%"]
        if user["role"] == "investigator":
            columns.insert(2, "Δ PR-AUC vs A")
        st.dataframe(view[columns], hide_index=True, use_container_width=True,
                     column_config={"PR-AUC test": st.column_config.NumberColumn(format="%.3f"),
                                    "Δ PR-AUC vs A": st.column_config.NumberColumn(format="%+.3f"),
                                    "recall": st.column_config.NumberColumn(format="%.2f"),
                                    "precision": st.column_config.NumberColumn(format="%.2f"),
                                    "precision@10%": st.column_config.NumberColumn(format="%.2f"),
                                    "FP": st.column_config.NumberColumn("alarm palsu", format="%d"),
                                    "FN": st.column_config.NumberColumn("kolusi lolos", format="%d")})
        st.caption("A = tanpa graph · B0 = graph lama (pegawai saja) · B = + fitur metapath graph heterogen yang lolos uji · "
                   "C = B + skor bukti jaringan di penggabung. Semua dipilih di validation 2025, test dibuka sekali.")
    st.divider()
    st.markdown("#### Bukti jaringan pada alert (Tinggi & Sedang, 2025–2026)")
    if evidence.empty:
        ui.empty_state("Belum ada bukti jaringan pada alert.")
    else:
        summary = (evidence.groupby("bukti").agg(alert=("request_id", "nunique"), alasan=("alasan", "first"),
                                                 batasan=("batasan_interpretasi", "first")).reset_index())
        summary["bukti"] = summary["bukti"].map(EVIDENCE_LABELS).fillna(summary["bukti"])
        st.dataframe(summary.sort_values("alert", ascending=False), hide_index=True, use_container_width=True,
                     column_config={"alert": st.column_config.NumberColumn("alert dengan bukti ini", format="%d"),
                                    "batasan": "batasan interpretasi"})
        st.caption("Setiap bukti adalah keterhubungan yang perlu dicek, bukan bukti kesalahan. Baca kolom batasan sebelum "
                   "menyimpulkan; konfirmasi selalu lewat dokumen dan wawancara.")
    render_evidence_strength()
    render_stability()
    with st.expander("Uji integritas graph (detail)"):
        st.dataframe(validation, hide_index=True, use_container_width=True)
        access = services.table("graph_access_match")
        if not access.empty:
            st.markdown("**Kecocokan log akses dengan transaksi** (validasi, bukan fitur)")
            access = access.assign(porsi=(access["porsi"] * 100).round(1).astype(str) + "%")
            st.dataframe(access, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False)
def evidence_strength(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ketajaman bukti jaringan di test 2026 (label hanya untuk mengukur)."""
    flags = hg.evidence_flags(test)
    label = test["fraud_label"].astype(int)
    per_evidence = pd.DataFrame({
        "bukti": [EVIDENCE_LABELS.get(c, c) for c in flags.columns], "cakupan": flags.mean().to_numpy(),
        "% kolusi bila ada": [label[flags[c] == 1].mean() for c in flags.columns],
        "recall": [flags.loc[label == 1, c].mean() for c in flags.columns]})
    per_count = (test.assign(n_bukti=flags.sum(axis=1)).groupby("n_bukti")["fraud_label"].agg(["size", "mean"])
                 .rename(columns={"size": "kasus", "mean": "% kolusi"}).reset_index())
    return per_evidence, per_count


def render_evidence_strength() -> None:
    test = services.table("test_features")
    if test.empty or "fraud_label" not in test or not set(hg.EVIDENCE_COLUMNS) <= set(test.columns):
        return
    per_evidence, per_count = evidence_strength(test)
    base_rate = test["fraud_label"].mean()
    st.divider()
    st.markdown("#### Seberapa tajam bukti jaringan? (test 2026, label hanya untuk mengukur)")
    left, right = st.columns([1, 1.1])
    with left:
        st.dataframe(per_evidence, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.2f")
                                    for c in ["cakupan", "% kolusi bila ada", "recall"]})
        st.caption("Cakupan = porsi kasus yang punya bukti ini · % kolusi = precision bukti · recall = porsi kasus kolusi "
                   "yang punya bukti ini.")
    with right:
        frame = per_count.assign(label=per_count["n_bukti"].astype(str) + " bukti")
        figure = px.bar(frame, x="label", y="% kolusi", text=frame["% kolusi"].map(lambda v: f"{v:.0%}"),
                        hover_data={"kasus": True, "label": False}, labels={"label": "", "% kolusi": "% kolusi"})
        figure.update_traces(marker_color=[ui.ORANGE if n >= 2 else ui.BLUE for n in frame["n_bukti"]], textposition="outside")
        figure.add_hline(y=base_rate, line_dash="dot", line_color=ui.MUTED, annotation_text=f"rata-rata {base_rate:.0%}")
        figure.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.12])
        st.plotly_chart(ui.style_figure(figure, "Jumlah bukti per kasus vs % kolusi",
                                        " · ".join(f"{int(r.n_bukti)} bukti: {int(r.kasus):,} kasus" for r in per_count.itertuples()),
                                        height=330), use_container_width=True)
    st.caption("Keterhubungan bukan bukti kesalahan: kasus tanpa bukti pun masih berisi kolusi, dan kasus dengan banyak "
               "bukti tidak 100% kolusi.")


def render_stability() -> None:
    evaluation = services.table("graph_evaluation")
    if evaluation.empty:
        return
    frame = evaluation.dropna(subset=["lift"]).copy()
    frame = frame[frame["bulan"].astype(str) >= "2022-01"]
    st.markdown("#### Apakah komunitasnya bertahan dan memberi peringatan dini?")
    long = frame.melt(id_vars="bulan", value_vars=["modularity", "persistensi", "precision"], var_name="ukuran", value_name="nilai")
    long["ukuran"] = long["ukuran"].map({"modularity": "modularity", "persistensi": "persistensi (Jaccard)",
                                         "precision": "precision peringatan dini (≤ 3 bln)"})
    figure = px.line(long, x="bulan", y="nilai", color="ukuran", labels={"bulan": "", "nilai": "", "ukuran": ""},
                     color_discrete_sequence=[ui.BLUE, ui.GREEN, ui.ORANGE])
    figure.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05], legend=dict(orientation="h", y=-0.15))
    recent = frame[frame["bulan"].astype(str) >= "2025-01"]
    st.plotly_chart(ui.style_figure(figure, "Kualitas komunitas per snapshot bulanan",
                                    f"rata-rata 2025–2026: persistensi {recent['persistensi'].mean():.0%} · precision "
                                    f"{recent['precision'].mean():.0%} · lift {recent['lift'].mean():.1f}× dibanding pegawai acak",
                                    height=320), use_container_width=True)
    st.caption("Modularity tinggi karena graph risiko sengaja jarang. Persistensi dan precision peringatan dini lebih bermakna: "
               "seberapa sering anggota komunitas terbukti terlibat kasus dalam 3 bulan berikutnya.")
