"""Investigator · Detail Kasus: cari kasus → scorecard → alasan skor → jaringan kasus → riwayat → keputusan & memo AI."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import pillars, services
from ..ui.network import LEGEND_HTML, build_graph, plotly_network, pyvis_html
from .common import DECISIONS, RED_FLAG_COLUMNS, flags_text, latest_decisions, show_guard_checks
from .inv_overview import search_mask
from .network import community_graph, latest_anomaly_scores, show_graph


def select_case(transactions: pd.DataFrame, decisions: pd.DataFrame) -> pd.Series | None:
    """Pencarian & filter di atas → satu kasus terpilih (default: kasus yang dibuka dari Overview)."""
    query = st.text_input("🔎 Cari kasus", key="case_search", placeholder="ID transaksi, nama, atau ID pegawai")
    c_level, c_status, c_unit, c_pick = st.columns([1.1, 1.1, 1.4, 2.4])
    levels = c_level.multiselect("Level risiko", ["Tinggi", "Sedang"], default=["Tinggi", "Sedang"], key="case_levels")
    status = c_status.selectbox("Status keputusan", ["semua", "belum diputus", "sudah diputus"], key="case_status")
    frame = transactions[transactions["risk_band"].isin(levels or ["Tinggi", "Sedang"])]
    units = c_unit.multiselect("Unit kerja", sorted(frame["requester_unit"].dropna().unique()), key="case_units")
    if units:
        frame = frame[frame["requester_unit"].isin(units)]
    if status == "belum diputus":
        frame = frame[~frame["request_id"].isin(decisions.index)]
    elif status == "sudah diputus":
        frame = frame[frame["request_id"].isin(decisions.index)]
    opened = st.session_state.get("case_id")
    if query.strip():
        frame = frame[search_mask(frame, query)]
        if query.strip().upper() in set(transactions["request_id"]):
            opened = query.strip().upper()
    options = frame.sort_values(["risk_score", "event_time"], ascending=False)["request_id"].head(500).tolist()
    if opened in set(transactions["request_id"]) and opened not in options:
        options.insert(0, opened)
    if not options:
        c_pick.selectbox("Pilih kasus", ["(tidak ada yang cocok)"], disabled=True)
        return None
    names = services.name_map()
    rows = transactions.set_index("request_id")
    request_id = c_pick.selectbox(
        "Pilih kasus", options, index=options.index(opened) if opened in options else 0, key="case_pick",
        format_func=lambda r: f"{r} · skor {rows.at[r, 'risk_score']:.0f} · {names.get(rows.at[r, 'requester_id'], '?')} → "
                              f"{names.get(rows.at[r, 'approver_id'], '?')}")
    st.session_state["case_id"] = request_id
    return transactions[transactions["request_id"] == request_id].iloc[0]


def score_gauge(score: float, threshold: float) -> go.Figure:
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " /100", "font": {"size": 30}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": ui.INK},
               "steps": [{"range": [0, threshold * 50], "color": "#dbe9fb"},
                         {"range": [threshold * 50, threshold * 100], "color": "#fbe7b5"},
                         {"range": [threshold * 100, 100], "color": "#f8c9b3"}],
               "threshold": {"line": {"color": ui.ORANGE, "width": 3}, "value": threshold * 100}}))
    figure.update_layout(height=210, margin=dict(l=20, r=20, t=20, b=0))
    return figure


def shap_bar(explanations: pd.DataFrame) -> go.Figure:
    return pillars.reasons_chart(explanations, "Alasan utama skor (kontribusi per pilar)", height=260)


def ego_network(row: pd.Series) -> nx.Graph | None:
    pairs = services.table("pair_network")
    people = {row["requester_id"], row["approver_id"]}
    neighborhood = pairs[pairs["requester_id"].isin(people) | pairs["approver_id"].isin(people)].copy()
    if neighborhood.empty:
        return None
    neighborhood = neighborhood.sort_values(["risky", "n"], ascending=False).head(30)
    edges = pd.DataFrame({"node_a": neighborhood["requester_id"], "node_b": neighborhood["approver_id"],
                          "weight": np.where(neighborhood["risky"].astype(str).str.lower().eq("true"),
                                             neighborhood["z"].clip(lower=1), 0.4),
                          "kind": "RA", "redflag": neighborhood["risky"].astype(str).str.lower().eq("true") & (neighborhood["z"] >= 3),
                          "dependency": neighborhood["dependency"], "n_tx": neighborhood["n"], "z": neighborhood["z"]})
    community_edges = services.table("community_edges")
    shared = community_edges[(community_edges["kind"] == "RR") & (community_edges["node_a"].isin(people) | community_edges["node_b"].isin(people))]
    edges = pd.concat([edges, shared[["node_a", "node_b", "weight", "kind", "shared"]]], ignore_index=True)
    return build_graph(edges, services.table("employees"), highlight=people)


def render_detail(row: pd.Series, settings: dict) -> None:
    names = services.name_map()
    explanations = services.table("alert_explanations")
    reasons = explanations[explanations["request_id"] == row["request_id"]]
    st.markdown(f"### Detail alert {row['request_id']} &nbsp; {ui.band_badge(row['risk_band'])}", unsafe_allow_html=True)
    st.caption(f"approval {row['approve_ts']:%A, %d %b %Y %H:%M} · periode model: {row.get('model_period', '-')}")
    column_a, column_b, column_c = st.columns([1, 1.4, 1.3])
    with column_a:
        st.plotly_chart(score_gauge(float(row["risk_score"]), settings.get("final_threshold", 0.5)), use_container_width=True)
        st.markdown(f"**Requester:** {names.get(row['requester_id'], row['requester_id'])} ({row['requester_id']}) · {row['requester_unit']}  \n"
                    f"**Approver:** {names.get(row['approver_id'], row['approver_id'])} ({row['approver_id']}) · {row['approver_position']}  \n"
                    f"**Debitur:** {row.get('debtor_name_masked', '-')} ({row['debtor_id']}) · {row['debtor_type']} · {row['business_sector']}  \n"
                    f"**Kredit:** {row['loan_product']} · {ui.rupiah(row['amount'], short=False)} · {ui.percent(row['limit_ratio'])} limit approver")
    with column_b:
        if len(reasons):
            st.plotly_chart(shap_bar(reasons), use_container_width=True)
        else:
            st.info("Alasan disimpan untuk alert Tinggi/Sedang.")
        flags = [ui.RED_FLAG_LABELS[c] for c in RED_FLAG_COLUMNS if row.get(c) == 1]
        st.markdown("**Red flag terpenuhi:** " + (ui.chips(flags) if flags else "tidak ada"), unsafe_allow_html=True)
    with column_c:
        st.markdown("**Bukti dari data**")
        evidence = [e for e in str(row.get("evidence") or "").split(" | ") if e and e != "nan"]
        st.markdown("<ul class='evidence'>" + "".join(f"<li>{e}</li>" for e in evidence) + "</ul>" if evidence else "-",
                    unsafe_allow_html=True)
        pillars.graph_evidence_block(str(row["request_id"]), services.table("graph_evidence"))
        anomaly_score = row.get("approver_anomaly_score")
        if pd.notna(anomaly_score):
            st.markdown(f"**Skor anomali approver (IF, model terpisah):** {anomaly_score:.0f}/100")
        st.caption("Skor adalah indikasi berbasis data, bukan bukti. Keputusan ada di tangan investigator.")
    tab_network, tab_history, tab_decision, tab_memo = st.tabs(["① Jaringan kasus", "② Riwayat pasangan", "③ Keputusan", "④ Draf memo (agen AI)"])
    with tab_history:
        transactions = services.table("transactions_scored")
        pair = transactions[(transactions["requester_id"] == row["requester_id"]) & (transactions["approver_id"] == row["approver_id"])]
        figure = px.scatter(pair, x="event_time", y="risk_score", color="risk_band", color_discrete_map=ui.BAND_COLORS,
                            size=pair["amount"] / pair["amount"].max() * 20 + 4, hover_data=["request_id", "amount", "rule_score"],
                            labels={"event_time": "waktu", "risk_score": "skor"})
        st.plotly_chart(ui.style_figure(figure, f"{len(pair)} transaksi pasangan ini (2025–2026)",
                                        "ukuran titik = nominal"), use_container_width=True)
        summary = {"rasio requester → approver": ui.percent(row.get("requester_approver_ratio")),
                   "kali lipat kebiasaan unit (lift)": f"{row.get('pair_dependency_lift', np.nan):.1f}×" if pd.notna(row.get("pair_dependency_lift")) else "-",
                   "% disetujui (1 thn)": ui.percent(row.get("pair_approval_rate")),
                   "debitur sama ≤ 1 thn": ui.number(row.get("debtor_loans_1y"))}
        ui.kpi_row([(k, v, "", ui.PURPLE) for k, v in summary.items()])
    with tab_network:
        st.caption("Jaringan kasus = hubungan requester & approver kasus ini dengan pegawai lain (graph 12 bulan terakhir). "
                   "Garis oranye/ungu menandakan hubungan menyimpang; garis putus-putus = berbagi debitur.")
        graph = ego_network(row)
        if graph is None:
            st.info("Pasangan ini belum punya riwayat di snapshot graph terbaru.")
        else:
            show_graph(graph, "Jaringan di sekitar pasangan ini", height=480)
        render_case_community(row)
    with tab_decision:
        user = services.current_user()
        decisions = latest_decisions()
        if row["request_id"] in decisions.index:
            last = decisions.loc[row["request_id"]]
            st.success(f"Keputusan terakhir: **{last['decision']}** oleh {last['username']} ({last['reviewed_at']}). {last['note'] or ''}")
        with st.form(f"decision_{row['request_id']}"):
            decision = st.radio("Keputusan investigator", DECISIONS, horizontal=True)
            note = st.text_area("Catatan (penjelasan sah yang dicek, bukti, rencana klarifikasi)", height=90)
            if st.form_submit_button("Simpan keputusan", type="primary"):
                services.get_store().save_review(user["username"], user["role"], row["request_id"], decision, note,
                                                 float(row["risk_score"]), settings.get("model_name", "-"))
                st.toast("Keputusan tersimpan (jejak audit + umpan balik model).")
                st.rerun()
    with tab_memo:
        memo_key = f"memo_{row['request_id']}"
        if st.button("Minta agen menyusun draf memo investigasi", key=f"btn_{memo_key}"):
            with st.spinner("Agen menjalankan detail_alert → cari_regulasi → draf_memo_investigasi…"):
                st.session_state[memo_key] = services.get_agent().run(
                    f"Siapkan draf memo investigasi untuk transaksi {row['request_id']}", "investigator",
                    services.current_user()["username"])
        result = st.session_state.get(memo_key)
        if result:
            st.caption(f"Mode planner: {result['mode']} · {len(result['steps'])} langkah · {result['latency_ms']:.0f} ms")
            for step in result["steps"]:
                st.markdown(f"- langkah {step['step']}: `{step['tool']}` → {step['status']}")
            if result.get("pending_action"):
                memo = st.text_area("Draf memo (boleh disunting sebelum disetujui)", result["pending_action"]["memo"], height=360,
                                    key=f"text_{memo_key}")
                approve, reject = st.columns(2)
                if approve.button("Setujui & simpan memo", type="primary", key=f"ok_{memo_key}"):
                    services.get_store().approve_agent_action(row["request_id"], "memo_investigasi", memo,
                                                              services.current_user()["username"])
                    st.success("Memo disetujui dan tersimpan (agent_actions, status DISETUJUI).")
                if reject.button("Buang draf", key=f"no_{memo_key}"):
                    st.session_state.pop(memo_key, None)
                    st.rerun()
            with st.expander("Status guardrail"):
                show_guard_checks(result["checks"])


def render_case_community(row: pd.Series) -> None:
    """Komunitas risiko tempat requester/approver kasus ini berada (hasil graph analytics)."""
    members, communities = services.table("community_members"), services.table("communities")
    people = {row["requester_id"], row["approver_id"]}
    ids = sorted(members.loc[members["employee_id"].isin(people), "community_id"].unique())
    if not ids:
        st.caption("Requester/approver kasus ini tidak termasuk komunitas risiko mana pun.")
        return
    st.markdown("**Komunitas risiko yang terkait kasus ini**")
    st.dataframe(communities[communities["community_id"].isin(ids)][["community_id", "skor_komunitas", "prioritas", "anggota",
                                                                     "approver_utama", "unit", "aktif_bulan"]],
                 hide_index=True, use_container_width=True,
                 column_config={"skor_komunitas": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f"),
                                "community_id": "ID", "aktif_bulan": "aktif (bln)"})
    community_id = st.selectbox("Lihat graph komunitas", ids, key=f"case_comm_{row['request_id']}")
    group = members[members["community_id"] == community_id].copy()
    group["skor anomali (IF)"] = group["employee_id"].map(latest_anomaly_scores())
    graph_column, members_column = st.columns([1.5, 1])
    with graph_column:
        show_graph(community_graph(community_id), f"Komunitas {community_id}", height=440)
    members_column.dataframe(group[["employee_name", "employee_id", "role", "unit", "pagerank", "skor anomali (IF)"]],
                             hide_index=True, use_container_width=True, height=440,
                             column_config={"employee_name": "nama",
                                            "pagerank": st.column_config.NumberColumn("sentralitas", format="%.2f"),
                                            "skor anomali (IF)": st.column_config.NumberColumn(format="%.0f")})


def render() -> None:
    ui.page_header("Detail Kasus", "Alur: cari kasus → pahami alasan skor → periksa jaringan kasus → catat keputusan & setujui memo")
    transactions = services.table("transactions_scored")
    if transactions.empty:
        ui.empty_state("Data belum tersedia.")
        return
    settings = services.app_settings()
    decisions = latest_decisions()
    row = select_case(transactions, decisions)
    if row is None:
        st.info("Tidak ada kasus yang cocok dengan pencarian/filter.")
        return
    flags = [c for c in RED_FLAG_COLUMNS if row.get(c) == 1]
    decision = decisions["decision"].get(row["request_id"], "Belum diputus")
    anomaly_score = row.get("approver_anomaly_score")
    ui.kpi_row([("Skor risiko", f"{row['risk_score']:.0f}/100", f"level {row['risk_band']}", ui.BAND_COLORS.get(row["risk_band"], ui.BLUE)),
                ("Nominal kredit", ui.rupiah(row["amount"]), f"{ui.percent(row['limit_ratio'])} limit approver", ui.YELLOW),
                ("Estimasi potensi kerugian", ui.rupiah(row["amount"] * row["risk_probability"] * settings.get("lgd", 0.6)),
                 "nominal × probabilitas × LGD", ui.PURPLE),
                ("Red flag terpenuhi", f"{len(flags)}/{len(RED_FLAG_COLUMNS)}", "rules bisnis", ui.ORANGE),
                ("Skor anomali approver", f"{anomaly_score:.0f}/100" if pd.notna(anomaly_score) else "-", "Isolation Forest", ui.GREEN),
                ("Status keputusan", decision.split(" (")[0], "tab ③ Keputusan", ui.RED if decision == "Belum diputus" else ui.BLUE)])
    ui.explain({
        "Skor risiko": "probabilitas kolusi dari model (×100). Garis oranye pada gauge = ambang alert Tinggi.",
        "Tiga alasan utama (SHAP)": "fitur yang paling menaikkan skor kasus ini; warna = kelompok fitur (transaksi, riwayat, rules, graph).",
        "Red flag": "aturan bisnis yang terpenuhi; berdiri sendiri bukan bukti, tetapi menguatkan alasan model.",
        "Skor anomali approver": "seberapa menyimpang perilaku approver dibanding rekan sejabatan (model terpisah, tanpa label).",
        "Jaringan kasus": "siapa saja yang terhubung dengan pasangan ini dan apakah mereka berada di komunitas risiko."})
    st.divider()
    render_detail(row, settings)
