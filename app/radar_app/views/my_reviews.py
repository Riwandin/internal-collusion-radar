"""Investigator · Riwayat Investigasi: filter & cari → total kasus dan keputusan → jejak audit → memo agen yang disetujui."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import DECISIONS


def render() -> None:
    ui.page_header("Riwayat Investigasi", "Alur: saring periode/investigator → lihat progres kasus → telusuri jejak audit dan memo yang disetujui")
    store = services.get_store()
    reviews, actions = store.reviews().copy(), store.agent_actions().copy()
    transactions = services.table("transactions_scored")
    reviews["reviewed_at"] = pd.to_datetime(reviews["reviewed_at"], errors="coerce", utc=True)
    actions["created_at"] = pd.to_datetime(actions["created_at"], errors="coerce", utc=True)

    query = st.text_input("🔎 Cari riwayat", key="hist_search", placeholder="ID transaksi atau kata dalam catatan/memo")
    c_date, c_user, c_decision = st.columns([1.4, 1.2, 1.8])
    dates = reviews["reviewed_at"].dropna()
    today = pd.Timestamp.now(tz="UTC").date()
    date_range = c_date.date_input("Rentang tanggal keputusan", (dates.min().date() if len(dates) else today, today), key="hist_dates")
    users = c_user.multiselect("Investigator", sorted(reviews["username"].dropna().unique()), key="hist_users")
    decisions = c_decision.multiselect("Keputusan", DECISIONS, key="hist_decisions")
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        day = reviews["reviewed_at"].dt.date
        reviews = reviews[(day >= date_range[0]) & (day <= date_range[1])]
        actions = actions[(actions["created_at"].dt.date >= date_range[0]) & (actions["created_at"].dt.date <= date_range[1])]
    if users:
        reviews = reviews[reviews["username"].isin(users)]
        actions = actions[actions["username"].isin(users)]
    if decisions:
        reviews = reviews[reviews["decision"].isin(decisions)]
    if query.strip():
        text = query.strip()
        reviews = reviews[reviews["request_id"].astype(str).str.contains(text, case=False, regex=False)
                          | reviews["note"].astype(str).str.contains(text, case=False, regex=False)]
        actions = actions[actions["request_id"].astype(str).str.contains(text, case=False, regex=False)
                          | actions["content"].astype(str).str.contains(text, case=False, regex=False)]

    total_cases = int((transactions["risk_band"] == "Tinggi").sum())
    latest = reviews.sort_values("reviewed_at").groupby("request_id").last() if len(reviews) else reviews.set_index("request_id")
    approved = actions[actions["status"] == "DISETUJUI"]
    ui.kpi_row([("Total kasus", ui.number(total_cases), "seluruh alert Tinggi 2025–2026", ui.BLUE),
                ("Kasus sudah diputus", ui.number(len(latest)), f"{ui.percent(len(latest) / max(total_cases, 1), 1)} dari total kasus", ui.GREEN),
                ("Tindak lanjut / eskalasi", ui.number(latest["decision"].str.startswith("Tindak").sum()), "keputusan terakhir per kasus", ui.ORANGE),
                ("Alarm palsu", ui.number(latest["decision"].str.startswith("Bukan").sum()), "umpan balik pelatihan ulang", ui.YELLOW),
                ("Perlu data tambahan", ui.number(latest["decision"].str.startswith("Perlu").sum()), "", ui.PURPLE),
                ("Memo disetujui", ui.number(len(approved)), f"sumber: {store.backend}", ui.RED)])
    ui.explain({"Total kasus": "jumlah transaksi berlevel Tinggi di seluruh data (tidak terpengaruh filter).",
                "Kasus sudah diputus": "kasus unik yang punya keputusan sesuai filter; bila diputus berkali-kali, dihitung keputusan terakhir.",
                "Jejak audit": "setiap keputusan disimpan sebagai baris baru (tidak ditimpa), lengkap dengan skor & model saat itu.",
                "Memo disetujui": "draf memo dari agen AI yang sudah ditinjau dan disetujui investigator (human-in-the-loop)."})
    if reviews.empty:
        st.info("Belum ada keputusan sesuai filter. Catat keputusan dari halaman Detail Kasus → tab Keputusan.")
    else:
        left, right = st.columns([1, 1.4])
        counts = latest["decision"].value_counts().rename_axis("keputusan").reset_index(name="jumlah")
        left.plotly_chart(ui.style_figure(px.pie(counts, names="keputusan", values="jumlah", hole=.55,
                                                 color_discrete_sequence=[ui.ORANGE, ui.GREEN, ui.YELLOW]), "Komposisi keputusan", height=320),
                          use_container_width=True)
        daily = reviews.groupby([reviews["reviewed_at"].dt.date.rename("tanggal"), "decision"]).size().rename("jumlah").reset_index()
        figure = px.bar(daily, x="tanggal", y="jumlah", color="decision", labels={"decision": "keputusan"},
                        color_discrete_sequence=[ui.ORANGE, ui.GREEN, ui.YELLOW])
        right.plotly_chart(ui.style_figure(figure, "Keputusan per hari", "produktivitas investigasi", height=320), use_container_width=True)
        st.markdown("#### Jejak audit keputusan")
        names = services.name_map()
        info = transactions.set_index("request_id")
        trail = reviews.sort_values("reviewed_at", ascending=False).copy()
        known = trail["request_id"].isin(info.index)
        trail.loc[known, "requester"] = trail.loc[known, "request_id"].map(info["requester_id"]).map(names)
        trail.loc[known, "approver"] = trail.loc[known, "request_id"].map(info["approver_id"]).map(names)
        trail.loc[known, "unit"] = trail.loc[known, "request_id"].map(info["requester_unit"])
        trail.loc[known, "nominal"] = trail.loc[known, "request_id"].map(info["amount"])
        columns = ["reviewed_at", "username", "request_id", "decision", "note", "risk_score", "requester", "approver", "unit",
                   "nominal", "model_name"]
        trail = trail.reindex(columns=columns)
        st.dataframe(trail, hide_index=True, use_container_width=True, height=320,
                     column_config={"reviewed_at": st.column_config.DatetimeColumn("waktu", format="DD MMM YYYY HH:mm"),
                                    "username": "investigator", "decision": "keputusan", "note": "catatan",
                                    "risk_score": st.column_config.NumberColumn("skor", format="%.0f"),
                                    "nominal": st.column_config.NumberColumn(format="Rp %,.0f"), "model_name": "model"})
        st.download_button("Unduh jejak audit (CSV)", trail.to_csv(index=False), "jejak_audit_keputusan.csv", "text/csv")
    st.markdown("#### Memo hasil agen yang disetujui")
    if approved.empty:
        st.info("Belum ada memo yang disetujui. Minta draf memo dari Detail Kasus → tab Draf memo (agen AI).")
    else:
        for _, memo in approved.sort_values("created_at", ascending=False).iterrows():
            with st.expander(f"{memo['request_id']} · disetujui {memo['approved_by']} · {memo['created_at']:%d %b %Y %H:%M}"):
                st.markdown(memo["content"])
    if len(actions):
        with st.expander("Log lengkap aksi agen (termasuk draf yang menunggu persetujuan)"):
            st.dataframe(actions.sort_values("created_at", ascending=False), hide_index=True, use_container_width=True)
