"""Tampilan bersama Risk Engine 4 pilar: kartu skor pilar, waterfall kontribusi, dan grafik alasan."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import components as ui


def sigmoid(value: float) -> float:
    return float(1 / (1 + np.exp(-value)))


def has_pillars(row) -> bool:
    return all(column in row and pd.notna(row[column]) for column in ("c_base", "c_rule", "c_classifier", "c_graph"))


def pillar_score_text(row, pillar: str) -> str:
    value = row.get(ui.PILLAR_SCORE_COLUMN[pillar])
    if value is None or pd.isna(value):
        return "kosong"
    return f"{float(value):.0f}"


def pillar_cards(row) -> None:
    """Empat kartu: skor tiap pilar (0–100) + kontribusinya ke skor akhir (log-odds)."""
    columns = st.columns(4)
    for column, pillar in zip(columns, ui.PILLARS):
        contribution = float(row.get(f"c_{pillar}", 0) or 0)
        sub = f"kontribusi {contribution:+.2f} log-odds"
        if pillar == "iforest" and pillar_score_text(row, pillar) == "kosong":
            sub = "approver < 10 keputusan · nilai netral"
        column.markdown(ui.kpi_card(ui.PILLAR_LABELS[pillar], pillar_score_text(row, pillar), sub, ui.PILLAR_COLORS[pillar]),
                        unsafe_allow_html=True)
    escalation = str(row.get("escalation") or "")
    if escalation and escalation != "nan":
        st.caption(f"🛟 Jaring pengaman aktif: {escalation} → minimal band **Sedang**.")


def contribution_waterfall(row, title: str = "", height: int = 300) -> go.Figure:
    """Dasar + kontribusi 4 pilar = logit skor akhir (eksak)."""
    base = float(row["c_base"])
    contributions = [float(row[f"c_{p}"]) for p in ui.PILLARS]
    final_logit = base + sum(contributions)
    labels = ["Dasar (rata-rata 2025)"] + [f"{ui.PILLAR_SHORT[p]} ({pillar_score_text(row, p)})" for p in ui.PILLARS] + ["Skor akhir"]
    figure = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=[base] + contributions + [final_logit],
        measure=["absolute"] + ["relative"] * 4 + ["total"],
        text=[f"{base:+.2f}"] + [f"{c:+.2f}" for c in contributions] + [f"{final_logit:+.2f} → {sigmoid(final_logit) * 100:.0f}/100"],
        textposition="outside", connector=dict(line=dict(color=ui.GRID)),
        increasing=dict(marker=dict(color=ui.ORANGE)), decreasing=dict(marker=dict(color=ui.BLUE)),
        totals=dict(marker=dict(color=ui.INK))))
    figure.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="log-odds (kanan = menaikkan risiko)", showlegend=False)
    return ui.style_figure(figure, title or "Dari mana skor ini berasal?",
                           "Kontribusi tiap pilar relatif terhadap rata-rata validation; jumlahnya persis = skor akhir",
                           height=height)


def reasons_chart(reasons: pd.DataFrame, title: str, height: int = 230) -> go.Figure:
    """Alasan per alert (kolom: reason, contribution, pilar), diurutkan dari kontribusi terbesar."""
    frame = reasons.sort_values("contribution")
    figure = go.Figure(go.Bar(
        x=frame["contribution"], y=frame["reason"], orientation="h",
        marker_color=[ui.PILLAR_COLORS.get(p, ui.ORANGE) for p in frame["pilar"]],
        text=[f"+{v:.2f} · {ui.PILLAR_SHORT.get(p, p)}" for v, p in zip(frame["contribution"], frame["pilar"])],
        textposition="outside", hovertext=[ui.PILLAR_LABELS.get(p, p) for p in frame["pilar"]]))
    figure.update_layout(xaxis_title="kontribusi ke skor akhir (log-odds)", yaxis=dict(automargin=True),
                         xaxis_range=[0, max(frame["contribution"].max() * 1.6, 0.5)])
    return ui.style_figure(figure, title, "Warna = pilar asal alasan", height=height)


def pillar_legend() -> str:
    return "".join(f"<span class='chip'><b style='color:{ui.PILLAR_COLORS[p]}'>■</b> {html.escape(ui.PILLAR_LABELS[p])}</span>"
                   for p in ui.PILLARS)


def band_rules_text(settings: dict) -> str:
    high = settings.get("final_threshold", 0.5) * 100
    watch = settings.get("watch_threshold", 0.25) * 100
    return (f"**Tinggi** ≥ {high:.0f} (F1 terbaik, prediksi out-of-fold 2025) · **Sedang** ≥ {watch:.0f} (½ threshold Tinggi) "
            "atau jaring pengaman pilar · **Rendah** di bawahnya")


EVIDENCE_LABELS = {"risky_pair": "Pasangan berisiko", "persistent_pair": "Hubungan berisiko bertahan",
                   "same_community": "Satu komunitas risiko", "shared_debtor_motif": "Motif debitur bersama",
                   "central_approver": "Approver sentral"}


def graph_evidence_block(request_id: str, evidence: pd.DataFrame) -> None:
    """Bukti jaringan (graph heterogen) untuk satu berkas: relasi pemicu, node pendukung, alasan, dan batasannya."""
    rows = evidence[evidence["request_id"].astype(str) == str(request_id)] if not evidence.empty else evidence
    if rows is None or rows.empty:
        return
    st.markdown(f"**Bukti jaringan** · {len(rows)} dari {len(EVIDENCE_LABELS)} bukti terpenuhi "
                f"(snapshot {rows['periode_snapshot'].iloc[0]}, {rows['jendela'].iloc[0]})")
    for row in rows.itertuples(index=False):
        st.markdown(f"- **{EVIDENCE_LABELS.get(row.bukti, row.bukti)}**: {html.escape(str(row.alasan))} "
                    f"(`{row.fitur}` = {row.nilai:g}; relasi {html.escape(str(row.relasi))}; {html.escape(str(row.node_pendukung))})  \n"
                    f"  <span style='color:{ui.MUTED};font-size:0.85em'>Batasan: {html.escape(str(row.batasan_interpretasi))}</span>",
                    unsafe_allow_html=True)
