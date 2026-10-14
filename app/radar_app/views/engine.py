"""Investigator & Manajemen · Risk Engine 4 Pilar: cara kerja, bobot yang dipelajari, perbandingan versi, dan bukti evaluasinya.

Semua angka dibaca dari hasil notebook (outputs/app_data). Halaman ini menjawab tiga pertanyaan penguji/pengguna:
1. Bagaimana skor akhir dibentuk dan dibaca?
2. Kenapa digabung dengan stacking (bobot dipelajari), bukan bobot manual?
3. Apakah memecah fitur menjadi 4 pilar mengorbankan akurasi? (dibandingkan dengan satu model besar / early fusion)
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import pillars, services

FINAL = "Risk Engine 4 pilar (final)"
EARLY = "Early fusion (1 model, 40 fitur)"
MANUAL70 = "Bobot manual 15/20/40/25 · ambang 70 (v9.x)"
MANUAL_VAL = "Bobot manual 15/20/40/25 · ambang val"
LR = "Logistic Regression (baseline)"
RULES = "Rules (rule_score)"
PILLAR_ONLY = {"iforest": "Pilar 2 · IF saja", "classifier": "Pilar 3 · Konten saja", "graph": "Pilar 4 · Relasi saja"}
VERSION_OF = {EARLY: "v9 · satu model, 40 fitur", MANUAL70: "v9.x · bobot manual, ambang 70", FINAL: "v10 · stacking 4 pilar"}


def _metric(frame: pd.DataFrame, model: str, column: str, default=float("nan")):
    return frame.loc[model, column] if model in frame.index else default


def _render_how_it_works(settings: dict) -> None:
    left, right = st.columns([1.25, 1])
    with left:
        st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
            data [label="Transaksi baru\\n(fitur as-of)", fillcolor="#f6f7f9", color="#8a8984"];
            p1 [label="Pilar 1 · Rules SOP\\n5 red flag → S_rule", fillcolor="#f1f1ef", color="#8a8984"];
            p2 [label="Pilar 2 · Isolation Forest\\nperilaku approver 90 hari → S_IF", fillcolor="#e7f6ef", color="#1baf7a"];
            p3 [label="Pilar 3 · Konten transaksi\\nnominal, waktu, debitur → S_cls", fillcolor="#eaf2fc", color="#2a78d6"];
            p4 [label="Pilar 4 · Relasi & jaringan\\npasangan, routing, graph → S_graph", fillcolor="#fdf0e9", color="#eb6834"];
            stk [label="Stacking\\n(logistic regression,\\nbobot dipelajari 2025)", fillcolor="#fff7e0", color="#eda100"];
            out [label="Skor risiko 0–100\\n+ band + kontribusi pilar", fillcolor="#1f2328", fontcolor="white", color="#1f2328"];
            data -> p1; data -> p2; data -> p3; data -> p4; p1 -> stk; p2 -> stk; p3 -> stk; p4 -> stk; stk -> out; }""")
    with right:
        st.markdown("##### Membaca skor")
        st.markdown(
            "- **Skor akhir** = peluang terkalibrasi × 100. *Skor 80* berarti dari kasus serupa yang direview audit, ±80% "
            "terbukti kolusi.\n"
            "- **Skor pilar** (0–100) menjawab satu pertanyaan saja; tidak dijumlahkan langsung.\n"
            "- **Kontribusi pilar** (log-odds) menunjukkan seberapa jauh pilar itu menggeser skor dari rata-rata 2025. "
            "Dasar + 4 kontribusi = skor akhir, **persis**.\n"
            f"- {pillars.band_rules_text(settings)}.")
    dictionary = services.table("feature_dictionary")
    models = settings.get("pillar_models", {})
    rows = []
    for pillar in ui.PILLARS:
        label = ui.PILLAR_LABELS[pillar]
        features = dictionary[dictionary["pilar"] == label]["fitur"].tolist() if "pilar" in dictionary else []
        rows.append({"pilar": label, "pertanyaan": ui.PILLAR_QUESTION[pillar], "model": models.get(pillar, "-"),
                     "pakai label?": "ya (train 2021–2024)" if pillar in ("classifier", "graph") else "tidak",
                     "input": ", ".join(features[:8]) + (f" … (+{len(features) - 8})" if len(features) > 8 else "")
                     if features else ("8 fitur perilaku approver" if pillar == "iforest" else "-")})
    st.markdown("##### Empat pilar, fitur tidak tumpang tindih")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Isolation Forest di-fit pada approver-bulan s.d. {settings.get('if_fit_end', '2024-12')}; skor 2025–2026 adalah "
               "persentil terhadap distribusi latih. Flag rules tidak menjadi fitur model mana pun, dan fitur graph hanya masuk Pilar 4, "
               "sehingga satu sinyal tidak terhitung berkali-kali.")
    with st.expander("Kenapa tidak memakai bobot manual (mis. 15/20/40/25)?"):
        st.markdown(
            "1. **Bobot manual adalah opini** yang tidak pernah diuji. Stacking *mengukur* tambahan informasi tiap pilar di data "
            "2025 yang belum dilihat pilar.\n"
            "2. **Rata-rata berbobot bukan peluang**: angka 70 hanya hasil penjumlahan, sehingga ambang 70/40 tidak punya arti. "
            "Skor stacking terkalibrasi (lihat tab Kalibrasi).\n"
            "3. **Satu pilar yang sangat yakin bisa tenggelam** di rata-rata. Di v10, penggabung belajar dari data dan ada "
            "jaring pengaman per pilar.\n"
            "4. **Skala berbeda** (persentil IF, hitungan rule, peluang model) diterjemahkan ke log-odds oleh model, bukan dirata-rata.\n"
            "5. **Monoton**: koefisien pilar dibatasi ≥ 0. Tanpa batasan, koefisien Rules sedikit negatif (flag SOP malah menurunkan "
            "skor) karena nilai mentahnya sudah dibaca Pilar 3. Itu tidak bisa dijelaskan ke auditor, jadi bobot Rules menjadi 0 dan "
            "rules tetap tampil sebagai bukti SOP.")


def _render_weights(settings: dict) -> None:
    weights = services.table("fusion_weights")
    if weights.empty:
        ui.empty_state("Tabel fusion_weights belum ada. Jalankan notebook v10.")
        return
    left, middle = st.columns([1.1, 1])
    figure = go.Figure()
    figure.add_bar(y=weights["label"], x=weights["bobot_manual_v9x"], orientation="h", name="manual v9.x (opini)",
                   marker_color=ui.MUTED)
    figure.add_bar(y=weights["label"], x=weights["bobot_efektif"], orientation="h", name="dipelajari (stacking)",
                   marker_color=[ui.PILLAR_COLORS[p] for p in weights["pilar"]], text=[ui.percent(v) for v in weights["bobot_efektif"]],
                   textposition="outside")
    figure.update_layout(barmode="group", xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"))
    left.plotly_chart(ui.style_figure(figure, "Bobot dari data vs bobot opini", "bobot efektif = |koefisien| × sebaran input (validation 2025)",
                                      height=330), use_container_width=True)
    correlation = services.table("pillar_correlation")
    if not correlation.empty:
        matrix = correlation.set_index("pilar")
        figure = px.imshow(matrix, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto")
        figure.update_layout(coloraxis_showscale=False)
        middle.plotly_chart(ui.style_figure(figure, "Korelasi antarpilar: sedang, bukan kembar",
                                            "Spearman di validation 2025; ±1 berarti satu pilar sudah cukup", height=330),
                            use_container_width=True)
    coefficients = settings.get("fusion_coefficients", {})
    st.markdown("**Koefisien penggabung monoton** (log-odds per satuan input, pilar ≥ 0): " + " · ".join(
        f"`{name}` {value:+.2f}" for name, value in coefficients.items()) + f" · intercept {settings.get('fusion_intercept', 0):+.2f}")

    left, right = st.columns(2)
    share = pd.Series(settings.get("pillar_share_test", {}))
    if len(share):
        figure = px.pie(values=share.values, names=[ui.PILLAR_LABELS[p] for p in share.index], hole=.55,
                        color=[ui.PILLAR_LABELS[p] for p in share.index],
                        color_discrete_map={ui.PILLAR_LABELS[p]: ui.PILLAR_COLORS[p] for p in ui.PILLARS})
        left.plotly_chart(ui.style_figure(figure, "Porsi pengaruh tiap pilar di test 2026", "rata-rata |kontribusi| per transaksi",
                                          height=330), use_container_width=True)
    explanations = services.table("alert_explanations")
    if not explanations.empty and "pilar" in explanations:
        first = explanations[explanations["rank"] == 1]["pilar"].map(ui.PILLAR_LABELS).value_counts(normalize=True)
        figure = px.bar(x=first.values, y=first.index, orientation="h", color=first.index,
                        color_discrete_map={ui.PILLAR_LABELS[p]: ui.PILLAR_COLORS[p] for p in ui.PILLARS},
                        labels={"x": "", "y": ""})
        figure.update_layout(showlegend=False, xaxis_tickformat=".0%")
        right.plotly_chart(ui.style_figure(figure, "Pilar pemberi alasan nomor 1 pada alert", "alert Tinggi & Sedang 2025–2026",
                                           height=330), use_container_width=True)
    shap_global = services.table("shap_global")
    if not shap_global.empty and "pengaruh terhadap skor akhir" in shap_global:
        top = shap_global.head(12).iloc[::-1]
        figure = go.Figure(go.Bar(x=top["pengaruh terhadap skor akhir"], y=top["feature"], orientation="h",
                                  marker_color=[ui.GROUP_COLORS.get(g, ui.BLUE) for g in top["kelompok"]],
                                  hovertext=top["arti"] + " · " + top["pilar"]))
        figure.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(ui.style_figure(figure, "Fitur yang paling menentukan skor akhir",
                                        "porsi |SHAP| di dalam pilar × porsi pilar; warna = kelompok fitur", height=400),
                        use_container_width=True)


def _render_versions(settings: dict) -> None:
    comparison = services.table("fusion_comparison")
    if comparison.empty:
        ui.empty_state("Tabel fusion_comparison belum ada. Jalankan notebook v10.")
        return
    period = st.radio("Periode", ["test 2026", "validation 2025"], horizontal=True, key="engine_period")
    frame = comparison[comparison["periode"] == period].set_index("model")
    final_pr, early_pr = _metric(frame, FINAL, "PR-AUC"), _metric(frame, EARLY, "PR-AUC")
    ui.kpi_row([
        ("v10 · Risk Engine", f"PR-AUC {final_pr:.3f}", f"precision {ui.percent(_metric(frame, FINAL, 'precision'))} · "
                                                         f"recall {ui.percent(_metric(frame, FINAL, 'recall'))}", ui.ORANGE),
        ("v9 · 1 model (40 fitur)", f"PR-AUC {early_pr:.3f}", f"precision {ui.percent(_metric(frame, EARLY, 'precision'))} · "
                                                              f"recall {ui.percent(_metric(frame, EARLY, 'recall'))}", ui.BLUE),
        ("v9.x · bobot manual, ambang 70", f"recall {ui.percent(_metric(frame, MANUAL70, 'recall'))}",
         f"precision {ui.percent(_metric(frame, MANUAL70, 'precision'))} · {int(_metric(frame, MANUAL70, 'FN', 0)):,} kolusi lolos",
         ui.MUTED),
        ("Rules saja", f"{int(_metric(frame, RULES, 'FP', 0)):,} alarm palsu", f"recall {ui.percent(_metric(frame, RULES, 'recall'))}",
         ui.YELLOW)])
    gap = final_pr - early_pr
    st.markdown(
        f"**Bacaan ({period}):** stacking 4 pilar {'setara atau lebih baik daripada' if gap > -0.01 else 'sedikit di bawah'} satu model "
        f"besar (selisih PR-AUC {gap:+.3f}), sementara skornya bisa diurai per pilar dan terkalibrasi. Bobot manual dengan ambang "
        f"70 yang tidak divalidasi meloloskan {int(_metric(frame, MANUAL70, 'FN', 0)):,} kasus kolusi, dibanding "
        f"{int(_metric(frame, FINAL, 'FN', 0)):,} pada Risk Engine. Lompatan terbesar tetap dari Rules ke ML.")
    order = [RULES, PILLAR_ONLY["iforest"], PILLAR_ONLY["graph"], PILLAR_ONLY["classifier"], LR, MANUAL_VAL, EARLY, FINAL]
    plot = frame.reindex([m for m in order if m in frame.index]).reset_index()
    figure = go.Figure(go.Bar(x=plot["PR-AUC"], y=plot["model"], orientation="h", text=[f"{v:.3f}" for v in plot["PR-AUC"]],
                              textposition="outside",
                              marker_color=[ui.ORANGE if m == FINAL else ui.BLUE if m in (EARLY, MANUAL_VAL, LR) else ui.MUTED
                                            for m in plot["model"]]))
    figure.update_layout(xaxis_range=[0, 1.08])
    st.plotly_chart(ui.style_figure(figure, "Kemampuan mengurutkan (PR-AUC)", "abu-abu = satu sumber saja; biru = pembanding gabungan",
                                    height=380), use_container_width=True)
    view = frame.reset_index()
    view.insert(1, "versi", view["model"].map(VERSION_OF).fillna(""))
    st.dataframe(view[["model", "versi", "threshold", "precision", "recall", "F1", "MCC", "PR-AUC", "ROC-AUC", "TP", "FP", "FN"]]
                 .style.format({c: "{:.3f}" for c in ["threshold", "precision", "recall", "F1", "MCC", "PR-AUC", "ROC-AUC"]}),
                 hide_index=True, use_container_width=True)
    st.caption("Baris 'bobot manual' memakai pilar v10 yang sudah bersih, sehingga yang dibandingkan murni cara menggabungkan "
               "(manual vs dipelajari). Semua threshold dipilih di validation 2025; Risk Engine memakai prediksi out-of-fold.")


def _render_ablation() -> None:
    ablation = services.table("ablation")
    if ablation.empty:
        ui.empty_state("Tabel ablation belum ada.")
        return
    left, right = st.columns(2)
    sequential, leave_out = ablation.iloc[:4], ablation.iloc[4:]
    figure = px.bar(sequential, x="fitur", y="PR-AUC test", text_auto=".3f", color_discrete_sequence=[ui.BLUE],
                    labels={"fitur": ""})
    figure.update_layout(yaxis_range=[0, 1])
    left.plotly_chart(ui.style_figure(figure, "Menambah pilar satu per satu", "PR-AUC test 2026", height=340), use_container_width=True)
    if len(leave_out):
        figure = px.bar(leave_out, x="tambahan (test)", y="fitur", orientation="h", text_auto="+.3f",
                        color_discrete_sequence=[ui.ORANGE], labels={"fitur": "", "tambahan (test)": "perubahan PR-AUC vs lengkap"})
        right.plotly_chart(ui.style_figure(figure, "Kalau satu pilar dimatikan", "semakin negatif = semakin dibutuhkan", height=340),
                           use_container_width=True)
    st.caption("Pilar dengan tambahan ±0 tetap dipertahankan kalau berguna di luar skor akhir: rules sebagai alasan yang dipahami "
               "auditor, IF sebagai radar individu untuk pola yang belum berlabel.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Kalau sumber data gagal** (test 2026)")
        st.dataframe(services.table("stress_graph_outage").style.format({"PR-AUC": "{:.3f}", "recall": "{:.0%}", "precision": "{:.0%}"}),
                     hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Kalau pelaku menghindari rules** (kasus kolusi test 2026)")
        st.dataframe(services.table("stress_evasion").style.format(
            {"recall Rules": "{:.0%}", "recall Model": "{:.0%}", "PR-AUC Model": "{:.3f}"}), hide_index=True, use_container_width=True)


def _render_calibration(settings: dict) -> None:
    curves = services.table("model_curves")
    calibration = curves[curves["curve"] == "CALIBRATION"]
    left, right = st.columns([1.2, 1])
    figure = go.Figure()
    figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=ui.MUTED), name="sempurna")
    for index, (name, subset) in enumerate(calibration.groupby("model", sort=False)):
        figure.add_scatter(x=subset["x"], y=subset["y"], mode="lines+markers", name=name,
                           line=dict(color=ui.ORANGE if "Risk Engine" in name else ui.BLUE, width=3))
    figure.update_layout(xaxis_title="skor prediksi", yaxis_title="proporsi kolusi aktual (test 2026)")
    left.plotly_chart(ui.style_figure(figure, "Skor stacking ≈ peluang; rata-rata berbobot tidak",
                                      "titik dekat diagonal = skor bisa dibaca sebagai peluang", height=380), use_container_width=True)
    with right:
        st.markdown("##### Threshold dan jaring pengaman")
        st.markdown(pillars.band_rules_text(settings))
        escalation = settings.get("escalation_thresholds", {})
        st.markdown("**Jaring pengaman** (skor pilar dengan precision ≥ 90% di validation; transaksi di atasnya minimal Sedang): "
                    + ", ".join(f"{ui.PILLAR_LABELS[p]} ≥ {v * 100:.0f}" for p, v in escalation.items() if v is not None))
        evaluation = services.table("escalation_eval")
        if not evaluation.empty:
            st.dataframe(evaluation.style.format({"proporsi kolusi": "{:.0%}"}), hide_index=True, use_container_width=True)
        st.caption("Kalibrasi berlaku untuk populasi kasus yang direview audit. Pada seluruh transaksi, skor dibaca sebagai prioritas "
                   "relatif; precision populasi disajikan sebagai rentang di Command Center.")


def render() -> None:
    settings = services.app_settings()
    ui.page_header("Risk Engine 4 Pilar",
                   "Bagaimana skor akhir dibentuk, kenapa bobotnya dipelajari dari data, dan bukti bahwa cara ini bisa dipercaya")
    ui.kpi_row([
        ("Arsitektur", "4 pilar + stacking", "Rules · IF · Konten · Relasi", ui.ORANGE),
        ("PR-AUC test 2026", f"{settings.get('test_pr_auc', float('nan')):.3f}",
         f"1 model 40 fitur: {settings.get('early_fusion_test_pr_auc', float('nan')):.3f}", ui.BLUE),
        ("Threshold Tinggi", f"{settings.get('final_threshold', 0) * 100:.0f}", f"Sedang ≥ {settings.get('watch_threshold', 0) * 100:.0f}",
         ui.RED),
        ("Recall test 2026", ui.percent(settings.get("test_recall", float("nan"))), "kolusi tertangkap di band Tinggi", ui.GREEN),
    ])
    st.markdown(pillars.pillar_legend(), unsafe_allow_html=True)
    tabs = st.tabs(["🧭 Cara kerja", "⚖️ Bobot & kontribusi", "📊 v9 · v9.x · v10", "🧪 Ablation & ketangguhan",
                    "🎯 Kalibrasi & threshold"])
    with tabs[0]:
        _render_how_it_works(settings)
    with tabs[1]:
        _render_weights(settings)
    with tabs[2]:
        _render_versions(settings)
    with tabs[3]:
        _render_ablation()
    with tabs[4]:
        _render_calibration(settings)
