"""Manajemen · Simulasi Dampak Bisnis & Kebijakan: manfaat bersih finansial (Rp), kapasitas investigator, dan stress testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .stress_test import GRAPH_FEATURES, apply_evasion, recall_precision


def _render_financial_impact(settings: dict) -> None:
    st.markdown("##### Parameter Asumsi Finansial & Operasional")
    p1, p2, p3, p4, p5 = st.columns(5)
    lgd = p1.slider("LGD (kerugian saat macet)", 0.2, 1.0, float(settings.get("lgd", 0.6)), 0.05)
    prevention = p2.slider("% alert benar dicegah", 0.2, 1.0, float(settings.get("prevention_rate", 0.75)), 0.05)
    hours = p3.slider("Jam investigasi / alert", 0.5, 8.0, float(settings.get("audit_hours_per_alert", 2.0)), 0.5)
    cost = p4.number_input("Biaya/jam investigator (Rp)", 50_000, 1_000_000, int(settings.get("investigator_cost_per_hour", 150_000)), 10_000)
    system_cost = p5.number_input("Biaya sistem/tahun (Rp)", 0, 20_000_000_000, 750_000_000, 50_000_000)
    base_loss = settings.get("loss_per_fraud", 0) / max(settings.get("lgd", 0.6), 1e-9)
    fraud_per_year = settings.get("fraud_cases_per_year", 0)
    scenarios = {
        "Rules saja": (settings.get("rules_test_recall", 0), settings.get("rules_alerts_per_month", 0)),
        "Model final": (settings.get("test_recall", 0), settings.get("model_alerts_per_month", 0)),
    }
    rows = []
    for name, (recall, alerts) in scenarios.items():
        prevented = fraud_per_year * recall * prevention * base_loss * lgd
        review = alerts * 12 * hours * cost
        extra = system_cost if name == "Model final" else 0
        rows.append({
            "skenario": name,
            "recall": recall,
            "alert/bulan": alerts,
            "kerugian dicegah": prevented,
            "biaya pemeriksaan": review,
            "biaya sistem": extra,
            "manfaat bersih": prevented - review - extra,
            "FTE investigator": alerts * hours / 120,
        })
    table = pd.DataFrame(rows).set_index("skenario")
    model, rules = table.loc["Model final"], table.loc["Rules saja"]
    roi = model["manfaat bersih"] / max(model["biaya pemeriksaan"] + model["biaya sistem"], 1)

    ui.kpi_row([
        ("Kerugian dicegah/tahun", ui.rupiah(model["kerugian dicegah"]), f"{fraud_per_year:,.0f} kasus/thn × recall {ui.percent(model['recall'])}", ui.GREEN),
        ("Biaya pemeriksaan/tahun", ui.rupiah(model["biaya pemeriksaan"]), f"{model['alert/bulan']:.0f} alert/bln × {hours} jam", ui.YELLOW),
        ("Manfaat bersih/tahun", ui.rupiah(model["manfaat bersih"]), "setelah biaya sistem", ui.BLUE),
        ("ROI", f"{roi:.1f}×", "manfaat bersih ÷ biaya", ui.PURPLE),
        ("Hemat jam investigasi", f"{ui.number((rules['alert/bulan'] - model['alert/bulan']) * hours * 12)} jam/thn", "dibanding rules", ui.ORANGE),
    ])

    left, right = st.columns([1.2, 1])
    figure = go.Figure()
    for column, color in [
        ("kerugian dicegah", ui.GREEN),
        ("biaya pemeriksaan", ui.YELLOW),
        ("biaya sistem", ui.MUTED),
        ("manfaat bersih", ui.BLUE),
    ]:
        figure.add_bar(x=table.index, y=table[column] / 1e9, name=column, marker_color=color)
    figure.update_layout(barmode="group", yaxis_title="Rp miliar / tahun")
    left.plotly_chart(ui.style_figure(figure, "Perbandingan finansial: Model vs Rules saja", "berdasarkan parameter di atas"), use_container_width=True)

    grid = pd.DataFrame(
        [[fraud_per_year * model["recall"] * p * base_loss * l - model["biaya pemeriksaan"] - model["biaya sistem"]
          for l in (0.4, 0.6, 0.8)] for p in (0.5, 0.75, 1.0)],
        index=["50% dicegah", "75% dicegah", "100% dicegah"],
        columns=["LGD 40%", "LGD 60%", "LGD 80%"],
    )
    figure = px.imshow(
        grid / 1e9, text_auto=".1f", color_continuous_scale=["#fbe7b5", ui.GREEN], aspect="auto",
        labels=dict(color="Rp M")
    )
    right.plotly_chart(ui.style_figure(figure, "Sensitivitas manfaat bersih (Rp miliar)", "tetap positif di skenario pesimistis?"), use_container_width=True)

    st.dataframe(
        table.style.format({
            "recall": "{:.0%}", "alert/bulan": "{:,.0f}", "kerugian dicegah": ui.rupiah,
            "biaya pemeriksaan": ui.rupiah, "biaya sistem": ui.rupiah, "manfaat bersih": ui.rupiah,
            "FTE investigator": "{:.1f}",
        }),
        use_container_width=True,
    )

    st.markdown("#### Dampak Non-Finansial & Kepatuhan Strategi Anti-Fraud (POJK 12/2024)")
    c1, c2 = st.columns(2)
    c1.markdown("""
- **Deteksi lebih cepat**: dari ±8 bulan setelah cair (saat NPL) menjadi sesaat setelah approval.
- **Efisiensi investigator**: alarm palsu jauh lebih sedikit dibanding rules; antrean berprioritas.
- **Kualitas bukti**: alasan bahasa manusia + perbandingan rekan + peta jaringan + jejak login per alert.
- **Efek jera**: pola kolusi dipantau sistematis; prinsip empat mata diperkuat.""")
    c2.markdown("""
| Pilar POJK 12/2024 | Peran sistem |
|---|---|
| Pencegahan | efek jera, profil approver berisiko |
| Deteksi | rules + graph + IF + classifier |
| Investigasi & pelaporan | meja investigasi 360°, PDF dossier, draf memo (disetujui manusia) |
| Pemantauan & evaluasi | command center, stress test, drift PSI, LLM-as-judge |""")
    st.caption("Estimasi memakai data sintetis dan asumsi yang dinyatakan terbuka; validasi dengan data internal sebelum dipakai untuk keputusan anggaran.")


def _render_stress_testing(settings: dict) -> None:
    test = services.table("test_features")
    transactions = services.table("transactions_scored")
    population_2026 = transactions[transactions["event_time"].dt.year == 2026]
    months = max(population_2026["event_time"].dt.to_period("M").nunique(), 1)
    model = services.get_risk_model()

    tab_capacity, tab_evasion, tab_outage, tab_prevalence, tab_drift = st.tabs([
        "A. Kapasitas Tim & Threshold",
        "B. Pelaku Beradaptasi (Evasion)",
        "C. Pipeline Graph Gagal",
        "D. Prevalensi Berubah",
        "E. Drift Data (PSI)",
    ])
    with tab_capacity:
        c1, c2, c3, c4 = st.columns(4)
        investigators = c1.slider("Jumlah investigator", 1, 30, int(settings.get("investigators_available", 6)))
        hours = c2.slider("Jam investigasi/orang/bulan", 40, 160, int(settings.get("hours_per_investigator_month", 120)), step=10)
        per_alert = c3.slider("Jam per alert", 0.5, 8.0, float(settings.get("audit_hours_per_alert", 2.0)), step=0.5, key="st_per_alert")
        threshold = c4.slider("Skor minimum alert", 5, 95, int(round(settings.get("final_threshold", 0.5) * 100)), step=1) / 100
        capacity = investigators * hours / per_alert
        alerts_per_month = (population_2026["risk_probability"] >= threshold).sum() / months
        recall, precision = recall_precision(test["fraud_label"].to_numpy(), test["risk_probability"].to_numpy(), threshold)
        missed = int(((test["risk_probability"] < threshold) & (test["fraud_label"] == 1)).sum())
        ui.kpi_row([
            ("Kapasitas", f"{ui.number(capacity)} alert/bln", f"{investigators} orang × {hours} jam ÷ {per_alert} jam", ui.BLUE),
            ("Beban alert", f"{ui.number(alerts_per_month)} /bln", "populasi 2026", ui.ORANGE if alerts_per_month > capacity else ui.GREEN),
            ("Utilisasi", ui.percent(alerts_per_month / capacity), "beban ÷ kapasitas", ui.RED if alerts_per_month > capacity else ui.GREEN),
            ("Recall (test)", ui.percent(recall), f"{missed} kasus kolusi lolos", ui.PURPLE),
            ("Precision (test)", ui.percent(precision), "di kasus yang direview", ui.YELLOW),
        ])
        if alerts_per_month > capacity:
            st.warning("Beban melebihi kapasitas: naikkan skor minimum, tambah investigator, atau gunakan antrean berprioritas (skor tertinggi dulu).")
        grid = np.arange(0.05, 0.96, 0.01)
        sweep = pd.DataFrame({
            "skor minimum": grid * 100,
            "alert/bulan": [(population_2026["risk_probability"] >= t).sum() / months for t in grid],
            "recall": [recall_precision(test["fraud_label"].to_numpy(), test["risk_probability"].to_numpy(), t)[0] for t in grid],
        })
        figure = go.Figure()
        figure.add_scatter(x=sweep["skor minimum"], y=sweep["alert/bulan"], name="alert/bulan", line=dict(color=ui.BLUE, width=3))
        figure.add_scatter(x=sweep["skor minimum"], y=sweep["recall"] * 100, name="recall (%)", yaxis="y2", line=dict(color=ui.GREEN, width=3))
        figure.add_hline(y=capacity, line_dash="dash", line_color=ui.ORANGE, annotation_text="kapasitas")
        figure.add_vline(x=threshold * 100, line_dash="dot", line_color=ui.INK)
        figure.update_layout(
            yaxis=dict(title="alert per bulan"),
            yaxis2=dict(title="recall (%)", overlaying="y", side="right", range=[0, 105]),
            xaxis_title="skor minimum",
        )
        st.plotly_chart(ui.style_figure(figure, "Trade-off beban kerja vs kolusi yang tertangkap"), use_container_width=True)

    with tab_evasion:
        st.markdown("Pelaku yang tahu rules bisa menghindarinya. Pilih perilaku yang dihindari dan berapa persen pelaku yang beradaptasi; "
                    "model menilai ulang kasus test 2026 secara langsung.")
        c1, c2, c3, c4 = st.columns(4)
        hide_near = c1.checkbox("Hindari near-limit (nominal 80–93%)", value=True)
        hide_night = c2.checkbox("Hindari jam malam (approve 09–16)", value=True)
        hide_fast = c3.checkbox("Hindari approval kilat (+30–180 mnt)", value=True)
        share = c4.slider("% pelaku beradaptasi", 0, 100, 100, step=10) / 100
        threshold = settings.get("final_threshold", 0.5)
        rules_threshold = settings.get("rules_threshold", 1)
        if model is None:
            st.info("Model tidak bisa dimuat di lingkungan ini; menampilkan hasil simulasi dari notebook.")
            st.dataframe(services.table("stress_evasion"), hide_index=True, use_container_width=True)
        else:
            adapted = apply_evasion(test, hide_near, hide_night, hide_fast, share)
            new_probability = model.predict_proba(adapted)
            positives = adapted["fraud_label"] == 1
            result = pd.DataFrame({
                "pendekatan": ["Rules", "Model (rules + graph + riwayat)"],
                "recall sebelum": [
                    (test.loc[positives, "rule_score"] >= rules_threshold).mean(),
                    (test.loc[positives, "risk_probability"] >= threshold).mean(),
                ],
                "recall sesudah": [
                    (adapted.loc[positives, "rule_score"] >= rules_threshold).mean(),
                    (new_probability[positives.to_numpy()] >= threshold).mean(),
                ],
            })
            figure = go.Figure()
            figure.add_bar(x=result["pendekatan"], y=result["recall sebelum"] * 100, name="sebelum", marker_color=ui.MUTED)
            figure.add_bar(x=result["pendekatan"], y=result["recall sesudah"] * 100, name="sesudah pelaku beradaptasi", marker_color=ui.ORANGE)
            figure.update_layout(barmode="group", yaxis_title="recall (%)", yaxis_range=[0, 105])
            st.plotly_chart(ui.style_figure(figure, "Rules runtuh, model bertahan karena pola hubungan sulit disembunyikan"), use_container_width=True)
            st.caption("Dalam jangka panjang fitur graph juga ikut melemah karena dibangun dari red flag; karena itu model dilatih ulang berkala.")

    with tab_outage:
        st.dataframe(services.table("stress_graph_outage"), hide_index=True, use_container_width=True)
        if model is not None and st.button("Hitung ulang langsung: kosongkan semua fitur graph"):
            outage = test.copy()
            outage[GRAPH_FEATURES] = np.nan
            probability = model.predict_proba(outage)
            recall, precision = recall_precision(test["fraud_label"].to_numpy(), probability, settings.get("final_threshold", 0.5))
            st.metric("Recall tanpa fitur graph", ui.percent(recall), f"{(recall - settings.get('test_recall', recall)) * 100:+.0f} poin")
        st.caption("Sistem tetap berjalan (imputasi), tetapi kinerja turun: kegagalan pipeline graph harus memicu alarm monitoring.")

    with tab_prevalence:
        multiplier = st.slider("Proporsi kolusi dibanding data uji (×)", 0.1, 3.0, 1.0, step=0.1)
        threshold = settings.get("final_threshold", 0.5)
        labels, scores = test["fraud_label"].to_numpy(), test["risk_probability"].to_numpy()
        predicted = scores >= threshold
        tp, fp = (predicted & (labels == 1)).sum(), (predicted & (labels == 0)).sum() / multiplier
        rules_predicted = test["rule_score"].to_numpy() >= settings.get("rules_threshold", 1)
        rules_tp, rules_fp = (rules_predicted & (labels == 1)).sum(), (rules_predicted & (labels == 0)).sum() / multiplier
        ui.kpi_row([
            ("Precision model", ui.percent(tp / max(tp + fp, 1e-9)), "recall tidak berubah", ui.ORANGE),
            ("Precision rules", ui.percent(rules_tp / max(rules_tp + rules_fp, 1e-9)), "", ui.MUTED),
        ])
        st.dataframe(services.table("stress_prevalence"), hide_index=True, use_container_width=True)

    with tab_drift:
        drift = services.table("drift_psi")
        figure = px.bar(
            drift.sort_values("PSI 2026 vs train"), x="PSI 2026 vs train", y="fitur", orientation="h", color="status",
            color_discrete_map={"stabil": ui.GREEN, "pantau": ui.YELLOW, "bergeser": ui.ORANGE},
        )
        figure.add_vline(x=0.1, line_dash="dot", line_color=ui.YELLOW)
        figure.add_vline(x=0.25, line_dash="dash", line_color=ui.ORANGE)
        st.plotly_chart(
            ui.style_figure(figure, "Population Stability Index fitur utama (2026 vs data latih)",
                            "< 0,1 stabil · 0,1–0,25 pantau · > 0,25 bergeser → evaluasi & latih ulang"),
            use_container_width=True,
        )


def render() -> None:
    ui.page_header(
        "Simulasi Dampak Bisnis & Kebijakan",
        "Hitung manfaat bersih finansial (Rp), atur kapasitas tim investigator & threshold, serta uji ketahanan sistem (stress testing)",
    )
    settings = services.app_settings()
    tab_roi, tab_stress = st.tabs([
        "💰 Simulasi Manfaat Bersih & ROI (Rp)",
        "⚙️ Kapasitas Tim Investigator & Stress Testing",
    ])
    with tab_roi:
        _render_financial_impact(settings)
    with tab_stress:
        _render_stress_testing(settings)
