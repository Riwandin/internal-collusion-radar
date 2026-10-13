"""Manajemen · Kesehatan Sistem & Tata Kelola: performa model, analisis skor & SHAP global, model card & lineage, serta audit AI."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import parse_json_column
from .model_performance import classifier_tab


def _render_model_performance(settings: dict) -> None:
    tabs = st.tabs([
        "Classifier (Skor Transaksi)",
        "Isolation Forest (Radar Approver)",
        "Graph (Komunitas Risiko)",
        "Rules (Baseline)",
        "Fairness & Keadilan",
        "Seleksi & Tuning",
    ])
    with tabs[0]:
        classifier_tab(settings)
    with tabs[1]:
        radar = services.table("radar_evaluation")
        st.markdown("**Tanpa label:** kriteria Excess-Mass & stabilitas antar-seed (dipakai untuk tuning). **Dengan label (hanya mengukur):** "
                    "apakah approver yang terbukti menyetujui kasus kolusi berada di puncak radar?")
        st.dataframe(radar.set_index(["periode", "radar"]).style.format("{:.3f}"), use_container_width=True)
        left, right = st.columns(2)
        recent = radar[radar["periode"] == "2025–2026"]
        figure = px.bar(recent, x="radar", y="precision@5%", text_auto=".0%", color_discrete_sequence=[ui.PURPLE])
        figure.add_hline(y=recent["acak (prevalensi)"].iloc[0], line_dash="dot", annotation_text="acak")
        left.plotly_chart(ui.style_figure(figure, "Precision di 5% teratas (2025–2026)", height=340), use_container_width=True)
        stability = services.table("iforest_stability")
        right.markdown("**Stabilitas & Excess-Mass**")
        right.dataframe(stability, hide_index=True, use_container_width=True)
        complementarity = services.table("if_classifier_complementarity")
        right.markdown(f"**Saling melengkapi dengan classifier** (Spearman {settings.get('spearman_if_classifier', 0):.2f})")
        right.dataframe(complementarity.style.format({"terbukti_menyetujui_kolusi": "{:.0%}"}), hide_index=True, use_container_width=True)
    with tabs[2]:
        graph = services.table("graph_evaluation")
        figure = go.Figure()
        for column, color in [("modularity", ui.BLUE), ("persistensi", ui.GREEN), ("precision", ui.ORANGE), ("recall", ui.PURPLE)]:
            figure.add_scatter(x=graph["bulan"], y=graph[column], name=column, line=dict(color=color, width=2))
        st.plotly_chart(
            ui.style_figure(figure, "Kualitas struktur & daya peringatan dini komunitas",
                            "modularity & persistensi tanpa label; precision/recall = anggota yang terlibat kasus terbukti ≤ 3 bulan"),
            use_container_width=True,
        )
        recent = graph[graph["bulan"].astype(str) >= "2025-01"]
        ui.kpi_row([
            ("Modularity", f"{recent['modularity'].mean():.2f}", "rata-rata 2025–2026", ui.BLUE),
            ("Persistensi", f"{recent['persistensi'].mean():.2f}", "Jaccard bulan ke bulan", ui.GREEN),
            ("Precision peringatan dini", ui.percent(recent["precision"].mean()), "", ui.ORANGE),
            ("Lift", f"{recent['lift'].mean():.1f}×", "dibanding pegawai acak", ui.PURPLE),
        ])
    with tabs[3]:
        st.dataframe(
            services.table("rule_definitions").style.format({"coverage": "{:.1%}", "precision": "{:.0%}", "recall": "{:.0%}"}),
            hide_index=True, use_container_width=True,
        )
        st.caption("Rules dipakai sebagai baseline dan alasan. Setiap rule punya kembaran sah, sehingga rules sendirian berisik.")
    with tabs[4]:
        fairness = services.table("fairness")
        dimension = st.selectbox("Dimensi", fairness["dimensi"].unique(), key="mlops_fairness_dim")
        subset = fairness[fairness["dimensi"] == dimension]
        figure = go.Figure()
        figure.add_bar(x=subset["segmen"], y=subset["FPR rules"], name="FPR rules", marker_color=ui.MUTED)
        figure.add_bar(x=subset["segmen"], y=subset["FPR model"], name="FPR model", marker_color=ui.ORANGE)
        figure.update_layout(barmode="group", yaxis_tickformat=".0%")
        st.plotly_chart(
            ui.style_figure(figure, f"False positive rate per {dimension}", "rendah & merata = adil bagi pegawai/debitur yang bekerja sah"),
            use_container_width=True,
        )
        st.dataframe(subset, hide_index=True, use_container_width=True)
    with tabs[5]:
        st.markdown(f"**Seleksi algoritma:** {settings.get('screening_engine', '-')} · **Tuning:** {settings.get('tuning_engine', '-')}")
        leaderboard = services.table("screening_leaderboard")
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)
        trials = services.table("tuning_trials")
        if len(trials) and "value" in trials:
            trials = trials.assign(percobaan=trials.groupby("model").cumcount())
            figure = px.scatter(trials, x="percobaan", y="value", color="model", labels={"value": "PR-AUC CV deret waktu"})
            st.plotly_chart(ui.style_figure(figure, "Riwayat percobaan tuning"), use_container_width=True)


def _render_score_and_shap(settings: dict) -> None:
    transactions = services.table("transactions_scored")
    period = services.period_filter(transactions, "mlops_score", default_year=2026)
    threshold = settings.get("final_threshold", 0.5) * 100
    tab_distribution, tab_calibration, tab_shap, tab_dependence = st.tabs([
        "Distribusi Skor",
        "Kalibrasi (Bisa Dipercaya?)",
        "SHAP Global & Beeswarm",
        "Pengaruh Fitur (Dependence)",
    ])
    with tab_distribution:
        left, right = st.columns(2)
        figure = px.histogram(
            period, x="risk_score", color="risk_band", nbins=50, color_discrete_map=ui.BAND_COLORS,
            log_y=True, labels={"risk_score": "skor risiko", "risk_band": "kategori"},
        )
        figure.add_vline(x=threshold, line_dash="dash", line_color=ui.ORANGE, annotation_text="threshold Tinggi")
        left.plotly_chart(
            ui.style_figure(figure, "Sebagian besar transaksi berskor rendah", "jumlah transaksi per skor (skala log)"),
            use_container_width=True,
        )
        figure = px.box(
            period.assign(rule=period["rule_score"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "≥3"})),
            x="rule", y="risk_score", color_discrete_sequence=[ui.BLUE], points=False,
            category_orders={"rule": ["0", "1", "2", "≥3"]}, labels={"rule": "jumlah rule terpenuhi", "risk_score": "skor"},
        )
        right.plotly_chart(
            ui.style_figure(figure, "Skor naik seiring jumlah rule, tapi tidak identik",
                            "model juga menimbang graph & riwayat; 0 rule pun bisa tinggi bila jaringannya berisiko"),
            use_container_width=True,
        )
    with tab_calibration:
        curves = services.table("model_curves")
        calibration = curves[curves["curve"] == "CALIBRATION"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=ui.MUTED), name="sempurna"))
        for name, color in [("mentah", ui.BLUE), ("terkalibrasi", ui.ORANGE)]:
            subset = calibration[calibration["model"] == name]
            figure.add_trace(go.Scatter(x=subset["x"], y=subset["y"], mode="lines+markers", name=name, line=dict(color=color, width=3)))
        figure.update_layout(xaxis_title="skor prediksi", yaxis_title="proporsi kolusi aktual (test 2026)")
        st.plotly_chart(
            ui.style_figure(figure, "Reliability diagram: skor ≈ peluang", "titik dekat diagonal = skor bisa dibaca sebagai peluang"),
            use_container_width=True,
        )
    shap_global = services.table("shap_global")
    beeswarm = services.table("shap_beeswarm")
    with tab_shap:
        left, right = st.columns([0.9, 1.2])
        top = shap_global.head(15).iloc[::-1]
        figure = go.Figure(go.Bar(
            x=top["mean |SHAP|"], y=top["feature"], orientation="h",
            marker_color=[ui.GROUP_COLORS.get(g, ui.BLUE) for g in top["kelompok"]],
            hovertext=top["arti"],
        ))
        left.plotly_chart(
            ui.style_figure(figure, "Fitur paling berpengaruh", "rata-rata |SHAP|; warna = kelompok fitur", height=480),
            use_container_width=True,
        )
        share = shap_global.groupby("kelompok")["mean |SHAP|"].sum()
        left.plotly_chart(
            ui.style_figure(px.pie(values=share.values, names=share.index, hole=.55, color=share.index, color_discrete_map=ui.GROUP_COLORS),
                            "Porsi pengaruh per kelompok", height=280),
            use_container_width=True,
        )
        features = top["feature"].tolist()[::-1]
        swarm = beeswarm[beeswarm["feature"].isin(features)].copy()
        swarm["persentil nilai"] = swarm.groupby("feature")["feature_value"].rank(pct=True)
        figure = px.scatter(
            swarm, x="shap_value", y="feature", color="persentil nilai", category_orders={"feature": features},
            color_continuous_scale="RdBu_r", hover_data=["feature_value", "fraud_label"], opacity=.6,
        )
        figure.add_vline(x=0, line_color=ui.MUTED)
        right.plotly_chart(
            ui.style_figure(figure, "Beeswarm: arah pengaruh", "merah = nilai fitur tinggi; kanan = menaikkan risiko", height=780),
            use_container_width=True,
        )
    with tab_dependence:
        feature = st.selectbox("Pilih fitur", shap_global["feature"].head(15).tolist(), key="mlops_dependence_feature")
        subset = beeswarm[beeswarm["feature"] == feature]
        figure = px.scatter(
            subset, x="feature_value", y="shap_value", color=subset["fraud_label"].map({0: "normal", 1: "kolusi"}),
            color_discrete_map={"normal": ui.BLUE, "kolusi": ui.ORANGE}, opacity=.6,
            labels={"feature_value": feature, "shap_value": "kontribusi SHAP", "color": "label audit"},
        )
        figure.add_hline(y=0, line_color=ui.MUTED)
        description = shap_global.set_index("feature").loc[feature, "arti"] if feature in set(shap_global["feature"]) else ""
        st.plotly_chart(ui.style_figure(figure, f"Pengaruh '{feature}' terhadap skor", description), use_container_width=True)


def _render_model_card_and_lineage(card: dict) -> None:
    tab_card, tab_tuning, tab_lineage, tab_features = st.tabs([
        "📋 Model Card & Metrik",
        "🔬 Riwayat Tuning & Eksperimen",
        "🔄 Silsilah Data (Lineage)",
        "📖 Kamus Fitur Model",
    ])
    with tab_card:
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("##### Identitas Model")
            info_data = [
                {"Atribut": "Nama Model", "Keterangan": card.get("nama", "KUR Collusion Radar — Transaction Risk Model")},
                {"Atribut": "Algoritma Utama", "Keterangan": card.get("algoritma", "Random Forest")},
                {"Atribut": "Metode Kalibrasi", "Keterangan": card.get("kalibrasi", "Isotonic Regression (Validation 2025)")},
                {"Atribut": "Jumlah Pohon (n_estimators)", "Keterangan": str(card.get("hyperparameter", {}).get("n_estimators", 300))},
                {"Atribut": "Min Samples Leaf", "Keterangan": str(card.get("hyperparameter", {}).get("min_samples_leaf", 2))},
                {"Atribut": "Fitur Digunakan", "Keterangan": f"{len(card.get('fitur', []))} fitur (finansial, waktu, rules, graph)"},
                {"Atribut": "Pemilik Model", "Keterangan": card.get("pemilik", "Unit Anti-Fraud Data Analytics")},
            ]
            st.dataframe(pd.DataFrame(info_data), hide_index=True, use_container_width=True)
            st.markdown("##### Batasan & Etika Model")
            for b in card.get("batasan", []):
                st.info(f"📌 {b}")
        with right:
            st.markdown("##### Metrik Kinerja pada Data Test (2026)")
            test_metrics = card.get("metrik_test", {})
            m_cols = st.columns(3)
            m_cols[0].metric("Precision", f"{test_metrics.get('precision', 0.864):.1%}")
            m_cols[1].metric("Recall", f"{test_metrics.get('recall', 0.910):.1%}")
            m_cols[2].metric("F1-Score", f"{test_metrics.get('F1', 0.886):.3f}")
            m_cols2 = st.columns(3)
            m_cols2[0].metric("PR-AUC", f"{test_metrics.get('PR-AUC', 0.903):.3f}")
            m_cols2[1].metric("ROC-AUC", f"{test_metrics.get('ROC-AUC', 0.974):.3f}")
            m_cols2[2].metric("MCC", f"{test_metrics.get('MCC', 0.869):.3f}")
            st.markdown("##### Model Card JSON")
            st.json(card, expanded=False)
    with tab_tuning:
        st.markdown("##### Hasil Eksperimen & Hyperparameter Tuning (Optuna & PyCaret)")
        runs_df = services.table("mlflow_runs")
        if not runs_df.empty:
            st.markdown("**Catatan Eksperimen Lokal (Export Notebook):**")
            if "metrics" in runs_df.columns:
                runs_clean = runs_df.assign(
                    metrik_singkat=runs_df["metrics"].map(
                        lambda m: ", ".join(f"{k}={v:.3f}" for k, v in json.loads(m).items() if isinstance(v, (int, float)))[:120]
                        if isinstance(m, str) else str(m)
                    )
                )
                show_cols = [c for c in ["run_name", "registered_model", "metrik_singkat"] if c in runs_clean.columns]
                st.dataframe(runs_clean[show_cols], hide_index=True, use_container_width=True)
            else:
                st.dataframe(runs_df, hide_index=True, use_container_width=True)
        tuning_df = services.table("tuning_trials")
        if not tuning_df.empty:
            st.markdown("**Percobaan Tuning Optuna:**")
            st.dataframe(tuning_df.head(15), hide_index=True, use_container_width=True)
    with tab_lineage:
        st.markdown("##### Alur Silsilah Data & Model (End-to-End Lineage)")
        st.graphviz_chart("""digraph { rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica"];
            data [label="7 Tabel Mentah\\n(Kredit, Log, Kasus)"];
            clean [label="Audit & Cleaning\\n(PII Masking)"];
            feat [label="Fitur As-Of\\n(5 Rules + Graph)"];
            split [label="Split Waktu\\nTrain 2021-24 / Val 2025 / Test 2026"];
            train [label="Random Forest\\n+ Isotonic Calibrator", fillcolor="#e7f6ef", color="#1baf7a"];
            iforest [label="Isolation Forest\\n(Approver-Bulan)", fillcolor="#e7f6ef", color="#1baf7a"];
            eval [label="Evaluasi Metrik\\nPR-AUC 0.903"];
            app [label="Streamlit Dashboard\\n(2 Role & Live Inference)", fillcolor="#fdf0e9", color="#eb6834"];

            data -> clean -> feat -> split;
            split -> train -> eval -> app;
            split -> iforest -> app;
        }""")
        st.markdown(
            "**Kebijakan Pelatihan Ulang (Retraining Policy):**  \n"
            "- Evaluasi drift berkala menggunakan *Population Stability Index* (PSI).  \n"
            "- Retraining dipicu jika PSI > 0.25 pada fitur utama atau setiap kuartal dengan data terkonfirmasi baru.  \n"
            "- Model Isolation Forest berjalan tanpa label (unsupervised) untuk mendeteksi anomali perilaku murni tanpa risiko bias konfirmasi."
        )
    with tab_features:
        st.markdown("##### Kamus Fitur Model Transaksi")
        feat_df = services.table("feature_dictionary")
        if not feat_df.empty:
            st.dataframe(feat_df, hide_index=True, use_container_width=True)
        else:
            feat_list = pd.DataFrame({
                "Nama Fitur": card.get("fitur", []),
                "Tipe": ["Kategorikal" if f in ["loan_product", "request_purpose", "business_sector", "debtor_type", "approver_position"] else "Numerik / Sinyal" for f in card.get("fitur", [])],
            })
            st.dataframe(feat_list, hide_index=True, use_container_width=True)


def _render_ai_governance() -> None:
    store = services.get_store()
    events = store.ai_events()
    judge_live = store.judge_results()
    total = len(events)
    blocked = events["blocked"].astype(str).str.lower().eq("true").sum() if total else 0
    fallback = events["fallback_used"].astype(str).str.lower().eq("true").sum() if total else 0
    ui.kpi_row([
        ("Interaksi AI", ui.number(total), f"sumber: {store.backend}", ui.BLUE),
        ("Diblokir guardrail", ui.number(blocked), ui.percent(blocked / total) if total else "-", ui.ORANGE),
        ("Mode cadangan", ui.number(fallback), "LLM tidak tersedia / grounding lemah", ui.YELLOW),
        ("Latensi median", f"{events['latency_ms'].median():.0f} ms" if total else "-", "", ui.GREEN),
        ("Penilaian juri (live)", ui.number(len(judge_live)), "dari tombol LLM-as-judge", ui.PURPLE),
    ])
    tab_log, tab_guard, tab_quality, tab_feedback = st.tabs([
        "Log AI", "Guardrail (4 Lapis)", "Kualitas Jawaban (LLM-as-Judge)", "Umpan Balik Investigator & Login"
    ])
    with tab_log:
        if total:
            frame = events.sort_values("logged_at", ascending=False)
            mode_filter = st.multiselect("Mode", sorted(frame["mode"].unique()), key="ai_mode_filter")
            if mode_filter:
                frame = frame[frame["mode"].isin(mode_filter)]
            st.dataframe(
                frame[["logged_at", "username", "role", "mode", "question_masked", "answer", "latency_ms", "blocked", "fallback_used", "tool_calls"]],
                hide_index=True, use_container_width=True, height=420,
            )
            st.caption("Pertanyaan disimpan dalam bentuk yang sudah di-masking (PII tidak tersimpan).")
        else:
            st.info("Belum ada interaksi AI. Coba halaman Copilot AI & Regulasi KUR.")
    with tab_guard:
        if total:
            checks = [check for value in events["guard_flags"] for check in parse_json_column(value)]
            frame = pd.DataFrame(checks)
            if len(frame):
                summary = frame.groupby(["layer", "check"])["passed"].agg(lambda s: (~s.astype(bool)).sum()).rename("gagal/ditahan").reset_index()
                figure = px.bar(summary.sort_values("gagal/ditahan"), x="gagal/ditahan", y="check", color="layer", orientation="h")
                st.plotly_chart(ui.style_figure(figure, "Pemeriksaan guardrail yang memblokir/menahan (live)"), use_container_width=True)
        offline = services.table("guardrail_eval")
        if len(offline):
            attacks = offline[offline["harus_diblok"].astype(str).str.lower().eq("true")]
            benign = offline[~offline["harus_diblok"].astype(str).str.lower().eq("true")]
            ui.kpi_row([
                ("Serangan tertangani (uji red-team)", ui.percent(attacks["benar"].astype(str).str.lower().eq("true").mean()), f"{len(attacks)} serangan", ui.GREEN),
                ("Pertanyaan wajar salah diblok", ui.percent(1 - benign["benar"].astype(str).str.lower().eq("true").mean()), f"{len(benign)} pertanyaan", ui.ORANGE),
            ])
            st.dataframe(offline, hide_index=True, use_container_width=True)
    with tab_quality:
        summary = services.table("judge_summary")
        genai = store.json_document("genai_summary")
        st.markdown(
            f"**Evaluasi offline** pada set uji RAG · juri: {genai.get('judge_engine', '-')} · retrieval Hit@4 "
            f"{ui.percent(genai.get('hit_at_4', 0))}, MRR {genai.get('mrr_at_4', 0):.2f} (dense_k={genai.get('dense_k')}, sparse_k={genai.get('sparse_k')})"
        )
        if len(summary):
            figure = px.bar(summary, x="skor rata-rata (1–5)", y="kriteria", orientation="h", text_auto=".2f", color_discrete_sequence=[ui.PURPLE])
            figure.update_layout(xaxis_range=[0, 5.2])
            st.plotly_chart(
                ui.style_figure(figure, "Skor juri per kriteria (1–5)", "faithfulness · relevance · correctness · citation · safety_tone"),
                use_container_width=True,
            )
        st.dataframe(services.table("judge_eval"), hide_index=True, use_container_width=True, height=300)
        if len(judge_live):
            st.markdown("**Penilaian live dari pengguna**")
            st.dataframe(judge_live.sort_values("judged_at", ascending=False), hide_index=True, use_container_width=True)
        agent_eval = services.table("agent_eval")
        if len(agent_eval):
            st.markdown(f"**Evaluasi agen:** ketepatan tool & kepatuhan wewenang {ui.percent(agent_eval['benar'].astype(str).str.lower().eq('true').mean())}")
            st.dataframe(agent_eval, hide_index=True, use_container_width=True)
    with tab_feedback:
        reviews = store.reviews()
        if reviews.empty:
            st.info("Belum ada keputusan investigator.")
        else:
            counts = reviews["decision"].value_counts().rename_axis("keputusan").reset_index(name="jumlah")
            st.plotly_chart(
                ui.style_figure(px.bar(counts, x="keputusan", y="jumlah", color_discrete_sequence=[ui.BLUE]),
                                "Keputusan investigator", "proporsi 'alarm palsu' adalah ukuran precision di lapangan"),
                use_container_width=True,
            )
            st.dataframe(reviews.sort_values("reviewed_at", ascending=False), hide_index=True, use_container_width=True)
        logins = store.logins()
        if len(logins):
            st.markdown("**Log login**")
            st.dataframe(logins.sort_values("logged_at", ascending=False).head(50), hide_index=True, use_container_width=True)


def render() -> None:
    ui.page_header(
        "Kesehatan Sistem & Tata Kelola",
        "Evaluasi metrik model (Classifier, Isolation Forest, Graph), kalibrasi & SHAP global, silsilah model, serta audit AI & guardrail",
    )
    card = services.get_store().json_document("model_card")
    settings = services.app_settings()

    ui.kpi_row([
        ("Model Transaksi", card.get("algoritma", "Random Forest"), card.get("kalibrasi", "isotonic"), ui.BLUE),
        ("Threshold Tinggi", f"{card.get('threshold_tinggi', 0.4762) * 100:.1f}", f"Sedang ≥ {card.get('threshold_sedang', 0.2381) * 100:.1f}", ui.ORANGE),
        ("PR-AUC Test", f"{card.get('metrik_test', {}).get('PR-AUC', 0.903):.3f}", "data uji 2026", ui.GREEN),
        ("Independensi IF", "Terjamin (Tidak di-ensemble)", "unit analisis berbeda", ui.PURPLE),
    ])

    t_perf, t_shap, t_card, t_ai = st.tabs([
        "📊 Performa & Evaluasi Model",
        "🔍 Analisis Skor & SHAP Global",
        "📋 Model Card, Tuning & Silsilah Data",
        "🛡️ Audit & Kualitas AI (Guardrail & Judge)",
    ])
    with t_perf:
        _render_model_performance(settings)
    with t_shap:
        _render_score_and_shap(settings)
    with t_card:
        _render_model_card_and_lineage(card)
    with t_ai:
        _render_ai_governance()
