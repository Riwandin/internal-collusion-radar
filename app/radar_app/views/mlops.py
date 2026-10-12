"""Manajemen · Tata Kelola & Info Model: model card, hyperparameter, kurva kalibrasi, riwayat seleksi offline."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..config import SETTINGS
from ..ui import components as ui
from ..ui import services


def render() -> None:
    ui.page_header(
        "Tata Kelola & Info Model",
        "Spesifikasi model machine learning yang aktif, silsilah data (lineage), hasil tuning, dan kebijakan tata kelola AI."
    )

    card = services.get_store().json_document("model_card")
    settings = services.app_settings()

    ui.kpi_row([
        ("Model Transaksi", card.get("algoritma", "Random Forest"), card.get("kalibrasi", "isotonic"), ui.BLUE),
        ("Threshold Tinggi", f"{card.get('threshold_tinggi', 0.4762) * 100:.1f}", f"Sedang ≥ {card.get('threshold_sedang', 0.2381) * 100:.1f}", ui.ORANGE),
        ("PR-AUC Test", f"{card.get('metrik_test', {}).get('PR-AUC', 0.903):.3f}", "data uji 2026", ui.GREEN),
        ("Independensi IF", "Terjamin (Tidak di-ensemble)", "unit analisis berbeda", ui.PURPLE)
    ])

    tab_card, tab_tuning, tab_lineage, tab_features = st.tabs([
        "📋 Model Card & Metrik",
        "🔬 Riwayat Tuning & Eksperimen",
        "🔄 Silsilah Data (Lineage)",
        "📖 Kamus Fitur Model"
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
                "Tipe": ["Kategorikal" if f in ["loan_product", "request_purpose", "business_sector", "debtor_type", "approver_position"] else "Numerik / Sinyal" for f in card.get("fitur", [])]
            })
            st.dataframe(feat_list, hide_index=True, use_container_width=True)
