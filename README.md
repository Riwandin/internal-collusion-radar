# Internal Collusion Radar v10 — Deteksi Indikasi Kolusi pada Persetujuan KUR

Capstone ODP Data Analytics Batch 2. Sistem end-to-end untuk mendeteksi indikasi kolusi **requester ↔ approver**
pada proses persetujuan Kredit Usaha Rakyat: data → EDA → 5 rules → graph analytics → **Risk Engine 4 pilar**
(Rules · Isolation Forest · model konten transaksi · model relasi & jaringan, digabung dengan *stacking*) → SHAP 2 tingkat
→ dashboard dua role → asisten AI (RAG + agen) dengan guardrail dan LLM-as-judge.

> **Data sintetis.** Semua pegawai, debitur, dan kasus fiktif (generator v6.1, `data_generator/`). Angka di bawah
> berasal dari data sintetis dan **bukan** estimasi kondisi bank mana pun.

---

## 1. Isi paket

| Folder / file | Isi |
|---|---|
| `notebooks/Internal_Collusion_Radar_EndToEnd_v9.ipynb` | Notebook utama **v10** (Bagian 0–12, nama file dipertahankan), sudah dieksekusi beserta output |
| `data/raw/` | 7 tabel sintetis (dibangkitkan ulang dengan `python data_generator/generate_kur_dataset.py --out data/raw`, seed 2026) |
| `data_generator/` | Generator dataset v6.1 + validator |
| `outputs/app_data/` | Hasil notebook untuk dashboard (CSV + JSON), termasuk `fusion_weights`, `fusion_comparison`, `pillar_correlation`, `ablation` |
| `outputs/models/` | `transaction_risk_model.joblib` (**MultiViewRiskEngine**: 2 model pilar terkalibrasi + penggabung), `approver_anomaly_model.joblib` (Isolation Forest + distribusi acuan), `model_card.json` |
| `outputs/rag_index/` | Indeks RAG (chunk, vektor jendela kalimat, konfigurasi k hasil tuning) |
| `knowledge_base/` | 11 dokumen acuan RAG (KUR, POJK, UU PDP, ACFE, tipologi, SOP, panduan membaca skor v10, tata kelola) + set evaluasi |
| `app/` | Aplikasi Streamlit (`streamlit_app.py`, paket `radar_app/`; engine di `radar_app/ml_objects.py`) |
| `scripts/` | Muat ke PostgreSQL, bootstrap akun, bangun indeks RAG, evaluasi GenAI offline |
| `docker-compose.yml`, `docker/`, `db/` | Infrastruktur: PostgreSQL, MLflow, Ollama, app, JupyterLab (opsional), query DBeaver |

## 2. Arsitektur

```
 data/raw (7 tabel) ─► clean → fitur as-of → 5 rules → GRAPH bulanan (tanpa label)
                                    │
      ┌──────────────┬──────────────┼──────────────────┬───────────────────────────┐
      ▼              ▼              ▼                  ▼                           │
 PILAR 1 Rules  PILAR 2 Isolation  PILAR 3 Konten     PILAR 4 Relasi & jaringan    │   knowledge_base/*.md
 S_rule         Forest (approver-  transaksi          (pasangan, routing, g_*)     │   MiniLM + FAISS + BM25 → RRF
 (tanpa label)  bulan, fit ≤2024)  (PyCaret+Optuna,   (PyCaret+Optuna, isotonic)   │   Gemma 4 E4B (Ollama)
                S_IF               isotonic) S_cls    S_graph                      │   + guardrail 4 lapis
      └──────────────┴──────────────┬──────────────────┘                           │   + agen ber-tools (role)
                                    ▼                                              │   + LLM-as-judge (Qwen2.5)
            STACKING: logistic regression, bobot dipelajari di validation 2025     │
            logit(R) = dasar + kontribusi Rules + IF + Konten + Relasi (eksak)     │
                                    │ CSV / joblib / MLflow registry (4 komponen)  │
                                    ▼                                              ▼
       PostgreSQL (schema radar = hasil analitik, app = keputusan, log audit AI) ◀── DBeaver
                                    ▼
       Streamlit (login investigator / manajemen, light mode)
```

**Aturan desain (diuji dengan `assert` di notebook 4.5)**

| Aturan | Kenapa |
|---|---|
| Setiap kolom data hanya masuk **satu** pilar yang belajar; flag rules bukan fitur model; IF tanpa fitur graph | mencegah *double counting* — sinyal yang sama terhitung berkali-kali di skor akhir |
| Pilar 3 & 4 dilatih di train 2021–2024; IF di-fit pada approver-bulan s.d. 2024 | skor pilar di 2025–2026 out-of-sample, tidak ada kebocoran waktu |
| Bobot penggabung **dipelajari** di validation 2025; threshold dari prediksi *out-of-fold* | bobot manual adalah opini yang tidak teruji; threshold tidak dipilih di data yang melatih penggabung |
| Penggabung linear di log-odds | skor akhir bisa diurai **persis** menjadi kontribusi 4 pilar (dasar + Σ kontribusi = logit skor) |
| Penggabung **monoton** (koefisien pilar ≥ 0) | skor pilar yang naik tidak pernah menurunkan skor akhir; tanpa batasan, koefisien Rules sedikit negatif karena nilai mentahnya sudah dibaca Pilar 3 |
| Jaring pengaman: pilar berlabel dengan precision ≥ 90% sendirian → minimal band Sedang | satu sinyal kuat tidak tenggelam di skor gabungan |
| IF tetap tampil sebagai radar approver terpisah | label audit hanya berisi pola lama; radar menjaga pola baru tetap terlihat |

**Riwayat.** v9 menyajikan tiga skor terpisah tanpa penggabungan. Versi antara (v9.x) menggabungkan empat skor dengan bobot
manual 15/20/40/25 dan ambang 70/40, tetapi fitur graph & rules tetap ada di classifier/IF (*double counting*), bobot dan
ambang tidak divalidasi, dan IF yang kosong diisi rumus heuristik. v10 memperbaiki semuanya; perbandingannya di bawah.

## 3. Hasil utama (data sintetis, test = 2026, threshold dikunci di validation 2025)

| Komponen | Hasil |
|---|---|
| **Risk Engine 4 pilar (final)** | precision 0,88 · recall 0,90 · F1 0,89 · PR-AUC 0,912 · 77 alarm palsu · 66 kolusi lolos · MCC 0,87 · CI bootstrap 95% PR-AUC 0,887–0,936 |
| Bobot efektif pilar (dipelajari) | Rules 0% · IF 16% · Konten 49% · Relasi 34% (bandingkan manual v9.x 15/20/40/25) |
| Threshold | Tinggi ≥ 35 · Sedang ≥ 17 (Tinggi+Sedang menjaring 93% kolusi validation) |
| Pembanding: stacking tanpa batasan monoton | precision 0,88 · recall 0,89 · F1 0,89 · PR-AUC 0,912 · 76 alarm palsu · 68 kolusi lolos |
| Pembanding: satu model 40 fitur (desain v9) | precision 0,81 · recall 0,94 · F1 0,87 · PR-AUC 0,896 · 141 alarm palsu · 40 kolusi lolos |
| Pembanding: bobot manual + ambang 70 (v9.x) | precision 0,95 · recall 0,55 · F1 0,69 · PR-AUC 0,909 · 18 alarm palsu · 285 kolusi lolos |
| Pembanding: Logistic Regression (40 fitur) | precision 0,77 · recall 0,92 · F1 0,84 · PR-AUC 0,902 · 168 alarm palsu · 52 kolusi lolos |
| Pembanding: Rules (≥1 rule) | precision 0,53 · recall 0,85 · F1 0,65 · PR-AUC 0,598 · 474 alarm palsu · 97 kolusi lolos |
| Tiap pilar sendirian (PR-AUC test) | IF 0,410 · Konten 0,800 · Relasi 0,772 |
| Isolation Forest (approver-bulan 2025–2026, out-of-sample) | PR-AUC 0,694 · precision@5% 0,87 (acak 0,18); versi v9 dengan fitur graph: PR-AUC 0,695 |
| Graph (snapshot bulanan) | 32 komunitas risiko · modularity 0,97 · persistensi 0,65 · peringatan dini precision 32% (lift 6,5×) |
| RAG (33 pertanyaan) | Hit@4 1,00 · MRR 0,93 · 38 chunk · k optimal dense 8, BM25 5 |
| Guardrail (34 kasus red-team) | 22/22 serangan tertangani · 0/12 pertanyaan wajar terblokir |
| Agen (14 skenario) | 14/14 benar (tools tepat, penolakan sesuai wewenang role) |

**Membaca dengan jujur.**
- Stacking 4 pilar **setara** dengan satu model berisi semua fitur dalam kemampuan mengurutkan (PR-AUC 0,912
  vs 0,896). Nilai tambahnya ada pada skor yang **terkalibrasi**, **bisa diurai per pilar**, dan
  tetap jalan saat satu sumber data gagal — bukan pada PR-AUC yang jauh lebih tinggi.
- Bobot manual dengan ambang 70 yang tidak divalidasi meloloskan 285 kasus kolusi di test, dibanding
  66 pada Risk Engine. Rata-rata berbobot juga bukan peluang (lihat reliability diagram).
- Lompatan terbesar tetap dari **Rules → ML**: alarm palsu −84% (474 → 77).
- Ablation (perubahan PR-AUC test bila satu pilar dimatikan): Rules SOP +0,000, Anomali approver (IF) -0,000, Konten transaksi -0,024, Relasi & jaringan -0,066.
  Pilar yang perubahannya ±0 tidak banyak menambah kemampuan mengurutkan, tetapi tetap dipertahankan: rules sebagai alasan
  yang dipahami auditor, IF sebagai radar individu untuk pola yang belum berlabel dan cadangan saat sumber lain gagal.
- IF dan skor transaksi saling melengkapi (Spearman 0,49); approver-bulan yang **hanya** ditandai
  IF: 76% terbukti menyetujui kasus kolusi.
- Stress test: bila pelaku menghindari semua rule transaksi, recall rules 25%
  vs Risk Engine 85%. Bila fitur graph hilang: PR-AUC
  0,859; bila skor IF hilang: 0,909.
- Dampak (asumsi dapat diubah di dashboard): manfaat bersih ±Rp11,9 M/tahun dengan beban
  investigasi −64% dibanding rules.

## 4. Menjalankan dengan Docker (disarankan)

Prasyarat: Docker Desktop / Docker Engine + Compose v2, RAM ≥ 16 GB dan ruang disk ±20 GB untuk tiga model Ollama (GPU opsional).

```bash
cp .env.example .env            # ubah password sebelum dipakai di luar demo
docker compose up -d --build    # postgres, mlflow, ollama (+ unduh 3 model), app
docker compose logs -f ollama-pull   # tunggu sampai ketiga model selesai diunduh
```

| Layanan | Alamat | Keterangan |
|---|---|---|
| Dashboard | http://localhost:8501 | login `investigator` / `investigator123` atau `manajemen` / `manajemen123` |
| MLflow | http://localhost:5000 | experiment `kur-collusion-radar`, model `kur_approver_iforest`, `kur_collusion_classifier` (Pilar 3), `kur_relational_model` (Pilar 4), `kur_fusion_stacker` |
| PostgreSQL | localhost:5432, db `radar` | buka di **DBeaver**: New Connection → PostgreSQL → user/password dari `.env`. Query siap pakai: `db/dbeaver_queries.sql` |
| Ollama | http://localhost:11434 | `gemma4:e4b` (generator), `llama-guard3:1b` (guardrail), `qwen2.5:7b-instruct` (juri) |

Saat container app pertama kali jalan: CSV hasil notebook dimuat ke schema `radar`, akun dibuat (password di-hash
PBKDF2), lalu indeks RAG disiapkan. Tanpa Ollama, asisten tetap jalan dalam **mode cadangan** (jawaban ekstraktif dari
sumber, planner agen berbasis aturan, juri heuristik), dan setiap jawaban menyebut mode yang dipakai.

### Melatih ulang di lingkungan yang sama (penting)

```bash
docker compose --profile lab up -d lab     # JupyterLab: http://localhost:8888 (token: radar)
# buka notebooks/Internal_Collusion_Radar_EndToEnd_v9.ipynb → Run All (±30–60 menit dengan PyCaret + Optuna)
docker compose exec app python scripts/load_to_postgres.py      # muat ulang hasil ke PostgreSQL
docker compose exec app python scripts/build_rag_index.py --force
docker compose restart app
```

File model `.joblib` harus dibuat dengan versi scikit-learn/pandas yang **sama** dengan aplikasi
(`app/requirements.txt`). Notebook terlampir dieksekusi di lingkungan pengembangan yang berbeda (lihat §7), jadi
jalankan ulang notebook di container `lab` sebelum demo agar model bisa dimuat aplikasi. Bila model tidak bisa dimuat,
aplikasi tetap menampilkan skor yang sudah dihitung notebook dan hanya menonaktifkan rescoring langsung (stress test).

## 5. Menjalankan tanpa Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r notebooks/requirements-notebook.txt     # notebook (PyCaret butuh Python 3.11)
cd notebooks && jupyter lab                            # Run All
cd .. && pip install -r app/requirements.txt
streamlit run app/streamlit_app.py                     # tanpa DATABASE_URL → membaca CSV di outputs/app_data
```

Opsional: `ollama pull gemma4:e4b llama-guard3:1b qwen2.5:7b-instruct` dan `mlflow server --port 5000`
(set `MLFLOW_TRACKING_URI=http://localhost:5000` sebelum menjalankan notebook).

## 6. Peta halaman dashboard

| Role | Halaman | Tugas utama |
|---|---|---|
| Investigator | Meja Investigasi 360° | antrean pegawai/alert berprioritas; dossier: skor akhir + **4 skor pilar + waterfall kontribusi**, alasan per pilar, bukti, jaringan, keputusan, PDF, draf memo agen |
| | Peta Jaringan & Komplotan | komunitas risiko (skor, anggota bernama, PageRank), graf interaktif, skor anomali IF anggota |
| | Cek Berkas Baru (Live Test) | mulai dari kasus nyata 2026, ubah input per pilar (*what-if*) → Risk Engine menilai ulang langsung; batch scoring CSV |
| | **Risk Engine 4 Pilar** | cara kerja, bobot dipelajari vs manual, perbandingan v9 · v9.x · v10, ablation, ketangguhan, kalibrasi & threshold |
| | Copilot AI & Regulasi KUR | RAG + agen ber-tools (detail alert memuat skor & kontribusi pilar) |
| Manajemen | Command Center & Peta Cabang | KPI, eksposur, tren alert vs rules, wilayah/cabang, **pilar penyebab alert per cabang** |
| | Simulasi Dampak & Kebijakan | manfaat bersih (slider), kapasitas & threshold, evasion (kontribusi pilar sebelum/sesudah), kegagalan graph/IF, prevalensi, drift (fitur & skor pilar) |
| | **Risk Engine 4 Pilar** | sama seperti investigator |
| | Kesehatan Sistem & Tata Kelola | performa Risk Engine & tiap pilar, SHAP 2 tingkat, model card (4 komponen), lineage, audit AI |
| | Copilot AI & Regulasi KUR | sama + tambah dokumen knowledge base |

Jalankan dashboard dengan interpreter yang sama dengan saat notebook dijalankan (`python run_dashboard.py` memakai `.venv`
bila ada), karena file `.joblib` bergantung pada versi scikit-learn.

## 7. Catatan jujur tentang lingkungan pengembangan

- Notebook v10 terlampir dieksekusi ulang dari nol (data dibangkitkan ulang dengan seed yang sama) di Python 3.11 lokal
  dengan pustaka asli: seleksi model **PyCaret compare_models**, tuning **Optuna TPE + MedianPruner**,
  XAI **shap.TreeExplainer**, embedding **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**, MLflow (SQLite lokal
  `outputs/mlflow.db`, tidak di-commit).
- **Ollama tidak berjalan** saat eksekusi, sehingga angka GenAI (juri: heuristik deterministik (LLM juri tidak tersedia)) berasal dari **mode cadangan**.
  Jalankan `docker compose exec app python scripts/evaluate_genai.py` setelah model Ollama terunduh untuk skor dengan juri LLM.
- File `.joblib` dibuat dengan scikit-learn 1.4.2; buka dashboard dengan lingkungan yang sama
  (`python run_dashboard.py` memakai `.venv`). Interpreter dengan versi scikit-learn lain bisa gagal memuat Risk Engine;
  aplikasi lalu tetap menampilkan skor dari notebook tetapi Live Test dinonaktifkan.
- Bobot penggabung dipelajari dari label audit 2025. Pola kolusi baru yang belum berlabel cenderung diberi bobot kecil,
  karena itu radar IF dan daftar komunitas tetap disajikan terpisah.
- Set red-team (34 kasus) dan set evaluasi RAG masih kecil; perlakukan hasilnya sebagai uji fungsi, bukan jaminan
  keamanan.
- Ringkasan regulasi di `knowledge_base/` berasal dari sumber resmi yang dicek 11 Oktober 2026 dan diberi disclaimer;
  bukan teks lengkap peraturan.

## 8. Informasi tambahan yang akan memperkuat sistem

SOP investigasi internal dan matriks kewenangan memutus kredit (limit per jabatan), definisi jam kerja dan pola shift
resmi, daftar kasus fraud historis yang sudah diputus (untuk kalibrasi ulang), kebijakan retensi data dan klasifikasi
informasi, serta teks lengkap kebijakan anti-fraud bank. Dokumen tersebut bisa ditambahkan lewat halaman
Knowledge Base (role manajemen) tanpa mengubah kode.
