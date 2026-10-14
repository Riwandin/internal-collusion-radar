---
doc_id: 07_kamus_rules_dan_fitur
title: Kamus 5 rules, fitur per pilar, dan fitur Isolation Forest pada Collusion Radar
category: internal
source: Notebook Internal Collusion Radar v10.1 (Bagian 1.8, 2.3, 2.3b, 2.4, 2.4b, 2.5, 2.5b)
url: internal://collusion-radar/kamus-fitur
last_verified: 2026-10-14
disclaimer: Definisi mengikuti konfigurasi notebook v10.1. Bila ambang diubah, dokumen ini harus diperbarui.
---

# Kamus rules dan fitur

## Lima rules dasar

Rules dihitung dari log transaksi tanpa memakai label. Ambangnya dikalibrasi sekali di data latih 2021–2024.

| Rule | Definisi | Alasan ambang | Kembaran sah |
|---|---|---|---|
| **R1 `rf_near_limit`** | nominal ≥ 95% dan < 100% limit kewenangan approver | 90% menambah alarm palsu; 100% (angka bulat) lazim | nominal bulat tepat di limit |
| **R2 `rf_after_hours`** | approval pukul 19:00–06:59 atau Sabtu/Minggu | jam 17–19 masih ramai lembur closing yang sah | lembur akhir bulan |
| **R3 `rf_fast_approval`** | approval kurang dari 20 menit setelah pengajuan | kompromi antara precision (10 menit) dan recall (60 menit) | top-up debitur lama |
| **R4 `rf_debtor_repeat`** | ≥ 2 pengajuan lain ke debitur yang sama dalam 1 tahun terakhir | 1×/tahun (perpanjangan, top-up) itu normal | perpanjangan tahunan |
| **R5 `rf_pair_dependency`** | `requester_approver_ratio` ≥ 10%, minimal 5 pengajuan pasangan setahun, dan ≥ 5× porsi approver itu di unit requester | rasio mentah tidak membedakan karena RM memang bergantung pada manager unitnya | Plt/delegasi cuti (sesekali) |

`rule_score` = jumlah rule yang terpenuhi (0–5). Rules adalah **Pilar 1** Risk Engine (S_rule = 20 × rule_score), sekaligus
baseline dan alasan yang dipahami auditor. Flag rules **tidak** dijadikan fitur model mana pun.

### Rumus indikator hubungan

- `requester_approver_ratio` = jumlah pengajuan pasangan (requester → approver) ÷ total pengajuan requester, dalam 1 tahun
  sebelum transaksi.
- `unit_approver_share` = porsi approver tersebut di seluruh pengajuan unit requester (1 tahun).
- `pair_dependency_lift` = `requester_approver_ratio` ÷ `unit_approver_share`. Lift 10× berarti requester memakai approver
  itu 10 kali lebih sering daripada kebiasaan rekan satu unitnya.
- `pair_approval_rate` = pengajuan pasangan yang disetujui ÷ seluruh keputusan pasangan (fitur Pilar 4, bukan rule).

## Pembagian fitur ke pilar (v10)

| Pilar | Kelompok | Fitur |
|---|---|---|
| 3 · Konten transaksi | Transaksi | `limit_ratio`, `approve_hour`, `approval_minutes`, `amount`, `tenor_months`, `bday_to_month_end`, produk, tujuan, sektor, tipe debitur, jabatan approver |
| 3 · Konten transaksi | Riwayat debitur | `debtor_tenure_days`, `is_new_debtor`, `debtor_loans_30d`, `debtor_loans_1y` |
| 4 · Relasi & jaringan | Relasi pasangan | `requester_prior_n`, `pair_prior_n`, `pair_share_of_approver`, `triad_prior_n`, `requester_approver_ratio`, `pair_dependency_lift`, `pair_approval_rate`, `routing_peer_share`, `same_unit` |
| 4 · Relasi & jaringan | Graph | 10 fitur `g_*` (tabel di bawah) |

Setiap fitur hanya masuk satu pilar sehingga sinyal yang sama tidak terhitung dua kali di skor akhir.

## Fitur graph per transaksi (input Pilar 4)

| Fitur | Arti |
|---|---|
| `g_pair_weight` | frekuensi hubungan requester–approver dalam 12 bulan |
| `g_appr_wdeg` | total transaksi approver dalam 12 bulan |
| `g_debtor_n_requesters` | jumlah requester berbeda yang melayani debitur ini |
| `g_req_shared_debtors` | jumlah debitur requester yang juga dilayani RM lain |
| `g_risk_pair_z` | z-score red flag pasangan (0 bila wajar) |
| `g_risk_pair_weight` | bobot hubungan di graph risiko (red flag berlebih + ketergantungan) |
| `g_risk_comm_score`, `g_risk_comm_size` | skor dan ukuran komunitas risiko tempat requester berada |
| `g_same_comm` | requester dan approver berada di komunitas risiko yang sama |
| `g_appr_risk_pagerank` | sentralitas approver di jaringan berisiko (PageRank) |

Graph dibangun per bulan dari approval 12 bulan sebelumnya, tanpa label. Hubungan disimpan di graph risiko bila red flag
transaksinya jauh di atas wajar (z ≥ 3 dan minimal 3 transaksi ber-red-flag) atau memenuhi rule ketergantungan.
Requester yang berbagi debitur juga dihubungkan. Komunitas dicari dengan algoritma Louvain.

## Graph heterogen temporal (v10.1)

Sejak v10.1 graph juga dibangun dengan tipe node dan relasi eksplisit: Employee (`EMP:`), Request (`REQ:`), Approval
(`APV:`), Debtor (`DEB:`), Unit (`UNIT:`), System (`SYS:`), dan LoanPerformance (`LP:`). Relasinya: SUBMITS
(pegawai → pengajuan), REQUESTED_BY_DEBTOR (pengajuan → debitur), HAS_APPROVAL (pengajuan → approval), DECIDED_BY
(approval → pegawai), BELONGS_TO (pegawai → unit, statis), ACCESSES (pegawai → sistem, agregat Login/View/Export), dan
HAS_PERFORMANCE (pengajuan → kinerja kredit, hanya untuk monitoring). Snapshot bulan m memakai kejadian 12 bulan sebelum
awal bulan m, dan setiap snapshot lolos uji otomatis: namespace, skema relasi, tidak ada timestamp ≥ cutoff, tanpa node
yatim/duplikat, tanpa label/PII, dan deterministik.

Fitur metapath kandidat Pilar 4 (dipakai hanya bila lolos uji di validation 2025; daftar yang terpilih ada di model card):

| Fitur | Arti |
|---|---|
| `h_debtor_n_approvers` | approver berbeda yang pernah memutus pengajuan debitur ini (12 bulan) |
| `h_debtor_n_units` | unit requester berbeda yang pernah melayani debitur ini |
| `h_shared_debtor_same_approver` | requester lain yang melayani debitur yang sama dan memakai approver yang sama (motif segitiga) |
| `h_req_n_approvers` | approver berbeda yang dipakai requester ini |
| `h_appr_n_units` | unit requester berbeda yang dilayani approver ini |
| `h_pair_risky_months` | berapa dari 12 snapshot terakhir pasangan ini menjadi hubungan berisiko (persistensi) |
| `h_unit_risky_pair_share` | porsi pasangan requester–approver dari unit requester yang berisiko |

**Bukti jaringan** (dipakai investigator; skornya = 100 × porsi bukti yang terpenuhi): pasangan berisiko, hubungan
berisiko bertahan ≥ 3 snapshot, requester & approver satu komunitas risiko, motif debitur bersama, dan approver ≥ 2×
lebih sentral dari rata-rata. Setiap bukti membawa batasan interpretasi: keterhubungan bukan bukti kolusi (pasangan kerja
tetap, mutasi portofolio, dan approver senior wajar terlihat terhubung).

Yang sengaja tidak dibuat: relasi akses → transaksi (log akses tidak punya `request_id`; log Approve/Edit cocok persis
dengan approval/pengajuan tetapi itu kejadian yang sama), node rekening/agunan/alamat (kolomnya tidak ada), dan
kolektibilitas sebagai fitur (snapshot akhir data = informasi masa depan). Riwayat NPL hanya tampil sebagai monitoring
pascapencairan di profil pegawai.

## Fitur Isolation Forest per approver-bulan

| Fitur | Arti |
|---|---|
| `ex_near_limit`, `ex_after_hours`, `ex_fast_approval`, `ex_debtor_repeat`, `ex_pair_dependency`, `ex_multi_flag` | kelebihan jumlah transaksi ber-rule dibanding rekan sejabatan (excess z-score, 90 hari) |
| `approve_rate` | persentase pengajuan yang disetujui |
| `top_requester_share` | porsi keputusan dari satu requester teratas |

Excess z-score = (jumlah teramati − jumlah wajar) ÷ simpangan baku binomial, dengan "wajar" dihitung dari rekan
sejabatan. Contoh: rekan sejabatan rata-rata 4% near-limit; approver dengan 40 keputusan wajarnya punya 1,6 transaksi
near-limit; bila ia punya 9, excess-nya sekitar +6 SD.

Fitur graph per approver (`ga_risky_pairs`, `ga_max_pair_z`, `ga_shared_debtor_share`, `ga_comm_score`, `ga_comm_size`,
`ga_risk_pagerank`) **tidak** dipakai Isolation Forest sejak v10 (posisi jaringan sudah ditangani Pilar 4); fitur ini hanya
ditampilkan sebagai konteks di profil pegawai. Isolation Forest di-fit pada approver-bulan s.d. 2024.

Kandidat fitur akses sistem (v10.1), dihitung dari Login/View/Export 90 hari terakhir dan dibandingkan median rekan
sejabatan: `ax_offhours_login` (porsi login di luar jam kerja), `ax_offhours_activity` (porsi view/export di luar jam
kerja), `ax_export_per_login`, dan `ax_view_per_login`. Fitur ini masuk Isolation Forest hanya bila PR-AUC radar approver
2025 naik minimal 0,005; bila tidak, tetap ditampilkan sebagai konteks di profil pegawai.

## Yang sengaja tidak dijadikan fitur

ID pegawai/debitur (model akan menghafal orang), nama unit dan wilayah (risiko bias), kolom `pattern` (kunci jawaban),
data kinerja kredit (terjadi setelah approval), flag rules (sudah menjadi Pilar 1), dan skor Isolation Forest (masuk ke
penggabung sebagai Pilar 2, bukan fitur model konten/relasi).
