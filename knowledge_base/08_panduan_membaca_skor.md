---
doc_id: 08_panduan_membaca_skor
title: Panduan membaca skor Risk Engine 4 pilar Collusion Radar dan batasannya
category: internal
source: Notebook Internal Collusion Radar v10 (Bagian 2.1, 3.6, 5, 6.1) dan model card
url: internal://collusion-radar/panduan-skor
last_verified: 2026-10-14
disclaimer: Nilai threshold dan bobot pilar terkini tercantum di model card dan halaman Risk Engine 4 Pilar; dokumen ini menjelaskan cara membacanya.
---

# Panduan membaca skor

Sejak v10, Collusion Radar menghasilkan **satu skor risiko transaksi** dari **empat pilar** yang fiturnya tidak
tumpang tindih, ditambah dua skor pendukung (radar approver dan skor komunitas). Skor transaksi bisa **diurai persis**
menjadi kontribusi keempat pilar.

## 1. Skor Risiko Transaksi (0–100) — Risk Engine 4 pilar

| Pilar | Pertanyaan yang dijawab | Skor pilar (0–100) |
|---|---|---|
| **Pilar 1 · Rules SOP** | aturan SOP apa yang dilanggar transaksi ini? | S_rule = 20 × jumlah rule terpenuhi |
| **Pilar 2 · Anomali approver (Isolation Forest)** | apakah approver menyimpang dari rekan sejabatannya dalam 90 hari terakhir? | S_IF = persentil anomali (kosong bila approver < 10 keputusan) |
| **Pilar 3 · Konten transaksi** | apakah isi pengajuan (nominal, waktu, debitur) mirip kasus kolusi terbukti? | S_cls = peluang terkalibrasi model konten |
| **Pilar 4 · Relasi & jaringan** | apakah hubungan requester–approver dan posisinya di jaringan mencurigakan? | S_graph = peluang terkalibrasi model relasi |

- **Penggabungan (stacking monoton).** Keempat skor pilar digabung oleh logistic regression yang bobotnya **dipelajari**
  dari data 2025 (bukan bobot manual) dan dibatasi ≥ 0, sehingga skor pilar yang naik tidak pernah menurunkan skor akhir.
  Bobot Pilar 1 (Rules) bisa 0 karena nilai mentah dasar rules sudah dibaca Pilar 3; rules tetap tampil sebagai bukti SOP. Hasilnya peluang terkalibrasi dikali 100. Skor 80 berarti: di antara kasus dengan skor
  serupa pada populasi kasus yang direview, sekitar 80% terbukti kolusi.
- **Kontribusi pilar.** Setiap alert menampilkan kontribusi tiap pilar dalam satuan log-odds, relatif terhadap rata-rata
  data 2025. Nilai dasar + kontribusi empat pilar = skor akhir (persis). Kontribusi positif = pilar itu menaikkan risiko.
- **Kategori.** **Tinggi** bila skor ≥ threshold kerja (F1 terbaik pada prediksi out-of-fold 2025). **Sedang** bila skor
  ≥ separuh threshold kerja, **atau** bila satu pilar berlabel (konten/relasi) sangat yakin
  sendirian (jaring pengaman: precision pilar itu ≥ 90% di 2025). Selain itu **Rendah**.
- **Alasan.** Maksimal empat alasan per alert, diurutkan dari kontribusinya terhadap skor akhir: alasan pilar konten dan
  relasi diturunkan dari nilai **SHAP** fitur-fiturnya; ditambah rule SOP yang terpenuhi (Pilar 1) dan anomali approver
  (Pilar 2) bila keduanya menaikkan skor. Setiap alasan menunjuk data yang bisa diverifikasi.

### Kenapa digabung dengan bobot yang dipelajari, bukan bobot manual?

1. Bobot manual (misalnya 15/20/40/25) adalah opini yang tidak pernah diuji; stacking mengukur tambahan informasi tiap
   pilar di data yang belum dilihat pilar.
2. Rata-rata berbobot bukan peluang, sehingga ambang seperti 70/40 tidak punya arti. Skor stacking terkalibrasi.
3. Skala skor berbeda (persentil, hitungan rule, peluang) diterjemahkan ke log-odds oleh model, bukan dirata-rata.
4. Setiap kolom data hanya masuk satu pilar yang belajar, dan flag rules tidak dijadikan fitur model, sehingga satu
   sinyal tidak terhitung berkali-kali (*double counting*).
5. Dibanding satu model besar berisi semua fitur, kemampuan mengurutkan setara, tetapi skornya bisa diurai per pilar dan
   sistem tetap jalan bila satu sumber data gagal.

## 2. Skor Anomali Approver (persentil 0–100) — radar individu

- **Model**: Isolation Forest **tanpa label**, unit approver × bulan, memakai 8 fitur perilaku 90 hari terakhir (tanpa
  fitur graph). Di-fit pada data s.d. 2024; skor 2025–2026 adalah persentil terhadap distribusi latih.
- **Cara membaca**: skor 97 berarti perilaku approver bulan itu lebih menyimpang daripada 97% approver-bulan 2021–2024.
- **Kategori**: **Perlu dilihat** ≥ 95; **Pantau** 80–95; selebihnya Normal.
- **Peran ganda**: skor ini menjadi Pilar 2 pada skor transaksi, **dan** tetap disajikan sebagai radar terpisah untuk
  pemeriksaan proaktif. Alasannya, label audit hanya berisi pola yang pernah ditemukan sehingga penggabung cenderung
  memberi bobot kecil pada IF; radar terpisah menjaga agar pola baru tidak tenggelam.

## 3. Skor Komunitas (0–100)

- **Model**: graph analytics tanpa label (Louvain pada graph hubungan berisiko 12 bulan terakhir).
- **Cara membaca**: total bobot hubungan berisiko di dalam komunitas, dibandingkan komunitas terbesar dalam 12 bulan
  terakhir. Semakin tinggi, semakin padat dan kuat lingkarannya.
- **Konteks penting**: kolom "aktif (bulan)" menunjukkan berapa bulan anggota terus muncul; komunitas yang persisten lebih
  patut diperiksa. Fitur graph per transaksi menjadi input Pilar 4.

### Bukti jaringan pada satu berkas (graph heterogen)

Di Meja Investigasi, berkas bisa membawa sampai 5 **bukti jaringan**: pasangan requester–approver berisiko, hubungan
berisiko yang bertahan ≥ 3 dari 12 snapshot, requester & approver satu komunitas risiko, motif debitur bersama (debitur
yang sama juga diajukan RM lain ke approver yang sama), dan approver yang sangat sentral di jaringan berisiko. Setiap bukti
menampilkan relasi pemicu, node pendukung (ID pegawai/debitur), periode snapshot, nilai fitur, alasan, dan **batasan
interpretasi**. Bukti jaringan adalah petunjuk untuk diperiksa, bukan bukti kesalahan: mutasi portofolio, pasangan kerja
tetap, atau approver senior bisa menghasilkan pola serupa secara sah. Halaman Jaringan Risiko → Graph heterogen
menunjukkan apakah graph terbukti membantu (perbandingan tanpa graph vs dengan graph di data test).

## Apa yang BOLEH dan TIDAK BOLEH disimpulkan

| Boleh | Tidak boleh |
|---|---|
| "Transaksi ini **berindikasi** kolusi dengan skor 85, terutama karena pilar relasi ..." | "Pegawai X **terbukti** melakukan kolusi" |
| memprioritaskan pemeriksaan | memberi sanksi hanya berdasarkan skor |
| meminta klarifikasi dan dokumen pendukung | mengumumkan nama pihak yang sedang diperiksa |
| memakai skor approver untuk pemeriksaan proaktif | menganggap skor approver tinggi sebagai bukti |

## Batasan yang harus diketahui pengguna

1. **Kalibrasi relatif terhadap populasi kasus yang direview.** Proporsi kolusi di seluruh transaksi jauh lebih kecil,
   sehingga pada populasi skor lebih tepat dibaca sebagai **prioritas relatif**. Precision populasi disajikan sebagai
   rentang di dashboard.
2. **Data sintetis.** Prototype memakai data simulasi; validasi ulang dengan data internal wajib sebelum produksi.
3. **Pelaku beradaptasi.** Kalau pelaku menghindari rules, Pilar 1 dan 3 melemah; Pilar 4 (relasi) lebih tahan, tetapi
   pemantauan drift (PSI fitur dan skor pilar) dan pelatihan ulang berkala tetap diperlukan.
4. **Ketergantungan pada sumber data.** Bila fitur graph atau skor IF tidak tersedia, sistem tetap jalan tetapi kinerja
   turun; kegagalan pipeline harus memicu peringatan.
5. **Bobot dipelajari dari label audit.** Pola kolusi baru yang belum berlabel cenderung diberi bobot kecil; karena itu
   radar IF dan daftar komunitas tetap disajikan terpisah.
6. **Label tidak sempurna.** Sebagian hasil review auditor ambigu; skor tidak lebih benar dari label yang dipelajarinya.

## Alur keputusan yang disarankan

Skor Tinggi → investigator membuka dossier → membaca pilar mana yang paling berkontribusi → memverifikasi alasan dan
bukti → memeriksa komunitas dan profil approver → memutuskan: **tindak lanjut / eskalasi**, **bukan kolusi (alarm
palsu)**, atau **perlu data tambahan** → keputusan tercatat sebagai jejak audit dan umpan balik pelatihan ulang model.
