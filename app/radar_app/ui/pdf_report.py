"""Generator PDF Dossier Investigasi (Pure Python PDF 1.4 — tanpa dependensi OS eksternal)."""
from __future__ import annotations

from datetime import datetime
import textwrap
import pandas as pd


def _pdf_escape(text: str) -> str:
    """Escape karakter khusus untuk string literal PDF."""
    clean = (
        str(text or "")
        .replace("—", "-")
        .replace("–", "-")
        .replace("•", "*")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("×", "x")
        .replace("±", "+/-")
        .replace("→", "->")
        .replace("↔", "<->")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )
    clean = clean.encode("latin-1", errors="replace").decode("latin-1")
    return clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class SimplePDFBuilder:
    """Builder PDF A4 multi-halaman menggunakan spesifikasi standar PDF 1.4."""

    PAGE_W = 595.28
    PAGE_H = 841.89
    MARGIN_X = 42.0
    MARGIN_TOP = 48.0
    MARGIN_BOTTOM = 48.0

    def __init__(self) -> None:
        self.pages: list[list[str]] = [[]]
        self.y = self.PAGE_H - self.MARGIN_TOP

    @property
    def stream(self) -> list[str]:
        return self.pages[-1]

    def new_page(self) -> None:
        self.pages.append([])
        self.y = self.PAGE_H - self.MARGIN_TOP

    def ensure_space(self, needed: float) -> None:
        if self.y - needed < self.MARGIN_BOTTOM:
            self.new_page()

    def draw_banner(self, title: str, subtitle: str, badge_text: str) -> None:
        self.ensure_space(68)
        top = self.y
        h = 56.0
        w = self.PAGE_W - 2 * self.MARGIN_X
        # Kotak biru tua header
        self.stream.append(f"0.12 0.23 0.37 rg {self.MARGIN_X:.1f} {top - h:.1f} {w:.1f} {h:.1f} re f")
        # Garis aksen oranye di bawah header
        self.stream.append(f"0.92 0.41 0.20 rg {self.MARGIN_X:.1f} {top - h:.1f} {w:.1f} 3.5 re f")
        # Teks judul
        self.stream.append(
            f"BT /F2 13 Tf 1 1 1 rg {self.MARGIN_X + 12:.1f} {top - 21:.1f} Td ({_pdf_escape(title)}) Tj ET"
        )
        self.stream.append(
            f"BT /F1 8.5 Tf 0.85 0.90 0.96 rg {self.MARGIN_X + 12:.1f} {top - 36:.1f} Td ({_pdf_escape(subtitle)}) Tj ET"
        )
        self.stream.append(
            f"BT /F2 8.5 Tf 0.98 0.82 0.45 rg {self.MARGIN_X + 12:.1f} {top - 48:.1f} Td ({_pdf_escape(badge_text)}) Tj ET"
        )
        self.y = top - h - 14

    def section_title(self, title: str) -> None:
        self.ensure_space(28)
        w = self.PAGE_W - 2 * self.MARGIN_X
        self.stream.append(f"0.95 0.96 0.98 rg {self.MARGIN_X:.1f} {self.y - 18:.1f} {w:.1f} 18 re f")
        self.stream.append(f"0.16 0.47 0.84 rg {self.MARGIN_X:.1f} {self.y - 18:.1f} 3.5 18 re f")
        self.stream.append(
            f"BT /F2 10 Tf 0.12 0.14 0.16 rg {self.MARGIN_X + 10:.1f} {self.y - 13:.1f} Td ({_pdf_escape(title)}) Tj ET"
        )
        self.y -= 26

    def paragraph(self, text: str, font: str = "F1", size: float = 9.0, indent: float = 0.0, width_chars: int = 92) -> None:
        lines = textwrap.wrap(str(text or "-"), width=width_chars) or ["-"]
        line_h = size + 4.0
        self.ensure_space(len(lines) * line_h + 4)
        for line in lines:
            self.stream.append(
                f"BT /{font} {size:.1f} Tf 0.15 0.16 0.18 rg {self.MARGIN_X + indent:.1f} {self.y:.1f} Td ({_pdf_escape(line)}) Tj ET"
            )
            self.y -= line_h
        self.y -= 2

    def key_value_row(self, pairs: list[tuple[str, str]]) -> None:
        self.ensure_space(34)
        w = (self.PAGE_W - 2 * self.MARGIN_X) / max(len(pairs), 1)
        top = self.y
        for idx, (label, val) in enumerate(pairs):
            x = self.MARGIN_X + idx * w
            self.stream.append(f"0.97 0.97 0.98 rg {x + 1:.1f} {top - 28:.1f} {w - 4:.1f} 28 re f")
            self.stream.append(
                f"BT /F1 7.5 Tf 0.45 0.45 0.45 rg {x + 7:.1f} {top - 10:.1f} Td ({_pdf_escape(label.upper())}) Tj ET"
            )
            self.stream.append(
                f"BT /F2 9.5 Tf 0.12 0.14 0.16 rg {x + 7:.1f} {top - 22:.1f} Td ({_pdf_escape(str(val)[:32])}) Tj ET"
            )
        self.y = top - 34

    def bullet_item(self, title: str, detail: str) -> None:
        self.ensure_space(24)
        self.stream.append(
            f"BT /F2 9.0 Tf 0.85 0.32 0.12 rg {self.MARGIN_X + 6:.1f} {self.y:.1f} Td (* {_pdf_escape(title)}) Tj ET"
        )
        self.y -= 12
        self.paragraph(detail, font="F1", size=8.5, indent=16.0, width_chars=86)

    def simple_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float]) -> None:
        row_h = 15.0
        self.ensure_space((len(rows) + 2) * row_h)
        total_w = sum(col_widths)
        # Header
        self.stream.append(f"0.20 0.25 0.32 rg {self.MARGIN_X:.1f} {self.y - row_h:.1f} {total_w:.1f} {row_h:.1f} re f")
        x = self.MARGIN_X
        for h_text, cw in zip(headers, col_widths):
            self.stream.append(
                f"BT /F2 8.0 Tf 1 1 1 rg {x + 4:.1f} {self.y - 10.5:.1f} Td ({_pdf_escape(h_text)}) Tj ET"
            )
            x += cw
        self.y -= row_h
        # Rows
        for r_idx, row in enumerate(rows):
            self.ensure_space(row_h + 4)
            if r_idx % 2 == 1:
                self.stream.append(
                    f"0.96 0.97 0.98 rg {self.MARGIN_X:.1f} {self.y - row_h:.1f} {total_w:.1f} {row_h:.1f} re f"
                )
            x = self.MARGIN_X
            for cell_text, cw in zip(row, col_widths):
                max_chars = max(int(cw / 4.6), 6)
                clipped = str(cell_text or "-")[:max_chars]
                self.stream.append(
                    f"BT /F1 7.8 Tf 0.15 0.16 0.18 rg {x + 4:.1f} {self.y - 10.5:.1f} Td ({_pdf_escape(clipped)}) Tj ET"
                )
                x += cw
            self.y -= row_h
        self.y -= 8

    def to_bytes(self) -> bytes:
        objects: list[bytes] = []

        def add_obj(content: bytes) -> int:
            objects.append(content)
            return len(objects)

        font_regular_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        font_bold_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        pages_placeholder_id = add_obj(b"")  # diisi belakangan

        page_ids: list[int] = []
        total_pages = len(self.pages)
        for idx, page_cmds in enumerate(self.pages, start=1):
            footer = (
                f"0.85 0.85 0.85 RG {self.MARGIN_X:.1f} 34 {self.PAGE_W - self.MARGIN_X:.1f} 34 m l S "
                f"BT /F1 7.5 Tf 0.45 0.45 0.45 rg {self.MARGIN_X:.1f} 22 Td "
                f"(Internal Collusion Radar v9 - Dokumen Rahasia Audit Internal | Halaman {idx} dari {total_pages}) Tj ET"
            )
            full_stream = ("\n".join(page_cmds + [footer])).encode("latin-1", errors="replace")
            content_id = add_obj(
                f"<< /Length {len(full_stream)} >>\nstream\n".encode("ascii") + full_stream + b"\nendstream"
            )
            page_dict = (
                f"<< /Type /Page /Parent {pages_placeholder_id} 0 R "
                f"/MediaBox [0 0 {self.PAGE_W:.2f} {self.PAGE_H:.2f}] "
                f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
            page_ids.append(add_obj(page_dict))

        kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects[pages_placeholder_id - 1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>".encode("ascii")
        catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_placeholder_id} 0 R >>".encode("ascii"))

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: list[int] = []
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{idx} 0 obj\n".encode("ascii"))
            out.extend(obj)
            out.extend(b"\nendobj\n")

        xref_pos = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for off in offsets:
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        out.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
        )
        return bytes(out)


def build_employee_dossier_pdf(
    employee: pd.Series,
    narrative: str,
    reasons_list: list[tuple[str, str]],
    partners_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    current_decision: str,
    investigator_name: str,
    ai_memo: str = "",
) -> bytes:
    """Bangun file PDF ringkasan investigasi lengkap untuk 1 pegawai."""
    pdf = SimplePDFBuilder()
    now_str = datetime.now().strftime("%d %b %Y %H:%M WIB")
    emp_id = str(employee.get("employee_id", "-"))
    emp_name = str(employee.get("employee_name", "-"))
    emp_role = str(employee.get("role", "-"))
    emp_pos = str(employee.get("position", "-"))
    emp_unit = str(employee.get("unit_name", "-"))
    anom_score = employee.get("skor_anomali")
    anom_str = f"{anom_score:.0f}/100 ({employee.get('kategori_anomali', '-')})" if pd.notna(anom_score) else "Tidak dinilai (Requester)"

    pdf.draw_banner(
        title=f"DOSSIER INVESTIGASI PEGAWAI: {emp_name} ({emp_id})",
        subtitle=f"Dicetak oleh: {investigator_name}  |  Tanggal: {now_str}  |  Unit: {emp_unit} ({emp_pos})",
        badge_text=f"STATUS KEPUTUSAN SAAT INI: {current_decision.upper()}",
    )

    pdf.section_title("1. PROFIL PEGAWAI & SKOR INDIKASI RISIKO")
    pdf.key_value_row([
        ("Nama & ID Pegawai", f"{emp_name} ({emp_id})"),
        ("Peran & Jabatan", f"{emp_role} - {emp_pos}"),
        ("Unit Kantor Cabang", f"{emp_unit} ({employee.get('region', '-')})"),
    ])
    pdf.key_value_row([
        ("Skor Anomali Perilaku (IF)", anom_str),
        ("Alert Transaksi Tinggi", f"{int(employee.get('alert_tinggi', 0))} berkas"),
        ("Total Eksposur Berisiko", str(employee.get("nominal_berisiko_fmt", "Rp0"))),
    ])

    pdf.section_title("2. RINGKASAN EKSEKUTIF (BAHASA MANUSIA)")
    pdf.paragraph(narrative, size=9.2)

    pdf.section_title("3. MENGAPA PEGAWAI INI DICURIGAI? (PERBANDINGAN VS REKAN SEJABATAN)")
    if reasons_list:
        for r_title, r_detail in reasons_list:
            pdf.bullet_item(r_title, r_detail)
    else:
        pdf.paragraph("Tidak ditemukan penyimpangan ekstrem di atas ambang batas pada periode ini.")

    pdf.section_title("4. INDIKASI JARINGAN & PASANGAN KERJA TERATAS (12 BULAN TERAKHIR)")
    if not partners_df.empty:
        p_rows = []
        for _, r in partners_df.head(6).iterrows():
            p_rows.append([
                str(r.get("pasangan", "-")),
                str(int(r.get("n", 0))),
                str(int(r.get("k", 0))),
                f"{float(r.get('dependency_lift', 0)):.1f}x",
                "BERISIKO" if str(r.get("risky", "")).lower() in ("true", "1") else "Normal",
            ])
        pdf.simple_table(
            headers=["Nama Pasangan Kerja", "Jml Tx", "Red Flag", "Lift Unit", "Status Relasi"],
            rows=p_rows,
            col_widths=[195.0, 55.0, 60.0, 75.0, 126.0],
        )
    else:
        pdf.paragraph("Tidak ada riwayat pasangan kerja yang mencolok pada snapshot 12 bulan terakhir.")

    pdf.section_title("5. DAFTAR TRANSAKSI KREDIT BERISIKO TERKAIT")
    if not alerts_df.empty:
        a_rows = []
        for _, r in alerts_df.head(10).iterrows():
            dt_str = pd.to_datetime(r.get("event_time")).strftime("%d/%m/%Y %H:%M") if pd.notna(r.get("event_time")) else "-"
            a_rows.append([
                str(r.get("request_id", "-")),
                dt_str,
                str(r.get("nominal_fmt", "-")),
                f"{float(r.get('risk_score', 0)):.0f}/100 ({r.get('risk_band', '-')})",
                str(r.get("alasan_singkat", "-")),
            ])
        pdf.simple_table(
            headers=["ID Berkas", "Waktu Putusan", "Nominal Kredit", "Skor Risiko", "Alasan Utama (SHAP / Rules)"],
            rows=a_rows,
            col_widths=[68.0, 82.0, 80.0, 82.0, 199.0],
        )
    else:
        pdf.paragraph("Tidak ada transaksi berisiko Tinggi/Sedang yang tercatat untuk pegawai ini.")

    if ai_memo:
        pdf.section_title("6. DRAF MEMO & REKOMENDASI KLARIFIKASI")
        for paragraph_block in str(ai_memo).split("\n"):
            cleaned = paragraph_block.strip().lstrip("#").strip()
            if cleaned:
                pdf.paragraph(cleaned, size=8.5)

    pdf.section_title("7. PENGESAHAN & TINDAK LANJUT INVESTIGATOR")
    pdf.paragraph(
        "Catatan Penting: Seluruh skor dan indikator pada dokumen ini merupakan petunjuk awal berbasis data "
        "(praduga tak bersalah), bukan bukti vonis kesalahan. Keputusan akhir wajib didasarkan pada hasil "
        "verifikasi dokumen fisik dan berita acara klarifikasi."
    )
    pdf.ensure_space(55)
    pdf.y -= 10
    pdf.stream.append(
        f"BT /F2 9.0 Tf 0.15 0.16 0.18 rg {pdf.MARGIN_X:.1f} {pdf.y:.1f} Td "
        f"(Disiapkan oleh Investigator: {_pdf_escape(investigator_name)}) Tj ET"
    )
    pdf.stream.append(
        f"BT /F2 9.0 Tf 0.15 0.16 0.18 rg {pdf.MARGIN_X + 280:.1f} {pdf.y:.1f} Td "
        f"(Mengetahui / Pejabat Anti-Fraud / SKAI:) Tj ET"
    )
    pdf.y -= 32
    pdf.stream.append(
        f"BT /F1 8.5 Tf 0.45 0.45 0.45 rg {pdf.MARGIN_X:.1f} {pdf.y:.1f} Td "
        f"(Tanda Tangan: __________________________) Tj ET"
    )
    pdf.stream.append(
        f"BT /F1 8.5 Tf 0.45 0.45 0.45 rg {pdf.MARGIN_X + 280:.1f} {pdf.y:.1f} Td "
        f"(Tanda Tangan: __________________________) Tj ET"
    )
    return pdf.to_bytes()
