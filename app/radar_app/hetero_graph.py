"""Heterogeneous temporal graph KUR: node & edge bertipe, snapshot as-of, fitur metapath, dan bukti jaringan.

Dipakai oleh notebook (training) dan aplikasi (scoring/penjelasan), sehingga definisi fitur graph hanya ada di satu tempat.

Skema (arah relasi konsisten; relasi balik untuk traversal TIDAK dihitung sebagai bukti tersendiri):

    Employee ─SUBMITS─────────────► Request ─REQUESTED_BY_DEBTOR─► Debtor
    Request  ─HAS_APPROVAL────────► Approval ─DECIDED_BY─────────► Employee
    Employee ─BELONGS_TO──────────► Unit                (statis: data mutasi pegawai tidak tersedia)
    Employee ─ACCESSES────────────► System              (agregat Login/View/Export per jendela; tanpa request_id)
    Request  ─HAS_PERFORMANCE─────► LoanPerformance     (hanya fakta yang sudah diketahui sebelum cutoff)

Aturan waktu: snapshot dengan cutoff `c` hanya berisi kejadian dengan timestamp < c (jendela `window_months` ke
belakang). Kolektibilitas/outstanding adalah snapshot per akhir data (tanpa tanggal), jadi TIDAK pernah dimasukkan;
dari loan_performance hanya `disbursement_date` dan `first_npl_date` (< cutoff) yang dipakai, dan itu pun hanya untuk
monitoring, bukan fitur skor.

Log `Approve` dan `Edit` di access_logs cocok persis (pegawai + timestamp) dengan kejadian approval/pengajuan, jadi
keduanya adalah kejadian yang SAMA, bukan bukti tambahan. Keduanya dipakai hanya untuk validasi kecocokan, dan edge
ACCESSES hanya memuat Login/View/Export.
"""
from __future__ import annotations

import hashlib

import networkx as nx
import numpy as np
import pandas as pd

NODE_PREFIX = {"Employee": "EMP", "Request": "REQ", "Approval": "APV", "Debtor": "DEB", "Unit": "UNIT", "System": "SYS",
               "LoanPerformance": "LP"}
EDGE_SCHEMA = {
    "SUBMITS": ("Employee", "Request"),
    "REQUESTED_BY_DEBTOR": ("Request", "Debtor"),
    "HAS_APPROVAL": ("Request", "Approval"),
    "DECIDED_BY": ("Approval", "Employee"),
    "BELONGS_TO": ("Employee", "Unit"),
    "ACCESSES": ("Employee", "System"),
    "HAS_PERFORMANCE": ("Request", "LoanPerformance"),
}
BEHAVIOR_ACTIONS = ("Login", "View", "Export")          # Approve/Edit = kejadian yang sama dengan approval/pengajuan
FORBIDDEN_ATTRIBUTES = {"fraud_label", "pattern", "label_audit", "case_id", "collectibility", "collectibility_status",
                        "outstanding_amount", "days_past_due", "debtor_name", "employee_name"}

# Fitur metapath per transaksi (as-of awal bulan transaksi, jendela 12 bulan). Kandidat Pilar 4.
METAPATH_FEATURES = {
    "h_debtor_n_approvers": ("Debtor ◄ Request ► Approval ► Employee",
                             "approver berbeda yang pernah memutus pengajuan debitur ini (12 bln)"),
    "h_debtor_n_units": ("Debtor ◄ Request ◄ Employee ► Unit", "unit requester berbeda yang pernah melayani debitur ini (12 bln)"),
    "h_req_n_approvers": ("Employee ► Request ► Approval ► Employee", "approver berbeda yang dipakai requester ini (12 bln)"),
    "h_appr_n_units": ("Employee ◄ Approval ◄ Request ◄ Employee ► Unit", "unit requester berbeda yang dilayani approver ini (12 bln)"),
    "h_shared_debtor_same_approver": ("Employee ► Request ► Debtor ◄ Request ◄ Employee', Employee' ► … ► approver yang sama",
                                      "requester lain yang melayani debitur yang sama DAN memakai approver yang sama (12 bln)"),
    "h_pair_risky_months": ("Employee ► … ► Employee (edge berisiko) lintas snapshot",
                            "berapa dari 12 snapshot terakhir pasangan ini menjadi hubungan berisiko"),
    "h_unit_risky_pair_share": ("Unit ◄ Employee ► … ► Employee (edge berisiko)",
                                "porsi pasangan requester–approver dari unit requester yang berisiko di snapshot ini"),
}
METAPATH_GROUPS = {
    "Debitur–pegawai": ["h_debtor_n_approvers", "h_debtor_n_units", "h_shared_debtor_same_approver"],
    "Jangkauan pegawai": ["h_req_n_approvers", "h_appr_n_units"],
    "Persistensi & unit": ["h_pair_risky_months", "h_unit_risky_pair_share"],
}

# Bukti jaringan yang bisa dijelaskan (konfigurasi C). Skor = 100 × porsi bukti yang terpenuhi, tanpa bobot opini.
EVIDENCE_RULES = {
    "risky_pair": {"kolom": "g_risk_pair_weight", "op": ">", "nilai": 0.0, "relasi": "SUBMITS → DECIDED_BY",
                   "alasan": "pasangan requester–approver ini adalah hubungan berisiko di graph bulan ini",
                   "batasan": "hubungan berisiko = red flag/ketergantungan di atas wajar, bukan bukti kolusi"},
    "persistent_pair": {"kolom": "h_pair_risky_months", "op": ">=", "nilai": 3.0, "relasi": "SUBMITS → DECIDED_BY (lintas snapshot)",
                        "alasan": "hubungan berisiko ini bertahan ≥ 3 dari 12 snapshot terakhir",
                        "batasan": "pasangan kerja tetap yang sah juga bisa bertahan lama"},
    "same_community": {"kolom": "g_same_comm", "op": ">=", "nilai": 1.0, "relasi": "komunitas Louvain graph risiko",
                       "alasan": "requester dan approver berada di komunitas risiko yang sama",
                       "batasan": "komunitas adalah kelompok keterhubungan, bukan kelompok pelaku"},
    "shared_debtor_motif": {"kolom": "h_shared_debtor_same_approver", "op": ">=", "nilai": 1.0,
                            "relasi": "REQUESTED_BY_DEBTOR ← SUBMITS (requester lain) → DECIDED_BY (approver sama)",
                            "alasan": "debitur yang sama juga diajukan requester lain ke approver ini",
                            "batasan": "debitur bisa sah berpindah RM (mutasi, cuti, pengalihan portofolio)"},
    "central_approver": {"kolom": "g_appr_risk_pagerank", "op": ">=", "nilai": 2.0, "relasi": "PageRank graph risiko",
                         "alasan": "approver ≥ 2× lebih sentral dari rata-rata di jaringan berisiko",
                         "batasan": "approver senior/berlimit besar wajar lebih sentral"},
}
EVIDENCE_COLUMNS = sorted({rule["kolom"] for rule in EVIDENCE_RULES.values()})


def node_id(kind: str, raw) -> str:
    return f"{NODE_PREFIX[kind]}:{raw}"


def _clean_attributes(attributes: dict) -> dict:
    return {k: v for k, v in attributes.items() if k not in FORBIDDEN_ATTRIBUTES and not (isinstance(v, float) and np.isnan(v))}


def build_hetero_snapshot(transactions: pd.DataFrame, employees: pd.DataFrame, debtors: pd.DataFrame,
                          access_logs: pd.DataFrame, loan_performance: pd.DataFrame, cutoff,
                          window_months: int = 12, night_hours: tuple[int, int] = (19, 7)) -> nx.MultiDiGraph:
    """Graph heterogen as-of `cutoff`: hanya kejadian di [cutoff − window, cutoff)."""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.DateOffset(months=window_months)
    graph = nx.MultiDiGraph(cutoff=cutoff, window_start=start)
    tx = transactions.copy()
    tx["request_time"] = tx["request_ts"].fillna(tx["event_time"]) if "request_ts" in tx else tx["event_time"]
    tx = tx[(tx["request_time"] >= start) & (tx["request_time"] < cutoff)]
    employee_info = employees.drop_duplicates("employee_id").set_index("employee_id")
    debtor_info = debtors.drop_duplicates("debtor_id").set_index("debtor_id")
    used_employees = set()

    def add_employee(raw):
        used_employees.add(raw)
        return node_id("Employee", raw)

    for row in tx.itertuples(index=False):
        request = node_id("Request", row.request_id)
        graph.add_node(request, node_type="Request", ts=row.request_time, **_clean_attributes(
            {"amount": getattr(row, "amount", np.nan), "tenor_months": getattr(row, "tenor_months", np.nan),
             "loan_product": getattr(row, "loan_product", None), "request_purpose": getattr(row, "request_purpose", None)}))
        graph.add_edge(add_employee(row.requester_id), request, key="SUBMITS", relation="SUBMITS", ts=row.request_time)
        if pd.notna(row.debtor_id):
            debtor = node_id("Debtor", row.debtor_id)
            if debtor not in graph:
                info = debtor_info.loc[row.debtor_id] if row.debtor_id in debtor_info.index else {}
                graph.add_node(debtor, node_type="Debtor", **_clean_attributes(
                    {k: info.get(k) for k in ("debtor_type", "business_sector", "region")} if len(info) else {}))
            graph.add_edge(request, debtor, key="REQUESTED_BY_DEBTOR", relation="REQUESTED_BY_DEBTOR", ts=row.request_time)
        approve_ts = getattr(row, "approve_ts", pd.NaT)
        if pd.notna(approve_ts) and approve_ts < cutoff and pd.notna(row.approver_id):
            approval = node_id("Approval", getattr(row, "approval_id", row.request_id))
            graph.add_node(approval, node_type="Approval", ts=approve_ts,
                           **_clean_attributes({"decision": getattr(row, "decision", None)}))
            graph.add_edge(request, approval, key="HAS_APPROVAL", relation="HAS_APPROVAL", ts=approve_ts)
            graph.add_edge(approval, add_employee(row.approver_id), key="DECIDED_BY", relation="DECIDED_BY", ts=approve_ts)

    logs = access_logs.dropna(subset=["employee_id", "timestamp"])
    logs = logs[(logs["timestamp"] >= start) & (logs["timestamp"] < cutoff) & logs["action"].isin(BEHAVIOR_ACTIONS)].copy()
    if len(logs):
        hour = logs["timestamp"].dt.hour
        logs["offhours"] = ((hour >= night_hours[0]) | (hour < night_hours[1]) | (logs["timestamp"].dt.dayofweek >= 5)).astype(int)
        logs["system"] = logs["system"].fillna("Unknown")
        access = logs.groupby(["employee_id", "system"]).agg(
            n_events=("action", "size"), n_login=("action", lambda s: int((s == "Login").sum())),
            n_view=("action", lambda s: int((s == "View").sum())), n_export=("action", lambda s: int((s == "Export").sum())),
            offhours_share=("offhours", "mean"), ts=("timestamp", "max")).reset_index()
        for row in access.itertuples(index=False):
            system = node_id("System", row.system)
            graph.add_node(system, node_type="System", name=row.system)
            graph.add_edge(add_employee(row.employee_id), system, key="ACCESSES", relation="ACCESSES", ts=row.ts,
                           n_events=int(row.n_events), n_login=int(row.n_login), n_view=int(row.n_view),
                           n_export=int(row.n_export), offhours_share=float(row.offhours_share))

    performance = loan_performance[loan_performance["request_id"].isin(tx["request_id"])]
    performance = performance[performance["disbursement_date"] < cutoff]
    for row in performance.itertuples(index=False):
        npl_known = pd.notna(row.first_npl_date) and row.first_npl_date < cutoff
        loan = node_id("LoanPerformance", row.request_id)
        graph.add_node(loan, node_type="LoanPerformance", disbursement_date=row.disbursement_date,
                       npl_known_before_cutoff=bool(npl_known), **({"first_npl_date": row.first_npl_date} if npl_known else {}))
        graph.add_edge(node_id("Request", row.request_id), loan, key="HAS_PERFORMANCE", relation="HAS_PERFORMANCE",
                       ts=row.disbursement_date)

    for raw in sorted(used_employees):
        info = employee_info.loc[raw] if raw in employee_info.index else None
        employee = node_id("Employee", raw)
        graph.add_node(employee, node_type="Employee", **_clean_attributes(
            {"position": info.get("position"), "approval_limit": info.get("approval_limit")} if info is not None else {}))
        if info is not None and pd.notna(info.get("unit_name")):
            unit = node_id("Unit", info["unit_name"])
            graph.add_node(unit, node_type="Unit", unit_type=info.get("unit_type"))
            since = pd.to_datetime(info.get("employment_start_date"), errors="coerce")
            graph.add_edge(employee, unit, key="BELONGS_TO", relation="BELONGS_TO",
                           ts=since if pd.notna(since) and since < cutoff else start, static=True)
    return graph


def validate_snapshot(graph: nx.MultiDiGraph, cutoff=None) -> pd.DataFrame:
    """Pemeriksaan integritas: namespace, skema edge, waktu ≤ cutoff, node yatim, duplikasi, atribut terlarang."""
    cutoff = pd.Timestamp(cutoff if cutoff is not None else graph.graph["cutoff"])
    node_type = nx.get_node_attributes(graph, "node_type")
    untyped = [n for n in graph if n not in node_type]
    bad_prefix = [n for n, t in node_type.items() if not str(n).startswith(NODE_PREFIX.get(t, "?") + ":")]
    bad_schema, late, keys, duplicates = [], [], set(), 0
    for u, v, k, data in graph.edges(keys=True, data=True):
        relation = data.get("relation")
        expected = EDGE_SCHEMA.get(relation)
        if expected is None or (node_type.get(u), node_type.get(v)) != expected:
            bad_schema.append((u, v, relation))
        ts = data.get("ts")
        if ts is None or pd.isna(ts) or pd.Timestamp(ts) >= cutoff:
            late.append((u, v, relation))
        if (u, v, relation) in keys:
            duplicates += 1
        keys.add((u, v, relation))
    late_nodes = [n for n, d in graph.nodes(data=True)
                  if any(isinstance(d.get(c), pd.Timestamp) and d[c] >= cutoff for c in ("ts", "disbursement_date", "first_npl_date"))]
    forbidden = sorted({k for _, d in graph.nodes(data=True) for k in d if k in FORBIDDEN_ATTRIBUTES}
                       | {k for *_, d in graph.edges(data=True) for k in d if k in FORBIDDEN_ATTRIBUTES})
    orphans = [n for n in graph if graph.degree(n) == 0]
    approvals = [n for n, t in node_type.items() if t == "Approval"]
    requests = [n for n, t in node_type.items() if t == "Request"]
    bad_approvals = sum(1 for n in approvals if graph.in_degree(n) != 1 or graph.out_degree(n) != 1)
    bad_requests = sum(1 for n in requests
                       if sum(1 for _, _, d in graph.in_edges(n, data=True) if d["relation"] == "SUBMITS") != 1)
    checks = [
        ("Setiap node bertipe", not untyped, f"{len(untyped)} node tanpa tipe"),
        ("ID node memakai namespace sesuai tipe", not bad_prefix, f"{len(bad_prefix)} node salah prefix"),
        ("Setiap edge sesuai skema tipe asal→tujuan", not bad_schema, f"{len(bad_schema)} edge melanggar skema"),
        ("Semua timestamp edge < cutoff (tanpa kebocoran waktu)", not late, f"{len(late)} edge ≥ cutoff"),
        ("Semua timestamp node < cutoff", not late_nodes, f"{len(late_nodes)} node ≥ cutoff"),
        ("Tidak ada edge ganda (asal, tujuan, relasi)", duplicates == 0, f"{duplicates} duplikat"),
        ("Tidak ada node yatim (degree 0)", not orphans, f"{len(orphans)} node yatim"),
        ("Tidak ada self-loop", nx.number_of_selfloops(graph) == 0, f"{nx.number_of_selfloops(graph)} self-loop"),
        ("Setiap Approval: 1 HAS_APPROVAL masuk & 1 DECIDED_BY keluar", bad_approvals == 0, f"{bad_approvals} approval tidak lengkap"),
        ("Setiap Request punya tepat 1 pengaju (SUBMITS)", bad_requests == 0, f"{bad_requests} request tidak lengkap"),
        ("Tanpa label/outcome/PII di atribut", not forbidden, ", ".join(forbidden) or "bersih"),
    ]
    return pd.DataFrame(checks, columns=["pemeriksaan", "lulus", "detail"])


def graph_fingerprint(graph: nx.MultiDiGraph) -> str:
    """Sidik jari deterministik (urutan input tidak berpengaruh) untuk uji reproducibility."""
    digest = hashlib.sha256()
    for node in sorted(graph.nodes):
        digest.update(f"{node}|{graph.nodes[node].get('node_type')}\n".encode())
    for u, v, data in sorted((u, v, d) for u, v, d in graph.edges(data=True)):
        digest.update(f"{u}|{v}|{data.get('relation')}|{data.get('ts')}\n".encode())
    return digest.hexdigest()


def schema_summary(graph: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_counts = pd.Series(nx.get_node_attributes(graph, "node_type")).value_counts().rename_axis("tipe_node").reset_index(name="jumlah")
    edge_counts = (pd.Series([d["relation"] for *_, d in graph.edges(data=True)]).value_counts()
                   .rename_axis("relasi").reset_index(name="jumlah"))
    edge_counts["asal"] = edge_counts["relasi"].map(lambda r: EDGE_SCHEMA[r][0])
    edge_counts["tujuan"] = edge_counts["relasi"].map(lambda r: EDGE_SCHEMA[r][1])
    return node_counts, edge_counts[["relasi", "asal", "tujuan", "jumlah"]]


def evidence_subgraph(graph: nx.MultiDiGraph, employee_ids, label_of_node: dict | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subgraph bukti untuk sekelompok pegawai: pengajuan ANTAR anggota (requester & approver sama-sama anggota),
    debitur, unit, dan sistem yang mereka akses. Return (nodes, edges) siap ekspor (tanpa PII)."""
    members = {node_id("Employee", e) for e in employee_ids}
    keep = set(members)
    for member in members & set(graph):
        for _, request, data in graph.out_edges(member, data=True):
            if data["relation"] != "SUBMITS":
                continue
            approvals = [v for _, v, d in graph.out_edges(request, data=True) if d["relation"] == "HAS_APPROVAL"]
            approvers = {w for a in approvals for _, w, d in graph.out_edges(a, data=True) if d["relation"] == "DECIDED_BY"}
            if approvers & members:
                keep.update([request, *approvals])
                keep.update(v for _, v, d in graph.out_edges(request, data=True) if d["relation"] == "REQUESTED_BY_DEBTOR")
        keep.update(v for _, v, d in graph.out_edges(member, data=True) if d["relation"] in ("BELONGS_TO", "ACCESSES"))
    sub = graph.subgraph(keep)
    nodes = pd.DataFrame([{"node_id": n, "node_type": d.get("node_type"), "label": (label_of_node or {}).get(n, n),
                           **{k: v for k, v in d.items() if k in ("position", "unit_type", "decision", "amount",
                                                                  "debtor_type", "business_sector", "region")}}
                          for n, d in sub.nodes(data=True)])
    edges = pd.DataFrame([{"source": u, "target": v, "relation": d["relation"], "ts": d.get("ts"),
                           **{k: d[k] for k in ("n_events", "n_export", "offhours_share") if k in d}}
                          for u, v, d in sub.edges(data=True)])
    return nodes, edges


def metapath_features(transactions: pd.DataFrame, risky_pairs_by_month: dict, pair_tables_by_month: dict,
                      unit_of_employee: pd.Series, window_months: int = 12) -> pd.DataFrame:
    """Fitur metapath per transaksi, as-of awal bulan transaksi (kejadian 12 bulan sebelum bulan itu saja).

    risky_pairs_by_month : {Period: set((requester, approver))} hubungan berisiko per snapshot (dibangun tanpa label)
    pair_tables_by_month : {Period: DataFrame index (requester_id, approver_id) dengan kolom `risky`}
    """
    tx = transactions[["request_id", "requester_id", "approver_id", "debtor_id", "event_time"]].copy()
    tx["requester_unit"] = tx["requester_id"].map(unit_of_employee)
    tx["ym"] = tx["event_time"].dt.to_period("M")
    decided = transactions[transactions["approver_id"].notna() & transactions["approve_ts"].notna()][
        ["requester_id", "approver_id", "debtor_id", "approve_ts"]].copy()
    decided["requester_unit"] = decided["requester_id"].map(unit_of_employee)
    out = pd.DataFrame(np.nan, index=tx.index, columns=list(METAPATH_FEATURES))
    for month, index in tx.groupby("ym").groups.items():
        if month not in pair_tables_by_month:
            continue
        month_start = month.start_time
        window = decided[(decided["approve_ts"] >= month_start - pd.DateOffset(months=window_months))
                         & (decided["approve_ts"] < month_start)]
        rows = tx.loc[index]
        debtor_approvers = window.groupby("debtor_id")["approver_id"].nunique()
        debtor_units = window.groupby("debtor_id")["requester_unit"].nunique()
        requester_approvers = window.groupby("requester_id")["approver_id"].nunique()
        approver_units = window.groupby("approver_id")["requester_unit"].nunique()
        # motif: requester lain (r' ≠ r) yang melayani debitur d DAN punya keputusan dari approver a
        debtor_requesters = window[["debtor_id", "requester_id"]].dropna().drop_duplicates()
        pair_set = window[["requester_id", "approver_id"]].drop_duplicates().assign(_pair=1)
        motif = (rows.reset_index()[["index", "requester_id", "approver_id", "debtor_id"]].dropna(subset=["debtor_id"])
                 .merge(debtor_requesters.rename(columns={"requester_id": "other"}), on="debtor_id"))
        motif = motif[motif["other"] != motif["requester_id"]]
        motif = motif.merge(pair_set.rename(columns={"requester_id": "other"}), on=["other", "approver_id"])
        motif_count = motif.groupby("index")["other"].nunique()
        persistence = np.zeros(len(rows))
        keys = list(zip(rows["requester_id"], rows["approver_id"]))
        for lag in range(window_months):
            risky = risky_pairs_by_month.get(month - lag)
            if risky:
                persistence += np.fromiter((key in risky for key in keys), dtype=float, count=len(keys))
        pairs = pair_tables_by_month[month].reset_index()
        pairs["requester_unit"] = pairs["requester_id"].map(unit_of_employee)
        unit_risky_share = pairs.groupby("requester_unit")["risky"].mean()
        out.loc[index, "h_debtor_n_approvers"] = rows["debtor_id"].map(debtor_approvers).fillna(0).to_numpy()
        out.loc[index, "h_debtor_n_units"] = rows["debtor_id"].map(debtor_units).fillna(0).to_numpy()
        out.loc[index, "h_req_n_approvers"] = rows["requester_id"].map(requester_approvers).fillna(0).to_numpy()
        out.loc[index, "h_appr_n_units"] = rows["approver_id"].map(approver_units).fillna(0).to_numpy()
        out.loc[index, "h_shared_debtor_same_approver"] = pd.Series(index, index=index).map(motif_count).fillna(0).to_numpy()
        out.loc[index, "h_pair_risky_months"] = persistence
        out.loc[index, "h_unit_risky_pair_share"] = rows["requester_unit"].map(unit_risky_share).fillna(0).to_numpy()
    return out


def evidence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Bukti jaringan 0/1 per transaksi. Nilai kosong (riwayat belum ada) = bukti tidak terpenuhi."""
    flags = {}
    for name, rule in EVIDENCE_RULES.items():
        value = pd.to_numeric(df[rule["kolom"]], errors="coerce") if rule["kolom"] in df else pd.Series(np.nan, index=df.index)
        hit = value > rule["nilai"] if rule["op"] == ">" else value >= rule["nilai"]
        flags[name] = hit.fillna(False).astype(int)
    return pd.DataFrame(flags, index=df.index)


def graph_evidence_score(df: pd.DataFrame) -> pd.Series:
    """Skor bukti jaringan 0–100 = 100 × porsi bukti yang terpenuhi (bobot sama, seperti rule_score)."""
    return evidence_flags(df).mean(axis=1) * 100


def evidence_records(df: pd.DataFrame, period_column: str = "month", max_rows: int | None = None) -> pd.DataFrame:
    """Satu baris per (transaksi, bukti terpenuhi): relasi pemicu, node pendukung, periode, nilai, alasan, batasan."""
    flags = evidence_flags(df)
    rows = []
    for name, rule in EVIDENCE_RULES.items():
        hit = df.index[flags[name] == 1]
        if not len(hit):
            continue
        part = df.loc[hit]
        period = part[period_column].astype(str) if period_column in part else ""
        rows.append(pd.DataFrame({
            "request_id": part["request_id"].to_numpy(), "bukti": name, "relasi": rule["relasi"],
            "node_pendukung": ("EMP:" + part["requester_id"].astype(str) + " → EMP:" + part["approver_id"].astype(str)
                               + np.where(part.get("debtor_id", pd.Series(index=part.index)).notna(),
                                          " · DEB:" + part.get("debtor_id", pd.Series("", index=part.index)).astype(str), "")),
            "periode_snapshot": period, "jendela": "12 bulan sebelum bulan transaksi",
            "fitur": rule["kolom"], "nilai": pd.to_numeric(part[rule["kolom"]], errors="coerce").round(3).to_numpy(),
            "alasan": rule["alasan"], "batasan_interpretasi": rule["batasan"]}))
    records = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["request_id", "bukti", "relasi", "node_pendukung", "periode_snapshot", "jendela", "fitur", "nilai", "alasan",
                 "batasan_interpretasi"])
    return records.head(max_rows) if max_rows else records


def access_monthly(access_logs: pd.DataFrame, night_hours: tuple[int, int] = (19, 7)) -> pd.DataFrame:
    """Agregat perilaku akses per (pegawai, bulan) dari Login/View/Export (Approve/Edit = kejadian transaksi itu sendiri)."""
    logs = access_logs.dropna(subset=["employee_id", "timestamp"])
    logs = logs[logs["action"].isin(BEHAVIOR_ACTIONS)].copy()
    hour = logs["timestamp"].dt.hour
    logs["offhours"] = ((hour >= night_hours[0]) | (hour < night_hours[1]) | (logs["timestamp"].dt.dayofweek >= 5)).astype(int)
    logs["ym"] = logs["timestamp"].dt.to_period("M")
    for action in BEHAVIOR_ACTIONS:
        logs[f"n_{action.lower()}"] = (logs["action"] == action).astype(int)
    logs["offhours_login"] = logs["offhours"] * logs["n_login"]
    logs["offhours_other"] = logs["offhours"] * (1 - logs["n_login"])
    return (logs.groupby(["employee_id", "ym"])[["n_login", "n_view", "n_export", "offhours_login", "offhours_other"]]
            .sum().reset_index())


def access_reconciliation(transactions: pd.DataFrame, access_logs: pd.DataFrame) -> pd.DataFrame:
    """Berapa approval/pengajuan yang punya log Approve/Edit dengan pegawai & timestamp PERSIS sama (validasi, bukan fitur)."""
    logs = access_logs.dropna(subset=["employee_id", "timestamp"])
    rows = []
    for action, who, when, label in [("Approve", "approver_id", "approve_ts", "approval ↔ log Approve"),
                                     ("Edit", "requester_id", "request_ts", "pengajuan ↔ log Edit")]:
        events = transactions.dropna(subset=[who, when])[[who, when]]
        keys = set(zip(logs.loc[logs["action"] == action, "employee_id"], logs.loc[logs["action"] == action, "timestamp"]))
        matched = np.fromiter(((w, t) in keys for w, t in zip(events[who], events[when])), dtype=bool, count=len(events))
        rows.append({"pencocokan": label, "kejadian": len(events), "cocok_persis": int(matched.sum()),
                     "porsi": float(matched.mean()) if len(events) else np.nan})
    return pd.DataFrame(rows)
