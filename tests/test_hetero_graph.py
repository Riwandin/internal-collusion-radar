"""Uji integritas, kebocoran waktu, dan determinisme graph heterogen (data sintetis kecil, tanpa data mentah).

Jalankan dari root proyek:  .venv/Scripts/python.exe -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from radar_app import hetero_graph as hg  # noqa: E402

CUTOFF = pd.Timestamp("2025-07-01")


@pytest.fixture
def tables():
    employees = pd.DataFrame({
        "employee_id": ["E1", "E2", "E3", "A1", "A2", "E1"],            # E1 duplikat persis
        "employee_name": ["n1", "n2", "n3", "a1", "a2", "n1"],
        "unit_name": ["U1", "U1", "U2", "U1", "U2", "U1"], "unit_type": ["KC"] * 6,
        "position": ["RM", "RM", "RM", "BM", "BM", "RM"], "approval_limit": [0, 0, 0, 5e8, 5e8, 0],
        "employment_start_date": pd.to_datetime(["2019-01-01"] * 6)})
    debtors = pd.DataFrame({"debtor_id": ["D1", "D2", "D3"], "debtor_name": ["x", "y", "z"],
                            "debtor_type": ["Individual"] * 3, "business_sector": ["Dagang"] * 3, "region": ["Jawa"] * 3})
    ts = pd.to_datetime(["2025-01-10 09:00", "2025-02-11 10:00", "2025-03-12 11:00", "2025-06-30 16:00",
                         "2025-06-30 23:59", "2025-07-02 09:00", "2024-05-01 09:00"])
    transactions = pd.DataFrame({
        "request_id": [f"R{i}" for i in range(7)], "approval_id": [f"A{i}" for i in range(7)],
        "requester_id": ["E1", "E2", "E1", "E3", "E2", "E1", "E3"], "approver_id": ["A1", "A1", "A1", "A2", np.nan, "A1", "A2"],
        "debtor_id": ["D1", "D1", "D2", "D3", "D1", np.nan, "D3"], "request_ts": ts,
        "approve_ts": ts + pd.Timedelta("30min"), "decision": ["Approved"] * 7, "amount": [1e7] * 7,
        "tenor_months": [12] * 7, "loan_product": ["KUR Mikro"] * 7, "request_purpose": ["Modal Kerja"] * 7,
        "fraud_label": [1, 0, 1, 0, 0, 1, 0], "pattern": ["x"] * 7})
    transactions["event_time"] = transactions["approve_ts"]
    access_logs = pd.DataFrame({
        "log_id": ["L1", "L2", "L3", "L4", "L5"], "employee_id": ["A1", "A1", "E1", "A1", "E2"],
        "timestamp": pd.to_datetime(["2025-03-01 22:00", "2025-03-02 09:00", "2025-03-03 10:00", "2025-07-05 09:00",
                                     "2025-01-10 09:00"]),
        "system": ["Approval System", "Approval System", "CRM", "Approval System", "Credit System"],
        "action": ["Login", "Approve", "Export", "Login", "Edit"]})
    loan_performance = pd.DataFrame({
        "request_id": ["R0", "R1", "R2"], "disbursement_date": pd.to_datetime(["2025-01-12", "2025-02-13", "2025-08-01"]),
        "first_npl_date": pd.to_datetime(["2025-05-01", "2025-09-01", pd.NaT]), "collectibility": [5, 3, 1]})
    return transactions, employees, debtors, access_logs, loan_performance


def build(tables, cutoff=CUTOFF):
    transactions, employees, debtors, access_logs, loan_performance = tables
    return hg.build_hetero_snapshot(transactions, employees, debtors, access_logs, loan_performance, cutoff)


def test_all_integrity_checks_pass(tables):
    checks = build(tables)
    report = hg.validate_snapshot(checks)
    assert report["lulus"].all(), report[~report["lulus"]]


def test_no_event_at_or_after_cutoff(tables):
    graph = build(tables)
    assert "REQ:R5" not in graph                      # pengajuan sesudah cutoff
    assert "APV:A4" not in graph                      # approver kosong → tidak ada kejadian approval
    assert all(pd.Timestamp(d["ts"]) < CUTOFF for *_, d in graph.edges(data=True))
    assert "LP:R2" not in graph                       # dicairkan sesudah cutoff
    assert graph.nodes["LP:R0"]["npl_known_before_cutoff"] is True
    assert graph.nodes["LP:R1"]["npl_known_before_cutoff"] is False and "first_npl_date" not in graph.nodes["LP:R1"]


def test_window_excludes_old_events(tables):
    assert "REQ:R6" not in build(tables)              # > 12 bulan sebelum cutoff


def test_leak_detector_catches_future_edge(tables):
    graph = build(tables)
    graph.add_edge("EMP:E1", "REQ:R0", key="SUBMITS_LATE", relation="SUBMITS", ts=CUTOFF + pd.Timedelta("1D"))
    report = hg.validate_snapshot(graph).set_index("pemeriksaan")["lulus"]
    assert not report["Semua timestamp edge < cutoff (tanpa kebocoran waktu)"]


def test_schema_and_namespace(tables):
    graph = build(tables)
    for u, v, d in graph.edges(data=True):
        assert (graph.nodes[u]["node_type"], graph.nodes[v]["node_type"]) == hg.EDGE_SCHEMA[d["relation"]]
    assert all(n.split(":")[0] == hg.NODE_PREFIX[d["node_type"]] for n, d in graph.nodes(data=True))
    assert len([n for n, d in graph.nodes(data=True) if d["node_type"] == "Employee" and n == "EMP:E1"]) == 1


def test_access_only_behavior_actions(tables):
    graph = build(tables)
    access = [d for *_, d in graph.edges(data=True) if d["relation"] == "ACCESSES"]
    assert sum(d["n_events"] for d in access) == 2   # Login A1 + Export E1; Approve/Edit/sesudah cutoff dibuang
    a1 = [d for u, _, d in graph.edges(data=True) if d["relation"] == "ACCESSES" and u == "EMP:A1"][0]
    assert a1["offhours_share"] == 1.0


def test_no_labels_outcomes_or_pii(tables):
    graph = build(tables)
    attributes = {k for _, d in graph.nodes(data=True) for k in d} | {k for *_, d in graph.edges(data=True) for k in d}
    assert not attributes & hg.FORBIDDEN_ATTRIBUTES


def test_deterministic_regardless_of_row_order(tables):
    transactions, employees, debtors, access_logs, loan_performance = tables
    first = hg.graph_fingerprint(build(tables))
    shuffled = (transactions.sample(frac=1, random_state=1), employees.sample(frac=1, random_state=2), debtors,
                access_logs.sample(frac=1, random_state=3), loan_performance.sample(frac=1, random_state=4))
    assert hg.graph_fingerprint(build(shuffled)) == first


def test_metapath_features_are_as_of(tables):
    transactions = tables[0]
    unit = tables[1].drop_duplicates("employee_id").set_index("employee_id")["unit_name"]
    months = pd.period_range("2025-01", "2025-07", freq="M")
    pair_tables = {m: pd.DataFrame({"requester_id": ["E1"], "approver_id": ["A1"], "risky": [True]})
                   .set_index(["requester_id", "approver_id"]) for m in months}
    risky = {m: {("E1", "A1")} for m in months}
    features = hg.metapath_features(transactions, risky, pair_tables, unit)
    by_request = features.set_index(transactions["request_id"])
    # R1 (Feb): sebelum Feb hanya ada R0 (E1→A1, D1) → debitur D1 punya 1 approver, motif: E1 ≠ E2 & (E1, A1) ada → 1
    assert by_request.loc["R1", "h_debtor_n_approvers"] == 1
    assert by_request.loc["R1", "h_shared_debtor_same_approver"] == 1
    # R0 (Jan): belum ada kejadian sebelumnya di jendela → 0, bukan informasi masa depan
    assert by_request.loc["R0", "h_debtor_n_approvers"] == 0
    assert by_request.loc["R0", "h_pair_risky_months"] == 1     # snapshot Jan (dibangun dari data < Jan) saja
    assert by_request.loc["R2", "h_pair_risky_months"] == 3     # Jan, Feb, Mar


def test_evidence_score_is_unweighted_share():
    frame = pd.DataFrame({"g_risk_pair_weight": [5.0, 0.0], "h_pair_risky_months": [4, 0], "g_same_comm": [1, 0],
                          "h_shared_debtor_same_approver": [0, 0], "g_appr_risk_pagerank": [np.nan, 0.0]})
    score = hg.graph_evidence_score(frame)
    assert score.tolist() == [60.0, 0.0]


def test_evidence_records_have_reason_and_limitation():
    frame = pd.DataFrame({"request_id": ["R1"], "requester_id": ["E1"], "approver_id": ["A1"], "debtor_id": ["D1"],
                          "month": ["2025-03"], "g_risk_pair_weight": [5.0], "h_pair_risky_months": [0], "g_same_comm": [0],
                          "h_shared_debtor_same_approver": [2], "g_appr_risk_pagerank": [0.5]})
    records = hg.evidence_records(frame)
    assert set(records["bukti"]) == {"risky_pair", "shared_debtor_motif"}
    assert records["batasan_interpretasi"].str.len().gt(10).all() and records["alasan"].str.len().gt(10).all()
    assert records["node_pendukung"].str.contains("DEB:D1").all()
