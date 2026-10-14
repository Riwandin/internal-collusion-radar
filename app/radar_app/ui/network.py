"""Visualisasi jaringan interaktif (PyVis: bisa digeser, di-zoom, diklik) dengan cadangan Plotly."""
from __future__ import annotations

import html

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .components import BLUE, GREEN, MUTED, ORANGE, PURPLE, YELLOW

EDGE_COLORS = {"redflag": ORANGE, "dependency": PURPLE, "shared": "#7a7a7a", "normal": "#c9c8c2"}


def truthy(value) -> bool:
    """True/1/'True' → True; False/0/NaN/None → False (CSV & PostgreSQL bisa berbeda tipe)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in ("true", "1", "1.0", "t", "yes")


def build_graph(edges: pd.DataFrame, employees: pd.DataFrame, highlight: set | None = None,
                node_scores: dict | None = None) -> nx.Graph:
    """edges: kolom node_a, node_b, weight, kind ('RA'/'RR'), redflag, dependency, shared, n_tx, z."""
    info = employees.set_index("employee_id")
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        kind = getattr(row, "kind", "RA")
        if kind == "RR":
            edge_type = "shared"
        elif truthy(getattr(row, "dependency", False)) and not truthy(getattr(row, "redflag", False)):
            edge_type = "dependency"
        elif truthy(getattr(row, "redflag", False)):
            edge_type = "redflag"
        else:
            edge_type = "normal"
        graph.add_edge(row.node_a, row.node_b, weight=float(row.weight), edge_type=edge_type,
                       n_tx=getattr(row, "n_tx", np.nan), z=getattr(row, "z", np.nan), shared=getattr(row, "shared", np.nan))
    for node in graph.nodes:
        record = info.loc[node] if node in info.index else None
        graph.nodes[node].update(
            name=record["employee_name"] if record is not None else node,
            role=record["role"] if record is not None else "?", position=record["position"] if record is not None else "?",
            unit=record["unit_name"] if record is not None else "?", highlight=node in (highlight or set()),
            score=(node_scores or {}).get(node))
    return graph


def edge_title(data: dict) -> str:
    if data["edge_type"] == "shared":
        return f"Berbagi {int(data['shared']) if pd.notna(data['shared']) else '?'} debitur · bobot {data['weight']:.1f}"
    parts = [f"bobot risiko {data['weight']:.1f}"]
    if pd.notna(data.get("n_tx")):
        parts.append(f"{int(data['n_tx'])} transaksi (12 bln)")
    if pd.notna(data.get("z")) and data["edge_type"] == "redflag":
        parts.append(f"red flag z = {data['z']:.1f}")
    if data["edge_type"] == "dependency":
        parts.append("ketergantungan tidak wajar (rule 5)")
    return " · ".join(parts)


def pyvis_html(graph: nx.Graph, height: int = 560) -> str | None:
    try:
        from pyvis.network import Network
    except Exception:
        return None
    network = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="#1f2328", directed=False,
                      cdn_resources="in_line")
    max_weight = max([d["weight"] for *_, d in graph.edges(data=True)] or [1])
    for node, data in graph.nodes(data=True):
        is_approver = data["role"] == "Approver"
        score = data.get("score")
        title = (f"{data['name']} ({node})\n{data['position']} · {data['unit']}" +
                 (f"\nSkor anomali: {score:.0f}" if score is not None and not pd.isna(score) else ""))
        network.add_node(node, label=f"{data['name']}\n{node}", title=title, shape="dot",
                         color={"background": GREEN if is_approver else BLUE, "border": ORANGE if data["highlight"] else "#ffffff"},
                         borderWidth=4 if data["highlight"] else 1, size=26 if is_approver else 16)
    for a, b, data in graph.edges(data=True):
        network.add_edge(a, b, value=1 + 7 * data["weight"] / max_weight, width=1 + 7 * data["weight"] / max_weight,
                         color=EDGE_COLORS[data["edge_type"]], title=edge_title(data),
                         dashes=data["edge_type"] == "shared")
    network.set_options("""{"physics": {"barnesHut": {"gravitationalConstant": -9000, "springLength": 140},
                            "stabilization": {"iterations": 180}},
                            "interaction": {"hover": true, "navigationButtons": true, "tooltipDelay": 80},
                            "nodes": {"font": {"size": 13, "face": "Inter"}}}""")
    return network.generate_html()


def plotly_network(graph: nx.Graph, title: str = "", height: int = 560):
    positions = nx.spring_layout(graph, seed=7, weight="weight", k=1.1 / np.sqrt(max(graph.number_of_nodes(), 1)))
    figure = go.Figure()
    max_weight = max([d["weight"] for *_, d in graph.edges(data=True)] or [1])
    for a, b, data in graph.edges(data=True):
        (x0, y0), (x1, y1) = positions[a], positions[b]
        figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text", text=edge_title(data),
                                    line=dict(width=1 + 7 * data["weight"] / max_weight, color=EDGE_COLORS[data["edge_type"]],
                                              dash="dash" if data["edge_type"] == "shared" else "solid"), showlegend=False))
    nodes = list(graph.nodes(data=True))
    figure.add_trace(go.Scatter(
        x=[positions[n][0] for n, _ in nodes], y=[positions[n][1] for n, _ in nodes], mode="markers+text",
        text=[d["name"] for _, d in nodes], textposition="bottom center", hoverinfo="text",
        hovertext=[f"{d['name']} ({n})<br>{d['position']}<br>{d['unit']}" for n, d in nodes],
        marker=dict(size=[24 if d["role"] == "Approver" else 14 for _, d in nodes],
                    color=[GREEN if d["role"] == "Approver" else BLUE for _, d in nodes],
                    line=dict(width=[4 if d["highlight"] else 1 for _, d in nodes],
                              color=[ORANGE if d["highlight"] else "white" for _, d in nodes])), showlegend=False))
    figure.update_layout(height=height, xaxis=dict(visible=False), yaxis=dict(visible=False),
                         title=dict(text=html.escape(title), x=0), margin=dict(l=0, r=0, t=40, b=0))
    return figure


HETERO_STYLE = {"approver": ("dot", GREEN, 24), "requester": ("dot", BLUE, 16), "Debtor": ("triangle", YELLOW, 13),
                "Unit": ("box", PURPLE, 14), "System": ("database", MUTED, 14)}
HETERO_EDGE_COLORS = {"MENGAJUKAN_KE": ORANGE, "MELAYANI_DEBITUR": YELLOW, "DIPUTUS_UNTUK_DEBITUR": GREEN,
                      "BELONGS_TO": PURPLE, "ACCESSES": "#9a9a9a"}


def collapse_hetero(nodes: pd.DataFrame, edges: pd.DataFrame, members: set, employees: pd.DataFrame,
                    shared_only: bool = True) -> nx.MultiGraph:
    """Ringkas graph heterogen untuk sekelompok pegawai: Request & Approval dilipat menjadi relasi antarentitas.

    requester ─MENGAJUKAN_KE (n pengajuan)─ approver · requester ─MELAYANI_DEBITUR─ debitur ─DIPUTUS_UNTUK_DEBITUR─ approver
    pegawai ─BELONGS_TO─ unit · pegawai ─ACCESSES─ sistem. `shared_only` = hanya debitur yang dilayani ≥ 2 requester berbeda atau diputus ≥ 2 approver berbeda.
    """
    member_nodes = {f"EMP:{m}" for m in members}
    by_relation = {name: frame for name, frame in edges.groupby("relation")}
    submits = by_relation.get("SUBMITS", pd.DataFrame(columns=["source", "target"]))
    request_of = submits[submits["source"].isin(member_nodes)].rename(columns={"source": "requester", "target": "request"})
    approval = by_relation.get("HAS_APPROVAL", pd.DataFrame(columns=["source", "target"])).rename(
        columns={"source": "request", "target": "approval"})
    decided = by_relation.get("DECIDED_BY", pd.DataFrame(columns=["source", "target"])).rename(
        columns={"source": "approval", "target": "approver"})
    debtor = by_relation.get("REQUESTED_BY_DEBTOR", pd.DataFrame(columns=["source", "target"])).rename(
        columns={"source": "request", "target": "debtor"})
    chain = (request_of[["requester", "request"]].merge(approval[["request", "approval"]], on="request", how="left")
             .merge(decided[["approval", "approver"]], on="approval", how="left").merge(debtor[["request", "debtor"]], on="request", how="left"))
    chain = chain[chain["approver"].isin(member_nodes)]
    info = employees.drop_duplicates("employee_id").set_index("employee_id")
    graph = nx.MultiGraph()

    def add_employee(node_id: str):
        raw = node_id.removeprefix("EMP:")
        record = info.loc[raw] if raw in info.index else None
        role = "approver" if record is not None and record.get("role") == "Approver" else "requester"
        graph.add_node(node_id, kind=role, label=(record["employee_name"] if record is not None else raw),
                       title=f"{record['employee_name'] if record is not None else raw} ({raw})"
                             + (f"\n{record['position']} · {record['unit_name']}" if record is not None else ""))

    for (requester, approver), group in chain.groupby(["requester", "approver"]):
        add_employee(requester), add_employee(approver)
        graph.add_edge(requester, approver, relation="MENGAJUKAN_KE", weight=len(group),
                       title=f"{len(group)} pengajuan antar anggota (12 bln)")
    links = pd.concat([chain[["requester", "debtor"]].rename(columns={"requester": "employee"}).assign(relation="MELAYANI_DEBITUR"),
                       chain[["approver", "debtor"]].rename(columns={"approver": "employee"}).assign(relation="DIPUTUS_UNTUK_DEBITUR")])
    links = links.dropna(subset=["debtor"])
    reach = links.groupby(["debtor", "relation"])["employee"].nunique().unstack(fill_value=0).max(axis=1)
    keep = reach[reach >= 2].index if shared_only else reach.index   # ≥ 2 requester berbeda ATAU ≥ 2 approver berbeda
    debtor_info = nodes.set_index("node_id") if "node_id" in nodes else pd.DataFrame()
    for (employee, debtor_node, relation), group in links[links["debtor"].isin(keep)].groupby(["employee", "debtor", "relation"]):
        attrs = debtor_info.loc[debtor_node] if debtor_node in debtor_info.index else {}
        graph.add_node(debtor_node, kind="Debtor", label=debtor_node.removeprefix("DEB:"),
                       title=f"Debitur {debtor_node.removeprefix('DEB:')} · {attrs.get('debtor_type', '?')} · "
                             f"{attrs.get('business_sector', '?')} · {int(reach[debtor_node])} requester/approver berbeda")
        graph.add_edge(employee, debtor_node, relation=relation, weight=len(group),
                       title=f"{'mengajukan' if relation == 'MELAYANI_DEBITUR' else 'memutus'} {len(group)}× untuk debitur ini")
    present = [n for n in graph if n.startswith("EMP:")]
    for relation in ("BELONGS_TO", "ACCESSES"):
        frame = by_relation.get(relation, pd.DataFrame(columns=["source", "target"]))
        for row in frame[frame["source"].isin(present)].itertuples(index=False):
            kind = "Unit" if relation == "BELONGS_TO" else "System"
            graph.add_node(row.target, kind=kind, label=row.target.split(":", 1)[1], title=row.target.split(":", 1)[1])
            title = "anggota unit" if relation == "BELONGS_TO" else (
                f"{int(getattr(row, 'n_events', 0) or 0)} aktivitas Login/View/Export · "
                f"{float(getattr(row, 'offhours_share', 0) or 0):.0%} di luar jam kerja")
            graph.add_edge(row.source, row.target, relation=relation, weight=1, title=title)
    return graph


def hetero_html(graph: nx.MultiGraph, height: int = 560) -> str | None:
    try:
        from pyvis.network import Network
    except Exception:
        return None
    network = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="#1f2328", directed=False,
                      cdn_resources="in_line")
    for node, data in graph.nodes(data=True):
        shape, color, size = HETERO_STYLE[data["kind"]]
        network.add_node(node, label=str(data["label"]), title=data["title"], shape=shape, color=color, size=size)
    max_weight = max([d["weight"] for *_, d in graph.edges(data=True)] or [1])
    for a, b, data in graph.edges(data=True):
        network.add_edge(a, b, width=1 + 5 * data["weight"] / max_weight, color=HETERO_EDGE_COLORS[data["relation"]],
                         title=data["title"], dashes=data["relation"] in ("BELONGS_TO", "ACCESSES"))
    network.set_options("""{"physics": {"barnesHut": {"gravitationalConstant": -7000, "springLength": 120},
                            "stabilization": {"iterations": 200}},
                            "interaction": {"hover": true, "navigationButtons": true, "tooltipDelay": 80},
                            "nodes": {"font": {"size": 12, "face": "Inter"}}}""")
    return network.generate_html()


HETERO_LEGEND_HTML = (f"<span class='chip'>● <b style='color:{GREEN}'>approver</b></span>"
                      f"<span class='chip'>● <b style='color:{BLUE}'>requester</b></span>"
                      f"<span class='chip'>▲ <b style='color:{YELLOW}'>debitur</b> (≥ 2 RM / approver)</span>"
                      f"<span class='chip'>■ <b style='color:{PURPLE}'>unit</b></span>"
                      f"<span class='chip'>⛁ <b style='color:{MUTED}'>sistem</b></span>"
                      f"<span class='chip'><b style='color:{ORANGE}'>━</b> pengajuan antar anggota</span>"
                      f"<span class='chip'>garis putus = keanggotaan unit / akses sistem</span>")

LEGEND_HTML = (f"<span class='chip'>● <b style='color:{GREEN}'>hijau</b> = approver</span>"
               f"<span class='chip'>● <b style='color:{BLUE}'>biru</b> = requester (RM/CA)</span>"
               f"<span class='chip'><b style='color:{ORANGE}'>━</b> red flag di atas wajar</span>"
               f"<span class='chip'><b style='color:{PURPLE}'>━</b> ketergantungan tidak wajar</span>"
               f"<span class='chip'><b style='color:#7a7a7a'>┅</b> berbagi debitur</span>"
               f"<span class='chip'>tebal garis = bobot risiko</span><span class='chip'>tepi oranye = sorotan</span>")
