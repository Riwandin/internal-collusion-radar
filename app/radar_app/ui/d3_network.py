"""Graph komunitas interaktif (D3 force layout) dengan panel detail, dirender lewat components.html.

Dua tampilan dalam satu komponen: hubungan berisiko antarpegawai (graph Louvain) dan multi-entitas (pegawai · debitur ·
unit · sistem). Data dikirim sebagai JSON; nama pegawai ikut hanya untuk tooltip investigator.
"""
from __future__ import annotations

import json

import networkx as nx
import pandas as pd

from .network import truthy

D3_URL = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"


def risk_payload(edges: pd.DataFrame, members: pd.DataFrame) -> dict:
    """Graph risiko satu komunitas: node = anggota, edge = hubungan berisiko (RA) & berbagi debitur (RR)."""
    nodes = [{"id": row.employee_id, "kind": "approver" if row.role == "Approver" else "requester",
              "name": row.employee_name, "position": row.position, "unit": row.unit,
              "pagerank": round(float(row.pagerank), 2), "score": None if pd.isna(row.get("skor anomali (IF)"))
              else round(float(row.get("skor anomali (IF)")), 0)} for _, row in members.iterrows()]
    links = []
    for row in edges.itertuples(index=False):
        if row.kind == "RR":
            kind = "shared"
        elif truthy(row.dependency) and not truthy(row.redflag):
            kind = "dependency"
        else:
            kind = "redflag"
        links.append({"s": row.node_a, "t": row.node_b, "type": kind, "w": round(float(row.weight), 1),
                      "n": None if pd.isna(row.n_tx) else int(row.n_tx), "z": None if pd.isna(row.z) else round(float(row.z), 1),
                      "shared": None if pd.isna(row.shared) else int(row.shared)})
    return {"nodes": nodes, "edges": links}


def hetero_payload(graph: nx.MultiGraph) -> dict:
    strip = lambda node: node.split(":", 1)[1] if node.startswith("EMP:") else node
    return {"nodes": [{"id": strip(n), "kind": d["kind"], "name": d.get("label", strip(n)), "title": d.get("title", "")}
                      for n, d in graph.nodes(data=True)],
            "edges": [{"s": strip(a), "t": strip(b), "type": d["relation"], "w": int(d["weight"]), "title": d["title"]}
                      for a, b, d in graph.edges(data=True)]}


def d3_network_html(risk: dict, hetero: dict, info: dict, height: int = 560) -> str:
    payload = json.dumps({"risk": risk, "hetero": hetero, "info": info}, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload).replace("__HEIGHT__", str(height)).replace("__D3__", D3_URL)


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<style>
:root { --bg:#ffffff; --ink:#1f2328; --ink2:#52514e; --muted:#8a8984; --line:#e6e5e1; --accent:#2a78d6; --soft:#eaf2fc;
  --approver:#1baf7a; --requester:#2a78d6; --debtor:#eda100; --unit:#7b5cd6; --system:#8a8984;
  --redflag:#eb6834; --dependency:#7b5cd6; --shared:#8a8984; }
* { box-sizing:border-box; } body { margin:0; font:13px "Source Sans Pro","Segoe UI",system-ui,sans-serif; color:var(--ink); background:var(--bg); }
.wrap { display:grid; grid-template-columns:minmax(0,1fr) 250px; gap:12px; height:__HEIGHT__px; }
@media (max-width:700px) { .wrap { grid-template-columns:1fr; height:auto; } svg { height:420px; } }
svg { width:100%; height:100%; border:1px solid var(--line); border-radius:8px; background:var(--bg); display:block; }
aside { border:1px solid var(--line); border-radius:8px; padding:12px; display:flex; flex-direction:column; gap:10px; overflow:auto; }
.seg { display:inline-flex; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
.seg button { font:600 12px inherit; font-family:inherit; border:0; background:var(--bg); color:var(--ink2); padding:6px 10px; cursor:pointer; }
.seg button + button { border-left:1px solid var(--line); }
.seg button[aria-pressed="true"] { background:var(--accent); color:#fff; }
h4 { margin:0; font-size:14px; }
dl { display:grid; grid-template-columns:auto 1fr; gap:3px 10px; margin:0; }
dt { color:var(--muted); } dd { margin:0; text-align:right; font-variant-numeric:tabular-nums; }
.legend { display:flex; flex-wrap:wrap; gap:4px 10px; color:var(--ink2); font-size:12px; }
.legend i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:-1px; }
.legend i.line { width:16px; height:3px; border-radius:2px; vertical-align:2px; }
#detail { border-top:1px solid var(--line); padding-top:8px; color:var(--ink2); line-height:1.45; min-height:70px; }
#detail b { color:var(--ink); }
text { font:600 11px "Source Sans Pro","Segoe UI",system-ui,sans-serif; fill:var(--ink); pointer-events:none;
  paint-order:stroke; stroke:var(--bg); stroke-width:3px; }
text .id { font:10px "Source Code Pro",Consolas,monospace; fill:var(--muted); font-weight:400; }
</style></head><body>
<div class="wrap">
  <svg id="net" role="img" aria-label="Graph komunitas"></svg>
  <aside>
    <div class="seg"><button id="vRisk" aria-pressed="true">Hubungan berisiko</button><button id="vHet" aria-pressed="false">Multi-entitas</button></div>
    <h4 id="title"></h4><dl id="stats"></dl><div class="legend" id="legend"></div>
    <div id="detail">Arahkan kursor atau klik node/garis untuk detail. Seret node untuk menata ulang.</div>
  </aside>
</div>
<script id="data" type="application/json">__DATA__</script>
<script src="__D3__"></script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NODE = {approver:"--approver", requester:"--requester", Debtor:"--debtor", Unit:"--unit", System:"--system"};
const EDGE = {redflag:"--redflag", dependency:"--dependency", shared:"--shared", MENGAJUKAN_KE:"--redflag",
              MELAYANI_DEBITUR:"--debtor", DIPUTUS_UNTUK_DEBITUR:"--approver", BELONGS_TO:"--unit", ACCESSES:"--system"};
const LABEL = {redflag:"red flag di atas wajar", dependency:"ketergantungan tidak wajar", shared:"berbagi debitur",
               MENGAJUKAN_KE:"pengajuan antaranggota", MELAYANI_DEBITUR:"RM → debitur", DIPUTUS_UNTUK_DEBITUR:"approver → debitur",
               BELONGS_TO:"anggota unit", ACCESSES:"akses sistem (Login/View/Export)"};
const KIND = {approver:"approver", requester:"requester (RM/CA)", Debtor:"debitur", Unit:"unit", System:"sistem"};
let view = "risk", sim;
const fmt = (v, d = 1) => Number(v).toLocaleString("id-ID", {maximumFractionDigits: d});
document.getElementById("vRisk").onclick = () => setView("risk");
document.getElementById("vHet").onclick = () => setView("hetero");
function setView(v) { view = v; document.getElementById("vRisk").setAttribute("aria-pressed", v === "risk");
  document.getElementById("vHet").setAttribute("aria-pressed", v === "hetero"); draw(); }
function draw() {
  const svg = d3.select("#net"); svg.selectAll("*").remove(); if (sim) sim.stop();
  const {width, height} = svg.node().getBoundingClientRect();
  const src = DATA[view], nodes = src.nodes.map(d => ({...d})), links = src.edges.map(e => ({...e, source: e.s, target: e.t}));
  const maxW = d3.max(links, l => l.w) || 1;
  const r = d => d.kind === "approver" ? 13 : d.kind === "requester" ? 9 : d.kind === "Debtor" ? 6 : 8;
  sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(l => ["BELONGS_TO","ACCESSES"].includes(l.type) ? 95 : 72).strength(0.6))
    .force("charge", d3.forceManyBody().strength(view === "risk" ? -420 : -180))
    .force("center", d3.forceCenter(width / 2, height / 2)).force("collide", d3.forceCollide().radius(d => r(d) + 6))
    .force("x", d3.forceX(width / 2).strength(0.04)).force("y", d3.forceY(height / 2).strength(0.06));
  const link = svg.append("g").selectAll("line").data(links).join("line").style("cursor", "pointer")
    .attr("stroke", l => css(EDGE[l.type])).attr("stroke-opacity", .75).attr("stroke-width", l => 1.5 + 5 * l.w / maxW)
    .attr("stroke-dasharray", l => ["shared","BELONGS_TO","ACCESSES"].includes(l.type) ? "5 4" : null)
    .on("mouseenter click", (_, l) => edgeDetail(l));
  const node = svg.append("g").selectAll("g").data(nodes).join("g").style("cursor", "grab")
    .call(d3.drag().on("start", (e, d) => { if (!e.active) sim.alphaTarget(.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; }).on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on("mouseenter click", (_, d) => nodeDetail(d, links));
  node.append("path").attr("d", d => d3.symbol().type(d.kind === "Debtor" ? d3.symbolTriangle : d.kind === "Unit" ? d3.symbolSquare
      : d.kind === "System" ? d3.symbolDiamond : d3.symbolCircle).size(Math.PI * r(d) ** 2)())
    .attr("fill", d => css(NODE[d.kind])).attr("stroke", "#fff").attr("stroke-width", 1.5);
  const person = d => d.kind === "approver" || d.kind === "requester";
  const label = node.filter(d => d.kind !== "Debtor").append("text").attr("x", d => r(d) + 4).attr("y", d => person(d) ? -1 : 4);
  label.append("tspan").text(d => person(d) ? short(d.name || d.id) : d.id.split(":")[1]);
  label.filter(person).append("tspan").attr("class", "id").attr("x", d => r(d) + 4).attr("dy", 12).text(d => d.id);
  sim.on("tick", () => { nodes.forEach(d => { d.x = Math.max(16, Math.min(width - 110, d.x)); d.y = Math.max(16, Math.min(height - 16, d.y)); });
    link.attr("x1", l => l.source.x).attr("y1", l => l.source.y).attr("x2", l => l.target.x).attr("y2", l => l.target.y);
    node.attr("transform", d => `translate(${d.x},${d.y})`); });
  side(nodes, links);
}
function side(nodes, links) {
  const info = DATA.info, kinds = d3.rollup(nodes, v => v.length, d => d.kind);
  document.getElementById("title").textContent = `Komunitas ${info.id} · ${info.unit}`;
  const rows = [["Skor komunitas", `${fmt(info.score, 0)}/100`], ["Prioritas", info.prio], ["Anggota", info.members],
                ["Aktif", `${fmt(info.active)} dari 12 bln`]];
  if (view === "hetero") rows.push(["Debitur dipakai ≥ 2", kinds.get("Debtor") || 0], ["Unit", kinds.get("Unit") || 0], ["Sistem", kinds.get("System") || 0]);
  else rows.push(["Hubungan berisiko", links.length]);
  document.getElementById("stats").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const kindsShown = view === "risk" ? ["approver", "requester"] : ["approver", "requester", "Debtor", "Unit", "System"];
  document.getElementById("legend").innerHTML = kindsShown.map(k => `<span><i style="background:${css(NODE[k])}"></i>${KIND[k]}</span>`).join("")
    + [...new Set(links.map(l => l.type))].map(t => `<span><i class="line" style="background:${css(EDGE[t])}"></i>${LABEL[t]}</span>`).join("");
  document.getElementById("detail").textContent = view === "risk"
    ? "Tebal garis = bobot risiko (z-score red flag + ketergantungan + debitur bersama). Klik node/garis untuk detail."
    : "Node pengajuan & approval dilipat menjadi garis; tebal garis = jumlah pengajuan. Klik node/garis untuk detail.";
}
const short = s => { const w = String(s).split(" "); return w.length > 2 ? `${w[0]} ${w[1]} ${w[2][0]}.` : String(s); };
const nameOf = n => n.name && n.name !== n.id ? `${n.name} (${n.id})` : n.id.replace(/^[A-Z]+:/, "");
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function nodeDetail(d, links) {
  const degree = links.filter(l => l.source.id === d.id || l.target.id === d.id).length;
  let html = `<b>${esc(d.name || d.id)}</b> (${esc(d.id)})<br>${KIND[d.kind]} · ${degree} hubungan`;
  if (d.position) html += `<br>${esc(d.position)} · ${esc(d.unit)}`;
  if (d.pagerank !== undefined) html += `<br>Sentralitas (PageRank) ${fmt(d.pagerank, 2)}`;
  if (d.score !== undefined && d.score !== null) html += ` · skor anomali IF ${fmt(d.score, 0)}`;
  if (d.title && view === "hetero" && d.kind !== "approver" && d.kind !== "requester") html += `<br>${esc(d.title)}`;
  document.getElementById("detail").innerHTML = html;
}
function edgeDetail(l) {
  let html = `<b>${esc(nameOf(l.source))}</b> ↔ <b>${esc(nameOf(l.target))}</b><br>${LABEL[l.type]}`;
  if (l.title) html += `<br>${esc(l.title)}`;
  else if (l.type === "shared") html += `<br>${l.shared} debitur bersama · bobot ${fmt(l.w)}`;
  else html += `<br>${l.n ?? "?"} transaksi (12 bln)${l.z ? ` · z red flag ${fmt(l.z)}` : ""} · bobot ${fmt(l.w)}`;
  document.getElementById("detail").innerHTML = html;
}
draw();
let t; window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 200); });
</script></body></html>"""
