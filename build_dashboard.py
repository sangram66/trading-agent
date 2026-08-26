"""
Dashboard generator. Reads data/ and research/ and writes a single HTML file
you open in a browser. No server, no framework, no npm.

    .venv/bin/python3 build_dashboard.py
    open dashboard.html
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.core.storage import Store
from engine.nulls.compare import compare
from engine.nulls.models import fit_garch11
from engine.nulls.statistics import (
    realised_vol, vol_bands, same_band_rate, independence_baseline,
    ann_vol, excess_kurtosis, acf_abs, sharpe, default_statistics
)
from agent.ledger import TrialLedger


def load_findings():
    findings_dir = Path("research/findings")
    if not findings_dir.exists():
        return []
    files = sorted(findings_dir.glob("oracle-*.json"), reverse=True)
    if not files:
        return []
    return json.loads(files[0].read_text())["findings"]


def load_returns(store, dataset="sp500_daily"):
    for key in store.manifest:
        if dataset in key:
            df = store.read(*key.split("/", 1))
            col = next((c for c in ("adj_close", "close") if c in df.columns), None)
            if col:
                px = df[col].to_numpy(float)
                r = np.diff(np.log(px[px > 0]))
                return r[np.isfinite(r)], df
    return None, None


def build_vol_data(r):
    """Build the volatility surface data."""
    rv = realised_vol(r, 20)
    bands = vol_bands(r, 20, 5, 5)
    valid_rv = rv[~np.isnan(rv)]

    # Band distribution
    valid_bands = bands[bands >= 0]
    band_counts = [int(np.sum(valid_bands == i)) for i in range(5)]

    # Regime tape (last 500 days)
    tape = bands[-500:].tolist()

    return {
        "rv_recent": rv[-252:].tolist(),
        "band_counts": band_counts,
        "tape": tape,
        "same_band": round(same_band_rate(r, 20, 5, 5) * 100, 1),
        "baseline": round(independence_baseline(r, 20, 5, 5) * 100, 1),
    }


def build_pipeline_data(store):
    """Build the pipeline health view."""
    nodes = []
    for key, meta in sorted(store.manifest.items()):
        layer, dataset = key.split("/", 1)
        nodes.append({
            "name": dataset,
            "layer": layer,
            "rows": meta["rows"],
            "bytes": meta["bytes"],
            "hash": meta["hash"],
            "written": meta["written_at"][:19],
            "partitions": len(meta.get("partitions", [1])),
        })
    return nodes


def build_html(findings, vol_data, pipeline_nodes, garch_params, ledger_summary,
               n_obs, sp500_prices):
    """Generate the dashboard HTML."""

    # Price series for the sparkline (last 5 years or all available)
    prices_json = json.dumps(sp500_prices[-1260:].tolist() if sp500_prices is not None else [])

    # Stats table from findings
    stats_rows = ""
    for f in findings:
        ds = f["dataset"].split("/")[-1]
        survivors = len(f["survives_all_nulls"])
        total = len(f["statistics"])
        verdict_class = "verdict-pass" if survivors > 0 else "verdict-fail"
        verdict_text = f"{survivors}/{total} survive" if survivors > 0 else "NO SIGNAL"

        for name, s in f["statistics"].items():
            p = s["p_garch"]
            p_class = "p-sig" if p < 0.05 else "p-nosig"
            stats_rows += f"""
            <tr class="stat-row" data-dataset="{ds}">
                <td class="stat-name">{name}</td>
                <td class="stat-real">{s['real']:.4g}</td>
                <td class="{p_class}">{p:.4f}</td>
            </tr>"""

    # Pipeline nodes
    layers = {"bronze": [], "silver": [], "gold": [], "marts": []}
    for n in pipeline_nodes:
        layers.setdefault(n["layer"], []).append(n)

    pipeline_html = ""
    for layer in ("bronze", "silver", "gold", "marts"):
        nodes = layers.get(layer, [])
        if not nodes:
            continue
        pipeline_html += f'<div class="layer"><div class="layer-label">{layer.upper()}</div>'
        for n in nodes:
            pipeline_html += f"""
            <div class="node node-ok">
                <div class="node-name">{n['name']}</div>
                <div class="node-meta">{n['rows']:,} rows · {n['bytes']/1024:.0f} KB</div>
                <div class="node-meta">{n['partitions']} partition{'s' if n['partitions']>1 else ''} · {n['hash'][:8]}</div>
            </div>"""
        pipeline_html += "</div>"

    # Findings summary cards
    findings_cards = ""
    for f in findings:
        ds = f["dataset"].split("/")[-1]
        survivors = f["survives_all_nulls"]
        g = f["garch"]
        hl = round(np.log(0.5) / np.log(g["persistence"])) if 0 < g["persistence"] < 1 else 999
        v_class = "card-signal" if survivors else "card-nosignal"
        v_text = f"{len(survivors)} statistics survive every null" if survivors else "NO SIGNAL — every statistic is reproducible by at least one null"
        surv_list = ", ".join(survivors) if survivors else ""

        findings_cards += f"""
        <div class="finding-card {v_class}">
            <div class="finding-header">
                <span class="finding-dataset">{ds}</span>
                <span class="finding-obs">{f['n_obs']:,} obs</span>
            </div>
            <div class="finding-garch">
                GARCH(1,1) α={g['alpha']:.4f} β={g['beta']:.4f}
                persistence={g['persistence']:.5f} half-life={hl}d
            </div>
            <div class="finding-verdict">{v_text}</div>
            {"<div class='finding-survivors'>" + surv_list + "</div>" if surv_list else ""}
        </div>"""

    # Regime tape visualization
    tape_html = ""
    if vol_data:
        for b in vol_data["tape"][-200:]:
            c = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ff6b6b"][b] if b >= 0 else "#111"
            tape_html += f'<div class="tape-bar" style="background:{c}"></div>'

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Agent — Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');

:root {{
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3a;
    --text: #e0e0e8;
    --text2: #8888a0;
    --accent: #6366f1;
    --accent2: #818cf8;
    --green: #22c55e;
    --red: #ef4444;
    --amber: #f59e0b;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Inter', -apple-system, sans-serif;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
}}

.header {{
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.header h1 {{
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 0.05em;
    color: var(--text);
}}

.header h1 span {{ color: var(--accent2); }}

.header-meta {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text2);
}}

.tabs {{
    display: flex;
    border-bottom: 1px solid var(--border);
    padding: 0 32px;
    gap: 0;
}}

.tab {{
    padding: 12px 20px;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text2);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}}

.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--accent2); border-bottom-color: var(--accent); }}

.panel {{ display: none; padding: 32px; }}
.panel.active {{ display: block; }}

/* ---- Volatility ---- */
.vol-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
}}

.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
}}

.card-title {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text2);
    margin-bottom: 12px;
}}

.big-number {{
    font-family: var(--mono);
    font-size: 32px;
    font-weight: 700;
}}

.big-number.green {{ color: var(--green); }}
.big-number.red {{ color: var(--red); }}
.big-number.amber {{ color: var(--amber); }}

.sub-number {{
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text2);
    margin-top: 4px;
}}

/* ---- Stats table ---- */
.stats-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 13px;
}}

.stats-table th {{
    text-align: left;
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text2);
    border-bottom: 1px solid var(--border);
}}

.stats-table td {{
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
}}

.stat-name {{ color: var(--text); }}
.stat-real {{ color: var(--text2); text-align: right; }}
.p-sig {{ color: var(--green); text-align: right; font-weight: 600; }}
.p-nosig {{ color: var(--text2); text-align: right; }}

/* ---- Findings ---- */
.findings-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}

.finding-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    border-left: 3px solid var(--border);
}}

.card-signal {{ border-left-color: var(--green); }}
.card-nosignal {{ border-left-color: var(--text2); }}

.finding-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}}

.finding-dataset {{
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 600;
}}

.finding-obs {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text2);
}}

.finding-garch {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text2);
    margin-bottom: 8px;
}}

.finding-verdict {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    padding: 6px 0;
}}

.card-signal .finding-verdict {{ color: var(--green); }}
.card-nosignal .finding-verdict {{ color: var(--text2); }}

.finding-survivors {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent2);
    margin-top: 4px;
}}

/* ---- Regime tape ---- */
.tape-container {{
    display: flex;
    gap: 1px;
    height: 32px;
    margin-top: 12px;
    border-radius: 4px;
    overflow: hidden;
}}

.tape-bar {{
    flex: 1;
    min-width: 2px;
}}

/* ---- Pipeline ---- */
.pipeline-flow {{
    display: flex;
    gap: 24px;
    overflow-x: auto;
    padding-bottom: 16px;
}}

.layer {{
    min-width: 200px;
    flex-shrink: 0;
}}

.layer-label {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: var(--accent2);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}

.node {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 8px;
}}

.node-ok {{ border-left: 3px solid var(--green); }}

.node-name {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
}}

.node-meta {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text2);
    margin-top: 2px;
}}

/* ---- Budget ---- */
.budget-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    max-width: 500px;
}}

.budget-line {{
    font-family: var(--mono);
    font-size: 13px;
    padding: 4px 0;
    color: var(--text2);
}}

.budget-line strong {{ color: var(--text); }}

/* ---- Sparkline ---- */
.sparkline-container {{
    height: 120px;
    position: relative;
    margin-top: 16px;
}}

canvas {{ width: 100%; height: 100%; }}

/* ---- Responsive ---- */
@media (max-width: 768px) {{
    .vol-grid {{ grid-template-columns: 1fr; }}
    .findings-grid {{ grid-template-columns: 1fr; }}
    .pipeline-flow {{ flex-direction: column; }}
    .header {{ padding: 16px; }}
    .panel {{ padding: 16px; }}
    .tabs {{ padding: 0 16px; overflow-x: auto; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1><span>▲</span> TRADING AGENT</h1>
    <div class="header-meta">{now} · {n_obs:,} obs</div>
</div>

<div class="tabs">
    <div class="tab active" onclick="show('volatility')">VOLATILITY</div>
    <div class="tab" onclick="show('research')">RESEARCH</div>
    <div class="tab" onclick="show('pipeline')">PIPELINE</div>
    <div class="tab" onclick="show('budget')">BUDGET</div>
</div>

<!-- VOLATILITY -->
<div id="volatility" class="panel active">
    <div class="vol-grid">
        <div class="card">
            <div class="card-title">SAME-BAND RATE</div>
            <div class="big-number amber">{vol_data['same_band'] if vol_data else '—'}%</div>
            <div class="sub-number">vs {vol_data['baseline'] if vol_data else '—'}% independence baseline</div>
            <div class="sub-number">lift +{round(vol_data['same_band'] - vol_data['baseline'], 1) if vol_data else '—'} pts</div>
        </div>
        <div class="card">
            <div class="card-title">GARCH(1,1) FIT</div>
            <div class="big-number" style="font-size:22px">
                α={garch_params.alpha:.4f} β={garch_params.beta:.4f}
            </div>
            <div class="sub-number">persistence {garch_params.persistence:.5f} · half-life {garch_params.half_life():.0f}d</div>
        </div>
        <div class="card">
            <div class="card-title">ANNUALISED VOL</div>
            <div class="big-number">{ann_vol(r_global)*100:.1f}%</div>
            <div class="sub-number">{n_obs:,} daily returns</div>
        </div>
        <div class="card">
            <div class="card-title">EXCESS KURTOSIS</div>
            <div class="big-number red">{excess_kurtosis(r_global):.1f}</div>
            <div class="sub-number">Gaussian = 0 · this is not Gaussian</div>
        </div>
    </div>

    <div class="card">
        <div class="card-title">REGIME TAPE — LAST 200 DAYS, COLOURED BY VOL BAND</div>
        <div style="display:flex;gap:12px;margin-bottom:8px;font-family:var(--mono);font-size:11px;color:var(--text2)">
            <span>◼ <span style="color:#1a1a2e">▊</span> low</span>
            <span>◼ <span style="color:#16213e">▊</span></span>
            <span>◼ <span style="color:#0f3460">▊</span></span>
            <span>◼ <span style="color:#e94560">▊</span></span>
            <span>◼ <span style="color:#ff6b6b">▊</span> high</span>
        </div>
        <div class="tape-container">{tape_html}</div>
    </div>

    <div class="card" style="margin-top:24px">
        <div class="card-title">EQUITY CURVE — S&P 500</div>
        <div class="sparkline-container"><canvas id="sparkCanvas"></canvas></div>
    </div>
</div>

<!-- RESEARCH -->
<div id="research" class="panel">
    <div class="findings-grid">{findings_cards}</div>
    <div class="card">
        <div class="card-title">THREE WORLDS, ONE PIPELINE — P-VALUES VS GARCH NULL</div>
        <table class="stats-table">
            <thead><tr><th>Statistic</th><th>Real</th><th>p (GARCH)</th></tr></thead>
            <tbody>{stats_rows}</tbody>
        </table>
    </div>
</div>

<!-- PIPELINE -->
<div id="pipeline" class="panel">
    <div class="card" style="margin-bottom:24px">
        <div class="card-title">DATA LINEAGE</div>
        <div class="pipeline-flow">{pipeline_html}</div>
    </div>
</div>

<!-- BUDGET -->
<div id="budget" class="panel">
    <div class="budget-card">
        <div class="card-title">TRIAL LEDGER</div>
        <pre class="budget-line">{ledger_summary}</pre>
    </div>
</div>

<script>
// Tab switching
function show(id) {{
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.classList.add('active');
    if (id === 'volatility') drawSparkline();
}}

// Sparkline
const prices = {prices_json};
function drawSparkline() {{
    const canvas = document.getElementById('sparkCanvas');
    if (!canvas || !prices.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width, h = rect.height;
    const mn = Math.min(...prices), mx = Math.max(...prices);
    const pad = 4;

    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    prices.forEach((p, i) => {{
        const x = pad + (i / (prices.length - 1)) * (w - 2 * pad);
        const y = h - pad - ((p - mn) / (mx - mn)) * (h - 2 * pad);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(99,102,241,0.15)');
    grad.addColorStop(1, 'rgba(99,102,241,0)');
    ctx.lineTo(w - pad, h); ctx.lineTo(pad, h); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
}}

window.addEventListener('load', drawSparkline);
window.addEventListener('resize', drawSparkline);
</script>
</body>
</html>"""


def main():
    store = Store("data")
    r, df = load_returns(store)
    if r is None:
        print("no price data found — run the collector first")
        return

    global r_global
    r_global = r

    findings = load_findings()
    vol_data = build_vol_data(r)
    pipeline_nodes = build_pipeline_data(store)
    garch_params = fit_garch11(r)
    ledger = TrialLedger()

    prices = df["adj_close"].to_numpy(float) if "adj_close" in df.columns else df["close"].to_numpy(float)

    html = build_html(
        findings=findings,
        vol_data=vol_data,
        pipeline_nodes=pipeline_nodes,
        garch_params=garch_params,
        ledger_summary=ledger.summary(),
        n_obs=r.size,
        sp500_prices=prices,
    )

    out = Path("dashboard.html")
    out.write_text(html)
    print(f"wrote {out} ({len(html):,} bytes)")
    print(f"open it:  open {out}")


if __name__ == "__main__":
    main()
