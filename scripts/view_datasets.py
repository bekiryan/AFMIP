#!/usr/bin/env python3
"""
Interactive browser dashboard for the Stocks and News datasets.

Usage:
    python scripts/view_datasets.py              # default port 8502
    python scripts/view_datasets.py --port 9000  # custom port

Opens http://localhost:<port> with a tabbed interface showing both
data/datasets/stocks.parquet and data/datasets/news.parquet.
"""

import argparse
import json
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ── Resolve project root so imports work from any cwd ────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DATASETS_DIR  # noqa: E402

STOCKS_PATH = DATASETS_DIR / "stocks.parquet"
NEWS_PATH = DATASETS_DIR / "news.parquet"

# ── HTML template ────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AFMIP — Dataset Explorer</title>
<style>
  :root {
    --bg: #0f1117;
    --bg-card: #181a20;
    --bg-table: #181a20;
    --bg-header: #1c1e26;
    --bg-hover: #22252e;
    --border: #2a2d38;
    --text: #e4e6eb;
    --text-dim: #8b8fa3;
    --text-muted: #5a5e72;
    --accent: #6c5ce7;
    --accent-light: #a29bfe;
    --green: #00cec9;
    --green-bg: rgba(0,206,201,.12);
    --red: #ff6b6b;
    --red-bg: rgba(255,107,107,.12);
    --blue: #74b9ff;
    --orange: #ffa502;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); font-size: 14px;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1520px; margin: 0 auto; padding: 24px 28px; }

  /* ── Header ── */
  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
  .header h1 { font-size: 20px; font-weight: 700; letter-spacing: -.3px; }
  .header h1 span { color: var(--accent-light); }
  .header .badge {
    font-size: 11px; padding: 4px 10px; border-radius: 20px;
    background: var(--accent); color: #fff; font-weight: 600; letter-spacing: .3px;
  }

  /* ── Tabs ── */
  .tabs { display: flex; gap: 2px; margin-bottom: 22px; background: var(--bg-card); border-radius: 10px; padding: 4px; width: fit-content; }
  .tab-btn {
    padding: 9px 28px; border: none; border-radius: 8px; background: transparent;
    color: var(--text-dim); font-size: 13px; font-weight: 600; cursor: pointer;
    transition: all .15s ease; letter-spacing: .2px;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { background: var(--accent); color: #fff; }
  .tab-btn.disabled { opacity: .35; cursor: not-allowed; }

  /* ── Cards ── */
  .cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
  .card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 22px; min-width: 160px; flex: 1;
  }
  .card .label {
    font-size: 10px; text-transform: uppercase; letter-spacing: .8px;
    color: var(--text-muted); margin-bottom: 6px; font-weight: 600;
  }
  .card .value { font-size: 22px; font-weight: 700; }
  .card .sub { font-size: 11px; color: var(--text-dim); margin-top: 4px; }

  /* ── Controls ── */
  .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
  .controls input[type=text] {
    padding: 8px 14px; border: 1px solid var(--border); border-radius: 8px;
    width: 280px; font-size: 13px; background: var(--bg-card); color: var(--text);
    outline: none; transition: border-color .15s;
  }
  .controls input[type=text]:focus { border-color: var(--accent); }
  .controls input[type=text]::placeholder { color: var(--text-muted); }
  .controls select {
    padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px;
    font-size: 13px; background: var(--bg-card); color: var(--text); cursor: pointer;
    outline: none;
  }
  .controls select:focus { border-color: var(--accent); }

  /* ── Table ── */
  .tbl-wrap {
    overflow-x: auto; background: var(--bg-table); border: 1px solid var(--border);
    border-radius: 10px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    position: sticky; top: 0; background: var(--bg-header); cursor: pointer;
    user-select: none; padding: 11px 14px; text-align: left;
    border-bottom: 1px solid var(--border); white-space: nowrap;
    font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
    color: var(--text-dim); font-weight: 600;
  }
  th:hover { color: var(--text); }
  th .arrow { font-size: 9px; margin-left: 4px; color: var(--accent-light); }
  td {
    padding: 9px 14px; border-bottom: 1px solid var(--border);
    white-space: nowrap; color: var(--text);
  }
  tr:hover td { background: var(--bg-hover); }
  .trunc {
    max-width: 260px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; cursor: help;
  }
  td a { color: var(--blue); text-decoration: none; }
  td a:hover { text-decoration: underline; }
  .num-pos { color: var(--green); }
  .num-neg { color: var(--red); }

  /* ── Return indicator ── */
  .ret-chip {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 700; font-variant-numeric: tabular-nums;
  }
  .ret-up { background: var(--green-bg); color: var(--green); }
  .ret-down { background: var(--red-bg); color: var(--red); }

  /* ── Sparkline ── */
  .spark { display: inline-block; vertical-align: middle; }

  /* ── Pagination ── */
  .pager {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 2px; font-size: 13px; color: var(--text-dim);
  }
  .pager button {
    padding: 7px 16px; border: 1px solid var(--border); border-radius: 8px;
    background: var(--bg-card); color: var(--text); cursor: pointer;
    font-size: 13px; transition: all .15s;
  }
  .pager button:disabled { opacity: .3; cursor: default; }
  .pager button:hover:not(:disabled) { border-color: var(--accent); background: var(--bg-hover); }
  .pager .btn-group { display: flex; gap: 6px; }

  /* ── Missing data message ── */
  .missing-msg {
    text-align: center; padding: 60px 20px; color: var(--text-dim);
    font-size: 15px; line-height: 1.7;
  }
  .missing-msg code {
    background: var(--bg-hover); padding: 2px 8px; border-radius: 4px;
    font-size: 13px; color: var(--accent-light);
  }

  /* ── Tab panels ── */
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .wrap { padding: 14px; }
    .cards { gap: 8px; }
    .card { min-width: 140px; padding: 12px 14px; }
    .card .value { font-size: 18px; }
    .controls input[type=text] { width: 100%; }
    .header { flex-direction: column; gap: 8px; align-items: flex-start; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>AFMIP <span>Dataset Explorer</span></h1>
    <div class="badge">STAGE 1 DATA</div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('stocks')" id="tabBtnStocks">Stocks</button>
    <button class="tab-btn" onclick="switchTab('news')" id="tabBtnNews">News</button>
  </div>

  <!-- ═══ Stocks Tab ═══ -->
  <div class="tab-panel active" id="panelStocks">
    <div class="cards" id="cardsStocks"></div>
    <div class="controls">
      <input type="text" id="searchStocks" placeholder="Search stocks data&hellip;">
      <select id="tickerFilterStocks"><option value="__all__">All tickers</option></select>
      <select id="colFilterStocks"><option value="__all__">All columns</option></select>
      <select id="perPageStocks">
        <option value="25">25 rows</option>
        <option value="50" selected>50 rows</option>
        <option value="100">100 rows</option>
        <option value="250">250 rows</option>
      </select>
    </div>
    <div class="tbl-wrap">
      <table><thead><tr id="theadStocks"></tr></thead><tbody id="tbodyStocks"></tbody></table>
    </div>
    <div class="pager">
      <span id="pageInfoStocks"></span>
      <div class="btn-group">
        <button id="prevBtnStocks" onclick="changePage('stocks',-1)">&laquo; Prev</button>
        <button id="nextBtnStocks" onclick="changePage('stocks',1)">Next &raquo;</button>
      </div>
    </div>
  </div>

  <!-- ═══ News Tab ═══ -->
  <div class="tab-panel" id="panelNews">
    <div class="cards" id="cardsNews"></div>
    <div class="controls">
      <input type="text" id="searchNews" placeholder="Search news data&hellip;">
      <select id="tickerFilterNews"><option value="__all__">All tickers</option></select>
      <select id="colFilterNews"><option value="__all__">All columns</option></select>
      <select id="perPageNews">
        <option value="25">25 rows</option>
        <option value="50" selected>50 rows</option>
        <option value="100">100 rows</option>
        <option value="250">250 rows</option>
      </select>
    </div>
    <div class="tbl-wrap">
      <table><thead><tr id="theadNews"></tr></thead><tbody id="tbodyNews"></tbody></table>
    </div>
    <div class="pager">
      <span id="pageInfoNews"></span>
      <div class="btn-group">
        <button id="prevBtnNews" onclick="changePage('news',-1)">&laquo; Prev</button>
        <button id="nextBtnNews" onclick="changePage('news',1)">Next &raquo;</button>
      </div>
    </div>
  </div>
</div>

<script>
/* ── data injected by Python ── */

/* ── State per tab ── */
const state = {
  stocks: { data: [], filtered: [], columns: [], sortCol: null, sortAsc: true, page: 0 },
  news:   { data: [], filtered: [], columns: [], sortCol: null, sortAsc: true, page: 0 }
};

const PRICE_COLS = new Set(["open","high","low","close"]);
const VOLUME_COL = "volume";
const TEXT_COLS  = new Set(["title","article"]);

/* ── Initialise ── */
function init() {
  if (window.__STOCKS_COLS__) {
    state.stocks.columns = window.__STOCKS_COLS__;
    state.stocks.data = window.__STOCKS_DATA__ || [];
    state.stocks.filtered = [...state.stocks.data];
    renderCards("stocks");
    initFilters("stocks");
    renderTable("stocks");
  } else {
    document.getElementById("panelStocks").innerHTML =
      '<div class="missing-msg">Stocks dataset not found.<br>Build it with: <code>python scripts/build_stock_dataset.py</code></div>';
    document.getElementById("tabBtnStocks").classList.add("disabled");
  }

  if (window.__NEWS_COLS__) {
    state.news.columns = window.__NEWS_COLS__;
    state.news.data = window.__NEWS_DATA__ || [];
    state.news.filtered = [...state.news.data];
    renderCards("news");
    initFilters("news");
    renderTable("news");
  } else {
    document.getElementById("panelNews").innerHTML =
      '<div class="missing-msg">News dataset not found.<br>Build it with: <code>python scripts/build_news_dataset.py</code></div>';
    document.getElementById("tabBtnNews").classList.add("disabled");
  }

  /* If stocks is missing but news exists, auto-switch */
  if (!window.__STOCKS_COLS__ && window.__NEWS_COLS__) switchTab("news");
}

/* ── Tab switching ── */
function switchTab(tab) {
  var btn = document.getElementById("tabBtn" + (tab === "stocks" ? "Stocks" : "News"));
  if (btn.classList.contains("disabled")) return;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  document.getElementById("panel" + (tab === "stocks" ? "Stocks" : "News")).classList.add("active");
}

/* ── Summary cards ── */
function renderCards(tab) {
  var d = state[tab].data;
  var n = d.length;
  var tickers = new Set(d.map(r => r.ticker));
  var dates = d.map(r => r.date).filter(Boolean).sort();
  var dateRange = dates.length ? dates[0] + " \u2192 " + dates[dates.length - 1] : "\u2014";
  var cards = [];

  cards.push({ l: "Total Rows", v: n.toLocaleString() });
  cards.push({ l: "Unique Tickers", v: tickers.size.toLocaleString() });
  cards.push({ l: "Date Range", v: dateRange });

  if (tab === "stocks") {
    /* average daily volume */
    var vols = d.map(r => r.volume).filter(v => v !== null && v !== undefined && !isNaN(v));
    var avgVol = vols.length ? vols.reduce((a, b) => a + b, 0) / vols.length : 0;
    cards.push({ l: "Avg Daily Volume", v: Math.round(avgVol).toLocaleString() });
    /* trading days */
    var uniqueDates = new Set(dates);
    cards.push({ l: "Trading Days", v: uniqueDates.size.toLocaleString() });
  } else {
    /* top publisher */
    var pubCount = {};
    d.forEach(r => { if (r.publisher) pubCount[r.publisher] = (pubCount[r.publisher] || 0) + 1; });
    var pubs = Object.entries(pubCount).sort((a, b) => b[1] - a[1]);
    var topPub = pubs.length ? pubs[0][0] : "\u2014";
    var topPubN = pubs.length ? pubs[0][1] : 0;
    cards.push({ l: "Top Publisher", v: topPub, s: topPubN.toLocaleString() + " articles" });
    /* articles per ticker */
    var avgPerTicker = tickers.size ? Math.round(n / tickers.size) : 0;
    cards.push({ l: "Avg Articles/Ticker", v: avgPerTicker.toLocaleString() });
  }

  var el = document.getElementById("cards" + cap(tab));
  el.innerHTML = cards.map(function (c) {
    var sub = c.s ? '<div class="sub">' + c.s + '</div>' : '';
    return '<div class="card"><div class="label">' + c.l + '</div><div class="value">' + c.v + '</div>' + sub + '</div>';
  }).join("");
}

function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

/* ── Filters init ── */
function initFilters(tab) {
  var suffix = cap(tab);
  var cols = state[tab].columns;
  var d = state[tab].data;

  /* Column filter */
  var colSel = document.getElementById("colFilter" + suffix);
  cols.forEach(function (c) {
    var o = document.createElement("option"); o.value = c; o.textContent = c; colSel.appendChild(o);
  });

  /* Ticker filter */
  var tickerSel = document.getElementById("tickerFilter" + suffix);
  var tickers = Array.from(new Set(d.map(r => r.ticker))).filter(Boolean).sort();
  tickers.forEach(function (t) {
    var o = document.createElement("option"); o.value = t; o.textContent = t; tickerSel.appendChild(o);
  });

  /* Event listeners */
  document.getElementById("search" + suffix).addEventListener("input", function () { applyFilter(tab); });
  document.getElementById("colFilter" + suffix).addEventListener("change", function () { applyFilter(tab); });
  document.getElementById("tickerFilter" + suffix).addEventListener("change", function () { applyFilter(tab); });
  document.getElementById("perPage" + suffix).addEventListener("change", function () { state[tab].page = 0; renderTable(tab); });
}

/* ── Format cell ── */
function fmtCell(tab, col, val) {
  if (val === null || val === undefined || val === "")
    return '<span style="color:var(--text-muted)">\u2014</span>';

  if (tab === "stocks") {
    if (PRICE_COLS.has(col)) return Number(val).toFixed(2);
    if (col === VOLUME_COL) return Number(val).toLocaleString("en-US", { maximumFractionDigits: 0 });
    if (col === "_return") {
      var r = Number(val);
      var cls = r >= 0 ? "ret-up" : "ret-down";
      var sign = r >= 0 ? "+" : "";
      return '<span class="ret-chip ' + cls + '">' + sign + (r * 100).toFixed(2) + '%</span>';
    }
  }

  if (tab === "news") {
    if (col === "url") {
      var short = String(val);
      if (short.length > 40) short = short.slice(0, 37) + "\u2026";
      return '<a href="' + val + '" target="_blank" rel="noopener" title="' + String(val).replace(/"/g, '&quot;') + '">' + short + '</a>';
    }
  }

  if (col === "date") return String(val).slice(0, 10);

  var s = String(val);
  if (TEXT_COLS.has(col) && s.length > 80)
    return '<span class="trunc" title="' + s.replace(/"/g, '&quot;').replace(/</g, '&lt;') + '">' + s.slice(0, 80) + '\u2026</span>';

  return s;
}

/* ── Render table ── */
function renderTable(tab) {
  var suffix = cap(tab);
  var st = state[tab];
  var cols = st.columns;
  var perPage = +document.getElementById("perPage" + suffix).value;
  var maxPage = Math.max(0, Math.ceil(st.filtered.length / perPage) - 1);
  if (st.page > maxPage) st.page = maxPage;

  /* header */
  document.getElementById("thead" + suffix).innerHTML = cols.map(function (c) {
    var arrow = "";
    if (st.sortCol === c) arrow = st.sortAsc ? ' <span class="arrow">&#9650;</span>' : ' <span class="arrow">&#9660;</span>';
    return '<th onclick="sortBy(\'' + tab + '\',\'' + c + '\')">' + c + arrow + '</th>';
  }).join("");

  /* body */
  var start = st.page * perPage;
  var slice = st.filtered.slice(start, start + perPage);
  document.getElementById("tbody" + suffix).innerHTML = slice.map(function (r) {
    return "<tr>" + cols.map(function (c) { return "<td>" + fmtCell(tab, c, r[c]) + "</td>"; }).join("") + "</tr>";
  }).join("");

  /* pager */
  document.getElementById("pageInfo" + suffix).textContent =
    st.filtered.length
      ? "Showing " + (start + 1) + "\u2013" + Math.min(start + perPage, st.filtered.length) + " of " + st.filtered.length.toLocaleString()
      : "No matching rows";
  document.getElementById("prevBtn" + suffix).disabled = st.page === 0;
  document.getElementById("nextBtn" + suffix).disabled = st.page >= maxPage;
}

/* ── Sort ── */
function sortBy(tab, col) {
  var st = state[tab];
  if (st.sortCol === col) st.sortAsc = !st.sortAsc; else { st.sortCol = col; st.sortAsc = true; }
  st.filtered.sort(function (a, b) {
    var va = a[col], vb = b[col];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "number" && typeof vb === "number") return st.sortAsc ? va - vb : vb - va;
    return st.sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  });
  st.page = 0;
  renderTable(tab);
}

/* ── Filter ── */
function applyFilter(tab) {
  var suffix = cap(tab);
  var st = state[tab];
  var q = document.getElementById("search" + suffix).value.toLowerCase();
  var col = document.getElementById("colFilter" + suffix).value;
  var ticker = document.getElementById("tickerFilter" + suffix).value;

  st.filtered = st.data.filter(function (r) {
    if (ticker !== "__all__" && r.ticker !== ticker) return false;
    if (!q) return true;
    if (col !== "__all__") return String(r[col] != null ? r[col] : "").toLowerCase().indexOf(q) !== -1;
    return st.columns.some(function (c) { return String(r[c] != null ? r[c] : "").toLowerCase().indexOf(q) !== -1; });
  });

  if (st.sortCol) {
    /* re-apply current sort */
    var savedAsc = st.sortAsc;
    st.sortAsc = !st.sortAsc; /* toggle so sortBy toggles back */
    sortBy(tab, st.sortCol);
    /* sortBy already sets page=0 and renders */
  } else {
    st.page = 0;
    renderTable(tab);
  }
}

/* ── Pagination ── */
function changePage(tab, d) {
  state[tab].page += d;
  renderTable(tab);
}

/* ── Boot ── */
init();
</script>
</body>
</html>"""


def build_page() -> str:
    """Read parquet files and return complete HTML with embedded data."""
    import pandas as pd

    data_blocks = []

    # ── Stocks ──
    if STOCKS_PATH.exists():
        df = pd.read_parquet(STOCKS_PATH)
        print(f"Loaded stocks: {len(df):,} rows, {len(df.columns)} columns from {STOCKS_PATH.name}")

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")

        # Compute daily return column per ticker for the return indicator
        if "close" in df.columns and "ticker" in df.columns:
            df = df.sort_values(["ticker", "date"])
            df["_return"] = df.groupby("ticker")["close"].pct_change()

        columns = list(df.columns)
        records = df.where(df.notna(), None).to_dict(orient="records")

        data_blocks.append(
            f"window.__STOCKS_COLS__ = {json.dumps(columns)};\n"
            f"window.__STOCKS_DATA__ = {json.dumps(records, default=str)};\n"
        )
    else:
        print(f"Stocks parquet not found at {STOCKS_PATH} — tab will show guidance message")
        data_blocks.append("/* stocks.parquet not found */\n")

    # ── News ──
    if NEWS_PATH.exists():
        df = pd.read_parquet(NEWS_PATH)
        print(f"Loaded news:   {len(df):,} rows, {len(df.columns)} columns from {NEWS_PATH.name}")

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")

        columns = list(df.columns)
        records = df.where(df.notna(), None).to_dict(orient="records")

        data_blocks.append(
            f"window.__NEWS_COLS__ = {json.dumps(columns)};\n"
            f"window.__NEWS_DATA__ = {json.dumps(records, default=str)};\n"
        )
    else:
        print(f"News parquet not found at {NEWS_PATH} — tab will show guidance message")
        data_blocks.append("/* news.parquet not found */\n")

    if not STOCKS_PATH.exists() and not NEWS_PATH.exists():
        print("\nNo datasets found. Build them first:")
        print("  python scripts/build_stock_dataset.py")
        print("  python scripts/build_news_dataset.py")
        sys.exit(1)

    # Inject data right after the marker comment
    injected = "\n".join(data_blocks)
    return HTML_TEMPLATE.replace(
        "/* ── data injected by Python ── */",
        "/* ── data injected by Python ── */\n" + injected,
    )


class _Handler(SimpleHTTPRequestHandler):
    """Serve the single-page dashboard from memory."""

    _page_bytes: bytes = b""

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self._page_bytes)))
        self.end_headers()
        self.wfile.write(self._page_bytes)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


def main():
    parser = argparse.ArgumentParser(description="AFMIP dataset explorer dashboard")
    parser.add_argument("--port", type=int, default=8502, help="HTTP port (default 8502)")
    args = parser.parse_args()

    page_html = build_page()
    _Handler._page_bytes = page_html.encode()

    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nDashboard running at {url}  (Ctrl+C to stop)")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
