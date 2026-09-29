"""
dashboard.py
Step 3: the interactive Rivera Portfolio Lab dashboard.

Run
    streamlit run dashboard.py

Every control in the sidebar reruns the walk-forward backtest or the
retirement simulation live, so you can see how each choice changes the result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

DATA_DIR = Path(__file__).parent / "data"
PRICES = DATA_DIR / "prices.csv"

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------

INK, MUTED, LINE, PAPER = "#E7EDF4", "#97A5B8", "#26364F", "#0E1726"
TEAL, BRASS, SLATE, LOSS, GAIN = "#45B5A6", "#D6AE62", "#7D9BD0", "#E27C70", "#5FC48C"
STRATEGY_COLORS = {
    "Max Sharpe (shrunk)": TEAL,
    "Risk parity": SLATE,
    "Minimum variance": "#A4B870",
    "Constrained equal weight": "#B795C9",
    "70/30 benchmark": BRASS,
}
CLASS_COLORS = {"Stocks": TEAL, "Bonds": SLATE, "Real assets": BRASS}
TICKER_COLORS = {
    "VTI": "#45B5A6", "VXUS": "#6FCBBE", "VIG": "#2E8C80", "USMV": "#9EDDD3",
    "BND": "#7D9BD0", "BNDX": "#A4BDE5", "TIP": "#5876AA", "SHY": "#C6D5F0",
    "VNQ": "#D6AE62", "IAU": "#EDD39C",
}
AUTHOR = "Antonino Casabianca"
BENCH = "70/30 benchmark"
PHRASES = {
    "Max Sharpe (shrunk)": "the shrunk max-Sharpe strategy",
    "Risk parity": "risk parity",
    "Minimum variance": "the minimum-variance strategy",
    "Constrained equal weight": "near-equal weights",
}
CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

st.set_page_config(page_title="Rivera Portfolio Lab", page_icon="🧭", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [class*="st-"], button, input, textarea, select {{
    font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{
    font-family: 'Material Symbols Rounded' !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3, .serif {{ font-family: 'Newsreader', Georgia, 'Times New Roman', serif; color: {INK}; }}
h3 {{ font-weight: 500; font-size: 1.35rem; margin-top: .6rem; }}
[data-testid="stSidebar"] {{ background: #111C2E; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.25rem; font-weight: 500; margin-bottom: 0; }}

.hero {{ display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(0, 1fr);
         gap: 3rem; align-items: end; padding: .5rem 0 1.8rem;
         border-bottom: 1px solid {LINE}; margin-bottom: 1.4rem; }}
.hero .lede {{ color: {MUTED}; font-size: .95rem; margin: 0 0 .4rem; }}
.hero h1 {{ font-size: clamp(2.4rem, 5vw, 4rem); line-height: 1.02; font-weight: 500;
            letter-spacing: -0.02em; margin: 0 0 1rem; padding: 0; }}
.hero .answer {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.45rem;
                 line-height: 1.4; max-width: 34em; margin: 0; }}
.hero .answer .prob {{ font-size: 2.3rem; color: {TEAL}; font-weight: 600;
                       font-variant-numeric: tabular-nums; }}
.hero .answer .prob.low {{ color: {LOSS}; }}
.hero .then {{ color: {MUTED}; margin: .8rem 0 0; font-size: 1rem; max-width: 40em; }}
.facts {{ display: grid; grid-template-columns: auto 1fr; gap: .45rem 1.2rem; margin: 0;
          padding: 1.1rem 1.3rem; background: #152238; border: 1px solid {LINE}; border-radius: 10px; }}
.facts dt {{ color: {MUTED}; font-size: .9rem; }}
.facts dd {{ margin: 0; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }}

.strip {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); background: #152238;
          border: 1px solid {LINE}; border-radius: 10px; margin-bottom: 1.2rem; }}
.strip > div {{ padding: .95rem 1.1rem; border-left: 1px solid {LINE}; }}
.strip > div:first-child {{ border-left: 0; }}
.strip .k {{ color: {MUTED}; font-size: .85rem; }}
.strip .v {{ font-size: 1.55rem; font-weight: 600; font-variant-numeric: tabular-nums; margin: .1rem 0; }}
.strip .d {{ font-size: .85rem; font-variant-numeric: tabular-nums; }}
.better {{ color: {GAIN}; }} .worse {{ color: {LOSS}; }} .same {{ color: {MUTED}; }}

.demo {{ background: #3A2F14; border: 1px solid #7A6428; color: #F2D995; border-radius: 8px;
         padding: .7rem 1rem; margin-bottom: 1.2rem; font-size: .95rem; }}
.note {{ color: {MUTED}; font-size: .92rem; max-width: 70em; }}

table.clean {{ width: 100%; border-collapse: collapse; background: #152238; border: 1px solid {LINE};
               border-radius: 10px; overflow: hidden; font-variant-numeric: tabular-nums; }}
table.clean th, table.clean td {{ padding: .55rem .8rem; border-bottom: 1px solid {LINE};
                                  text-align: right; font-size: .93rem; }}
table.clean th {{ background: #1A2A42; font-weight: 600; color: {INK}; }}
table.clean td:first-child, table.clean th:first-child {{ text-align: left; }}
table.clean tr:last-child td {{ border-bottom: 0; }}
table.clean td.best {{ background: #1D4A48; font-weight: 600; }}
table.clean tr.sel td {{ background: #182A40; }}
.pass {{ color: {GAIN}; font-weight: 600; }} .fail {{ color: {LOSS}; font-weight: 600; }}
.tablewrap {{ overflow-x: auto; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 0; font-weight: 500; }}

@media (max-width: 900px) {{
  .hero {{ grid-template-columns: 1fr; gap: 1.4rem; }}
  .strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .strip > div {{ border-left: 0; border-top: 1px solid {LINE}; }}
}}
</style>
""", unsafe_allow_html=True)


def style(fig, height=380, title=None, pct_y=False, money_y=False, legend=True):
    fig.update_layout(
        height=height, title=dict(text=title, x=0, xanchor="left",
                                  font=dict(family="Newsreader, Georgia, serif", size=20, color=INK)),
        margin=dict(l=6, r=6, t=(84 if legend else 48) if title else 16, b=6),
        paper_bgcolor="#152238", plot_bgcolor="#152238",
        font=dict(family="Public Sans, system-ui, sans-serif", size=13, color=INK),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#152238", bordercolor=LINE, font_color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE)
    fig.update_yaxes(gridcolor="#22324B", zeroline=False, linecolor=LINE)
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    if money_y:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


def money(x):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x:,.0f}"


def html(s):
    st.markdown(s, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data and cached computations
# ---------------------------------------------------------------------------

if not PRICES.exists():
    st.title("Rivera Portfolio Lab")
    st.error("No price data yet. From the project folder, run `python data_pull.py` "
             "(or `python data_pull.py --demo` to try it offline), then refresh this page.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_returns(mtime):
    return E.load_returns(PRICES)


@st.cache_data(show_spinner=False)
def backtest(mtime, lookback_years, rebalance_days, cost_bps, vol_cap, max_position, shrink):
    ips = E.IPS(max_position=max_position, vol_cap=vol_cap)
    settings = E.Settings(lookback_days=int(lookback_years * 252), rebalance_days=rebalance_days,
                          cost_bps=cost_bps, mean_shrink=shrink)
    return E.run_all(load_returns(mtime), ips, settings), ips


@st.cache_data(show_spinner=False)
def simulate(mtime, daily, start, contrib, years, inflation, haircut):
    return E.monte_carlo(daily, start, contrib, years, inflation, haircut)


mtime = PRICES.stat().st_mtime
returns = load_returns(mtime)
cash = returns[E.CASH_PROXY]

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Strategy")
    strategy = st.selectbox("Allocation method", list(E.STRATEGIES), index=1,
                            label_visibility="collapsed")
    st.caption(E.STRATEGIES[strategy][1])

    st.markdown("## Backtest settings")
    lookback = st.select_slider("Years of history each rebalance learns from",
                                options=[2, 3, 4, 5], value=3)
    rebalance = st.radio("Rebalance", ["Monthly", "Quarterly", "Yearly"], index=1,
                         horizontal=True)
    rebalance_days = {"Monthly": 21, "Quarterly": 63, "Yearly": 252}[rebalance]
    cost_bps = st.slider("Trading cost (basis points per trade)", 0, 30, 5)
    vol_cap = st.slider("Volatility ceiling for max Sharpe", 8, 20, 14, format="%d%%") / 100
    max_pos = st.slider("Largest allowed single holding", 15, 40, 25, format="%d%%") / 100
    shrink = st.slider("How much to distrust past returns", 0.0, 1.0, 0.5, 0.1,
                       help="0 uses raw historical averages. 1 assumes every asset earns the "
                            "same. Only affects max Sharpe.")

    st.markdown("## Maya's plan")
    c = E.CLIENT
    start_value = st.number_input("Saved so far ($)", 0, 5_000_000, c["start_value"], 5_000)
    contrib = st.number_input("Monthly saving ($)", 0, 50_000, c["monthly_contribution"], 100)
    goal = st.number_input("Goal in today's dollars ($)", 100_000, 10_000_000, c["goal_real"], 50_000)
    retire_age = st.slider("Retirement age", c["age"] + 5, 70, c["retire_age"])
    inflation = st.slider("Inflation", 0.0, 5.0, c["inflation"] * 100, 0.25, format="%.2f%%") / 100
    haircut = st.slider("Assume future returns are lower by", 0.0, 3.0, 1.0, 0.25, format="%.2f%%",
                        help="Backtests usually look better than the future. This trims every "
                             "simulated year by this much.") / 100

with st.spinner("Running the walk-forward backtest for all four strategies..."):
    results, ips = backtest(mtime, lookback, rebalance_days, float(cost_bps), vol_cap, max_pos, shrink)

res, bench = results[strategy], results[BENCH]
table = E.metrics_table(results, cash)
years = retire_age - c["age"]
mc = simulate(mtime, res.returns, start_value, contrib, years, inflation, haircut)
prob = E.success_probability(mc, goal)
need80 = E.contribution_for(mc, goal, start_value, 0.80)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

if (DATA_DIR / "DEMO").exists():
    html('<div class="demo"><b>Simulated prices.</b> These numbers come from made-up data so the '
         'project runs offline. Run <code>python data_pull.py</code> for real market history.</div>')

period = f"{res.returns.index[0]:%b %Y} to {res.returns.index[-1]:%b %Y}"
then = (f"To be 80% confident, she would need to save about <b>{money(need80)}</b> a month."
        if need80 > contrib else
        f"She is already saving more than the {money(need80)} a month needed for 80% confidence.")
html(f"""
<div class="hero">
  <div>
    <p class="lede">Rivera Portfolio Lab by {AUTHOR}, a walk-forward backtest for a fictional client</p>
    <h1>Can Maya retire at {retire_age}?</h1>
    <p class="answer">Using <b>{PHRASES[strategy]}</b> and saving {money(contrib)} a month,
      <span class="prob{' low' if prob < 0.5 else ''}">{prob:.0%}</span> of 5,000 simulated
      futures reach {money(goal)} in today's dollars.</p>
    <p class="then">{then} Simulations reuse this strategy's own out-of-sample returns
      ({period}), trimmed by {haircut:.2%} a year.</p>
  </div>
  <dl class="facts">
    <dt>Age today</dt><dd>{c['age']}</dd>
    <dt>Years to retirement</dt><dd>{years}</dd>
    <dt>Saved so far</dt><dd>{money(start_value)}</dd>
    <dt>Monthly saving</dt><dd>{money(contrib)}</dd>
    <dt>Goal, today's dollars</dt><dd>{money(goal)}</dd>
    <dt>Largest fall she accepts</dt><dd>{ips.max_drawdown:.0%}</dd>
  </dl>
</div>
""")

# KPI strip: selected strategy versus the benchmark
m, b = table[strategy], table[BENCH]


def delta(v, ref, higher_is_better=True, kind="pct"):
    diff = v - ref
    if abs(diff) < (0.0005 if kind == "pct" else 0.005):
        return '<span class="same">same as 70/30</span>'
    good = (diff > 0) == higher_is_better
    txt = f"{diff:+.1%}" if kind == "pct" else f"{diff:+.2f}"
    return f'<span class="{"better" if good else "worse"}">{txt} vs 70/30</span>'


cells = [
    ("Annual return", f"{m['Annual return']:.1%}", delta(m["Annual return"], b["Annual return"])),
    ("Volatility", f"{m['Volatility']:.1%}", delta(m["Volatility"], b["Volatility"], False)),
    ("Sharpe ratio", f"{m['Sharpe ratio']:.2f}", delta(m["Sharpe ratio"], b["Sharpe ratio"], kind="x")),
    ("Worst fall", f"{m['Max drawdown']:.1%}", delta(m["Max drawdown"], b["Max drawdown"])),
    ("Costs per year", f"{m['Trading costs per year']:.2%}",
     f'<span class="same">{m["Avg. turnover per rebalance"]:.0%} turnover per rebalance</span>'),
]
html('<div class="strip">' + "".join(
    f'<div><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
    for k, v, d in cells) + "</div>")

tabs = st.tabs(["Performance", "Compare strategies", "Risk", "Holdings",
                "Retirement outlook", "Policy check", "How it works"])

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

with tabs[0]:
    log_scale = st.toggle("Log scale", value=False)
    fig = go.Figure()
    for name, color, width in [(BENCH, BRASS, 2), (strategy, TEAL, 3)]:
        w = 10_000 * (1 + results[name].returns).cumprod()
        fig.add_scatter(x=w.index, y=w, name=name, line=dict(color=color, width=width),
                        hovertemplate="%{y:$,.0f}")
    style(fig, 420, "Growth of $10,000 after trading costs", money_y=True)
    if log_scale:
        fig.update_yaxes(type="log")
    show(fig)

    fig = go.Figure()
    for name, color in [(BENCH, BRASS), (strategy, TEAL)]:
        dd = E.drawdown_series(results[name].returns)
        fig.add_scatter(x=dd.index, y=dd, name=name, line=dict(color=color, width=1.5),
                        fill="tozeroy" if name == strategy else None,
                        fillcolor="rgba(69,181,166,0.12)", hovertemplate="%{y:.1%}")
    fig.add_hline(y=-ips.max_drawdown, line=dict(color=LOSS, dash="dash", width=1.2),
                  annotation_text=f"Policy limit, -{ips.max_drawdown:.0%}",
                  annotation_position="bottom left", annotation_font_color=LOSS)
    show(style(fig, 300, "Fall from previous peak", pct_y=True))

    left, right = st.columns(2)
    with left:
        cal = pd.DataFrame({strategy: E.calendar_returns(res.returns),
                            BENCH: E.calendar_returns(bench.returns)})
        fig = go.Figure()
        for name, color in [(strategy, TEAL), (BENCH, BRASS)]:
            fig.add_bar(x=cal.index.astype(str), y=cal[name], name=name, marker_color=color,
                        hovertemplate="%{y:.1%}")
        style(fig, 360, "Calendar-year returns", pct_y=True)
        fig.update_layout(barmode="group", bargap=0.25)
        show(fig)
    with right:
        grid = E.monthly_grid(res.returns)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        lim = np.nanmax(np.abs(grid.values))
        fig = go.Figure(go.Heatmap(
            z=grid.values, x=[months[i - 1] for i in grid.columns], y=grid.index.astype(str),
            zmin=-lim, zmax=lim, colorscale=[[0, LOSS], [0.5, "#152238"], [1, TEAL]],
            text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.1%}")(grid.values),
            texttemplate="%{text}", textfont=dict(size=10), showscale=False,
            hovertemplate="%{x} %{y}: %{z:.2%}<extra></extra>", xgap=2, ygap=2))
        style(fig, 360, "Monthly returns", legend=False)
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        show(fig)

# ---------------------------------------------------------------------------
# Compare strategies
# ---------------------------------------------------------------------------

with tabs[1]:
    fig = go.Figure()
    for name, r in results.items():
        w = 10_000 * (1 + r.returns).cumprod()
        fig.add_scatter(x=w.index, y=w, name=name, hovertemplate="%{y:$,.0f}",
                        line=dict(color=STRATEGY_COLORS[name],
                                  width=3 if name == strategy else 1.6,
                                  dash="dot" if name == BENCH else "solid"),
                        opacity=1 if name in (strategy, BENCH) else 0.75)
    show(style(fig, 420, "All strategies, same dates, same costs", money_y=True))

    left, right = st.columns([1, 1.25])
    with left:
        fig = go.Figure()
        for name in results:
            fig.add_scatter(
                x=[table.loc["Volatility", name]], y=[table.loc["Annual return", name]],
                mode="markers+text", name=name, text=[name], textposition="top center",
                textfont=dict(size=11),
                marker=dict(size=14 + 22 * max(table.loc["Sharpe ratio", name], 0),
                            color=STRATEGY_COLORS[name],
                            line=dict(color="#152238", width=2)),
                hovertemplate=f"{name}<br>Return %{{y:.1%}}<br>Volatility %{{x:.1%}}<extra></extra>")
        style(fig, 420, "Return against risk", legend=False)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="Volatility", tickformat=".0%", showgrid=True, gridcolor="#22324B")
        fig.update_yaxes(title="Annual return", tickformat=".0%")
        show(fig)
        st.caption("Bigger circles have a higher Sharpe ratio. Up and to the left is better.")
    with right:
        st.markdown("### Scorecard")
        rows = [("Annual return", "pct", True), ("Volatility", "pct", False),
                ("Sharpe ratio", "x", True), ("Sortino ratio", "x", True),
                ("Max drawdown", "pct", True), ("Calmar ratio", "x", True),
                ("Worst month", "pct", True), ("Positive months", "pct0", True),
                ("Downside capture", "pct0", False), ("Upside capture", "pct0", True),
                ("Trading costs per year", "pct2", False)]
        names = list(results)
        head = "".join(f"<th>{n}</th>" for n in names)
        body = ""
        for row, kind, hib in rows:
            vals = table.loc[row, names]
            pool = vals.drop(BENCH) if row.endswith("capture") else vals
            best = pool.idxmax() if hib else pool.idxmin()
            fmt = {"pct": "{:.1%}", "pct0": "{:.0%}", "pct2": "{:.2%}", "x": "{:.2f}"}[kind]
            cells_html = "".join(
                f'<td class="{"best" if n == best else ""}">'
                f'{"n/a" if pd.isna(vals[n]) else fmt.format(vals[n])}</td>' for n in names)
            body += f"<tr><td>{row}</td>{cells_html}</tr>"
        html(f'<div class="tablewrap"><table class="clean"><tr><th></th>{head}</tr>{body}</table></div>')
        st.caption("Shaded cells are the best in each row. Capture ratios compare each "
                   "strategy's average up and down months with the benchmark's.")

# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

with tabs[2]:
    stress = E.stress_table(results)
    fig = go.Figure()
    for name in results:
        if name in stress.columns:
            fig.add_bar(y=stress.index, x=stress[name], name=name, orientation="h",
                        marker_color=STRATEGY_COLORS[name], hovertemplate="%{x:.1%}",
                        opacity=1 if name in (strategy, BENCH) else 0.55)
    style(fig, 420, "Stress tests: return during each market shock")
    fig.update_layout(barmode="group", hovermode="y unified", bargap=0.2)
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor="#22324B")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    show(fig)

    rs = E.rolling_stats(res.returns, cash)
    rb = E.rolling_stats(bench.returns, cash)
    left, right = st.columns(2)
    for col, label, fmt in [(left, "Rolling volatility", ".0%"), (right, "Rolling Sharpe", ".1f")]:
        with col:
            fig = go.Figure()
            fig.add_scatter(x=rb.index, y=rb[label], name=BENCH, line=dict(color=BRASS, width=1.5))
            fig.add_scatter(x=rs.index, y=rs[label], name=strategy, line=dict(color=TEAL, width=2))
            style(fig, 320, f"{label}, trailing 12 months")
            fig.update_yaxes(tickformat=fmt)
            show(fig)

    left, right = st.columns(2)
    last = res.weights.index[-1]
    recent = returns[E.ASSETS].iloc[-756:]
    with left:
        corr = recent.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
            colorscale=[[0, SLATE], [0.5, "#152238"], [1, TEAL]],
            text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10),
            showscale=False, xgap=2, ygap=2,
            hovertemplate="%{x} and %{y}: %{z:.2f}<extra></extra>"))
        style(fig, 440, "How the funds move together, last 3 years", legend=False)
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        show(fig)
    with right:
        w_now = res.weights.iloc[-1]
        cov = E.ledoit_wolf(recent) * 252
        rc = pd.Series(E.risk_contributions(w_now.values, cov), index=w_now.index)
        keep = (w_now > 0.001) | (rc.abs() > 0.001)
        fig = go.Figure()
        fig.add_bar(x=w_now.index[keep], y=w_now[keep], name="Share of money",
                    marker_color="#3A5A66", hovertemplate="%{y:.1%}")
        fig.add_bar(x=rc.index[keep], y=rc[keep], name="Share of risk",
                    marker_color=TEAL, hovertemplate="%{y:.1%}")
        style(fig, 440, f"Money versus risk, {last:%b %Y} allocation", pct_y=True)
        fig.update_layout(barmode="group", bargap=0.3)
        show(fig)
    st.caption("Stocks usually supply most of the risk even when they are well under half "
               "the money. Risk parity tries to even that out.")

# ---------------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------------

with tabs[3]:
    w_now = res.weights.iloc[-1].sort_values()
    left, right = st.columns([1, 1.4])
    with left:
        labels = [f"{t}  {E.UNIVERSE[t][0]}" for t in w_now.index]
        fig = go.Figure(go.Bar(
            x=w_now.values, y=labels, orientation="h",
            marker_color=[CLASS_COLORS[E.UNIVERSE[t][1]] for t in w_now.index],
            text=[f"{v:.1%}" for v in w_now.values], textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        style(fig, 440, f"Current target, set {last:%d %b %Y}", legend=False)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(tickformat=".0%", range=[0, max(w_now.max() * 1.25, 0.1)])
        show(fig)
        by_class = res.weights.iloc[-1].groupby(lambda t: E.UNIVERSE[t][1]).sum()
        st.caption("   ".join(f"{k}: {by_class.get(k, 0):.0%}" for k in E.ASSET_CLASSES))
    with right:
        W = res.weights
        fig = go.Figure()
        for t in E.ASSETS:
            fig.add_scatter(x=W.index, y=W[t], name=t, stackgroup="one", mode="lines",
                            line=dict(width=0.5, color=TICKER_COLORS[t], shape="hv"),
                            fillcolor=TICKER_COLORS[t], hovertemplate="%{y:.1%}")
        style(fig, 440, "Target weights at every rebalance", pct_y=True)
        fig.update_yaxes(range=[0, 1])
        show(fig)

    fig = go.Figure(go.Bar(x=res.turnover.index[1:], y=res.turnover.iloc[1:],
                           marker_color=SLATE, hovertemplate="%{y:.1%}<extra></extra>"))
    style(fig, 260, "Turnover at each rebalance", pct_y=True, legend=False)
    show(fig)

    uni = "".join(f"<tr><td><b>{t}</b></td><td style='text-align:left'>{d}</td>"
                  f"<td style='text-align:left'>{cls}</td><td>{er:.2%}</td>"
                  f"<td>{res.weights.iloc[-1][t]:.1%}</td></tr>"
                  for t, (d, cls, er) in E.UNIVERSE.items())
    html('<h3>The fund menu</h3><div class="tablewrap"><table class="clean"><tr><th>Ticker</th>'
         '<th style="text-align:left">Fund</th><th style="text-align:left">Asset class</th>'
         f'<th>Expense ratio</th><th>Weight now</th></tr>{uni}</table></div>')

# ---------------------------------------------------------------------------
# Retirement outlook
# ---------------------------------------------------------------------------

with tabs[4]:
    p = mc["percentiles"]
    ages = c["age"] + mc["months"] / 12
    fig = go.Figure()
    fig.add_scatter(x=ages, y=p["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=ages, y=p["p10"], fill="tonexty", fillcolor="rgba(69,181,166,0.12)",
                    line=dict(width=0), name="8 in 10 futures land here", hoverinfo="skip")
    fig.add_scatter(x=ages, y=p["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=ages, y=p["p25"], fill="tonexty", fillcolor="rgba(69,181,166,0.25)",
                    line=dict(width=0), name="Middle half", hoverinfo="skip")
    fig.add_scatter(x=ages, y=p["p50"], name="Median", line=dict(color=TEAL, width=3),
                    hovertemplate="Age %{x:.1f}: %{y:$,.0f}<extra></extra>")
    fig.add_hline(y=goal, line=dict(color=BRASS, dash="dash", width=1.5),
                  annotation_text=f"Goal {money(goal)}", annotation_position="top left",
                  annotation_font_color=BRASS)
    style(fig, 460, "Maya's savings in today's dollars, 5,000 simulated futures", money_y=True)
    fig.update_xaxes(title="Age")
    fig.update_layout(hovermode="x")
    show(fig)

    final = mc["final"]
    left, right = st.columns([1.4, 1])
    with left:
        fig = go.Figure(go.Histogram(x=final, nbinsx=60, marker_color="#3F8F86",
                                     hovertemplate="%{x}<br>%{y} futures<extra></extra>"))
        fig.add_vline(x=goal, line=dict(color=BRASS, dash="dash", width=1.5))
        style(fig, 340, f"Where she ends up at {retire_age}", legend=False)
        fig.update_xaxes(tickprefix="$", tickformat=",.2s")
        fig.update_layout(hovermode="closest")
        show(fig)
    with right:
        need90 = E.contribution_for(mc, goal, start_value, 0.90)
        rows = [("Chance of reaching the goal", f"{prob:.0%}"),
                ("Median outcome", money(np.median(final))),
                ("Bad case (1 in 10 do worse)", money(np.percentile(final, 10))),
                ("Good case (1 in 10 do better)", money(np.percentile(final, 90))),
                ("Monthly saving for 80% confidence", money(need80)),
                ("Monthly saving for 90% confidence", money(need90)),
                ("4% rule income from the median", money(np.median(final) * 0.04) + " a year")]
        html('<h3>What it means</h3><table class="clean">' +
             "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>")
    html(f'<p class="note">Each simulated future stitches together random 12-month stretches of '
         f'this strategy\'s real out-of-sample returns, removes {inflation:.2%} inflation, and '
         f'subtracts a {haircut:.2%} yearly haircut. It assumes Maya keeps saving the same '
         f'real amount every month and never panics out of the plan. It cannot invent crises '
         f'worse than the ones in the history it samples from.</p>')

# ---------------------------------------------------------------------------
# Policy check
# ---------------------------------------------------------------------------

with tabs[5]:
    check = E.ips_check(res, ips)
    rows = "".join(
        f"<tr><td>{r.Rule}</td><td>{r.Limit}</td><td>{r.Actual}</td>"
        f"<td class='{'pass' if r.Pass else 'fail'}'>{'Met' if r.Pass else 'Broken'}</td></tr>"
        for r in check.itertuples())
    verdict = ("follows every rule in Maya's policy" if check["Pass"].all()
               else "breaks at least one rule in Maya's policy")
    html(f"<h3>{strategy} {verdict}</h3>"
         '<div class="tablewrap"><table class="clean"><tr><th>Rule</th><th>Limit</th>'
         f"<th>Actual</th><th>Result</th></tr>{rows}</table></div>")

    st.markdown("### Every strategy at a glance")
    names = [n for n in results if n != BENCH]
    checks = {n: E.ips_check(results[n], ips) for n in names}
    head = "".join(f"<th>{n}</th>" for n in names)
    body = ""
    for i, rule in enumerate(checks[names[0]]["Rule"]):
        cells_html = "".join(
            f"<td class='{'pass' if checks[n].iloc[i]['Pass'] else 'fail'}'>"
            f"{'Met' if checks[n].iloc[i]['Pass'] else 'Broken'}</td>" for n in names)
        body += f"<tr><td>{rule}</td>{cells_html}</tr>"
    html(f'<div class="tablewrap"><table class="clean"><tr><th>Rule</th>{head}</tr>{body}</table></div>')
    st.caption("Weight rules are checked at every rebalance. Between rebalances weights drift "
               "with prices, as they would in a real account.")

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

with tabs[6]:
    html(f"""
<div class="note" style="color:{INK}; font-size:1rem; line-height:1.6">
<h3>The client</h3>
<p>Maya Rivera is 34, a fictional software engineer who wants to stop working at 55. She has
{money(c['start_value'])} invested, saves {money(c['monthly_contribution'])} a month, wants
{money(c['goal_real'])} in today's dollars, and has said she would abandon a plan that fell more than
25% from its high. Her Investment Policy Statement allows only low-cost ETFs (expense ratio 0.25%
or less), no shorting or leverage, 50% to 85% in stocks, at least 10% in bonds, at most 20% in real
estate and gold combined, and no single fund above 25%.</p>

<h3>The test</h3>
<p>At every rebalance, each strategy sees only the previous {lookback} years of prices, chooses its
weights, and holds them until the next rebalance while they drift with the market. It never sees
the future. Every trade pays {cost_bps} basis points, and the 70/30 benchmark (VT and BND) is
rebalanced on the same dates with the same costs. Sharpe and Sortino ratios use the actual T-bill
return (BIL) of each day rather than a fixed guess.</p>

<h3>Why these methods</h3>
<p>Plain max-Sharpe optimizers tend to pour money into whatever did best in the training window.
This project reduces that with Ledoit-Wolf shrinkage of the covariance matrix, by pulling expected
returns toward their average, and by capping expected volatility. Risk parity and minimum variance
ignore return forecasts entirely, and constrained equal weight is the humble baseline every clever
method has to beat.</p>

<h3>What this does not capture</h3>
<p>Taxes, bid-ask spreads beyond the flat cost, fund closures, and Maya's own behavior are not
modeled. About ten years of out-of-sample history is short, and it was a strong decade for US
stocks, which flatters the benchmark. Past results do not predict future ones. This is an
educational project, not investment advice.</p>
</div>
""")

html(f'<p class="note" style="margin-top:2.5rem; border-top:1px solid {LINE}; padding-top:1rem">Designed and built by {AUTHOR}, with AI assistance. Educational project, not investment advice.</p>')
