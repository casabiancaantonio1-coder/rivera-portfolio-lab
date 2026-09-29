"""
engine.py
The calculation core of the Rivera Portfolio Lab.

Everything the command-line script and the dashboard show comes from here:
the investment universe, the client's Investment Policy Statement (IPS),
four allocation strategies, a walk-forward backtest with trading costs,
performance and risk statistics, and a Monte Carlo retirement projection.

Nothing in this file downloads data or draws charts, so it is easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Universe, benchmark and client
# ---------------------------------------------------------------------------

# ticker: (description, asset class, expense ratio)
UNIVERSE = {
    "VTI": ("US total stock market", "Stocks", 0.0003),
    "VXUS": ("International stocks", "Stocks", 0.0005),
    "VIG": ("US dividend growers", "Stocks", 0.0005),
    "USMV": ("US minimum-volatility stocks", "Stocks", 0.0015),
    "BND": ("US total bond market", "Bonds", 0.0003),
    "BNDX": ("International bonds, currency hedged", "Bonds", 0.0007),
    "TIP": ("Inflation-protected Treasuries", "Bonds", 0.0018),
    "SHY": ("1-3 year Treasuries", "Bonds", 0.0015),
    "VNQ": ("US real estate", "Real assets", 0.0013),
    "IAU": ("Gold", "Real assets", 0.0025),
}
ASSETS = list(UNIVERSE)
ASSET_CLASSES = ("Stocks", "Bonds", "Real assets")

BENCHMARK = {"VT": 0.70, "BND": 0.30}  # global 70/30
CASH_PROXY = "BIL"                     # 1-3 month T-bills, used as the risk-free rate
ALL_TICKERS = ASSETS + ["VT", CASH_PROXY]

CLIENT = {
    "name": "Maya Rivera",
    "age": 34,
    "retire_age": 55,
    "start_value": 85_000,
    "monthly_contribution": 2_200,
    "goal_real": 1_000_000,   # in today's dollars
    "inflation": 0.025,
}

STRESS_PERIODS = {
    "Q4 2018 sell-off": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock": ("2025-02-19", "2025-04-08"),
}


@dataclass(frozen=True)
class IPS:
    """The client's written rules. Every strategy must obey these."""
    max_position: float = 0.25     # no single ETF above 25%
    min_stocks: float = 0.50
    max_stocks: float = 0.85
    min_bonds: float = 0.10
    max_real: float = 0.20         # real estate + gold combined
    max_drawdown: float = 0.25     # hard limit, checked after the fact
    vol_cap: float = 0.14          # expected volatility ceiling for the optimizer
    max_expense: float = 0.0025


@dataclass(frozen=True)
class Settings:
    """How the backtest is run."""
    lookback_days: int = 756       # about 3 years of training data
    rebalance_days: int = 63       # about one quarter
    cost_bps: float = 5.0          # trading cost per unit of turnover, in basis points
    mean_shrink: float = 0.5       # how far to pull expected returns toward their average
    n_starts: int = 6              # random starting points for the optimizer
    seed: int = 42


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance (daily), toward a scaled identity.

    Sample covariances from a few years of data are noisy; shrinking them
    makes optimizer weights far more stable from one rebalance to the next.
    """
    X = returns.values - returns.values.mean(axis=0)
    n, p = X.shape
    emp = X.T @ X / n
    mu = np.trace(emp) / p
    delta = emp.copy()
    delta.flat[:: p + 1] -= mu
    delta = (delta ** 2).sum() / p
    X2 = X ** 2
    beta = ((X2.T @ X2) / n - emp ** 2).sum() / (p * n)
    shrinkage = 0.0 if delta == 0 else min(beta, delta) / delta
    cov = (1 - shrinkage) * emp
    cov.flat[:: p + 1] += shrinkage * mu
    return cov


def shrunk_means(returns: pd.DataFrame, strength: float) -> np.ndarray:
    """Annual expected returns pulled toward their cross-sectional average.

    Raw historical averages are the main reason max-Sharpe portfolios pile
    into whatever did best recently.
    """
    mu = returns.mean().values * TRADING_DAYS
    return (1 - strength) * mu + strength * mu.mean()


def risk_contributions(weights, cov) -> np.ndarray:
    """Share of total portfolio variance coming from each holding (sums to 1)."""
    w = np.asarray(weights, dtype=float)
    port_var = w @ cov @ w
    return w * (cov @ w) / port_var


# ---------------------------------------------------------------------------
# Constrained optimizer shared by every strategy
# ---------------------------------------------------------------------------

def _class_index(cols):
    return {c: [i for i, t in enumerate(cols) if UNIVERSE[t][1] == c]
            for c in ASSET_CLASSES}


def _feasible_default(cols, ips: IPS) -> np.ndarray:
    """A simple portfolio that always meets the IPS; used as a safe fallback."""
    idx = _class_index(cols)
    w = np.zeros(len(cols))
    targets = {"Stocks": 0.60, "Bonds": 0.30, "Real assets": 0.10}
    for cls, share in targets.items():
        w[idx[cls]] = share / len(idx[cls])
    return w


def is_valid(w, cols, ips: IPS, tol: float = 1e-4) -> bool:
    idx = _class_index(cols)
    s, b, r = (w[idx[c]].sum() for c in ASSET_CLASSES)
    return (abs(w.sum() - 1) < tol and w.min() > -tol
            and w.max() <= ips.max_position + tol
            and ips.min_stocks - tol <= s <= ips.max_stocks + tol
            and b >= ips.min_bonds - tol and r <= ips.max_real + tol)


def _solve(objective, cols, ips: IPS, settings: Settings, extra=None) -> np.ndarray:
    n = len(cols)
    idx = _class_index(cols)
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": lambda w: w[idx["Stocks"]].sum() - ips.min_stocks},
        {"type": "ineq", "fun": lambda w: ips.max_stocks - w[idx["Stocks"]].sum()},
        {"type": "ineq", "fun": lambda w: w[idx["Bonds"]].sum() - ips.min_bonds},
        {"type": "ineq", "fun": lambda w: ips.max_real - w[idx["Real assets"]].sum()},
    ] + (extra or [])
    bounds = [(0.0, ips.max_position)] * n

    rng = np.random.default_rng(settings.seed)
    starts = [_feasible_default(cols, ips)]
    starts += [np.minimum(rng.dirichlet(np.ones(n)), ips.max_position)
               for _ in range(settings.n_starts)]

    best, best_val = None, np.inf
    for x0 in starts:
        res = minimize(objective, x0 / x0.sum(), method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        extra_ok = all(c["fun"](w) >= -1e-4 for c in (extra or []))
        if is_valid(w, cols, ips) and extra_ok:
            val = objective(w)
            if val < best_val:
                best, best_val = w, val
    return best if best is not None else _feasible_default(cols, ips)


# ---------------------------------------------------------------------------
# Strategies: each takes a training window and returns target weights
# ---------------------------------------------------------------------------

def max_sharpe(train: pd.DataFrame, rf: float, ips: IPS, settings: Settings):
    cols = list(train.columns)
    mu = shrunk_means(train, settings.mean_shrink)
    cov = ledoit_wolf(train) * TRADING_DAYS

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    vol_cap = {"type": "ineq", "fun": lambda w: ips.vol_cap - np.sqrt(w @ cov @ w)}
    return _solve(neg_sharpe, cols, ips, settings, extra=[vol_cap])


def risk_parity(train: pd.DataFrame, rf: float, ips: IPS, settings: Settings):
    cols = list(train.columns)
    cov = ledoit_wolf(train) * TRADING_DAYS
    target = 1 / len(cols)

    def spread(w):
        rc = risk_contributions(np.maximum(w, 1e-10), cov)
        return 1e3 * np.sum((rc - target) ** 2)

    return _solve(spread, cols, ips, settings)


def min_variance(train: pd.DataFrame, rf: float, ips: IPS, settings: Settings):
    cols = list(train.columns)
    cov = ledoit_wolf(train) * TRADING_DAYS
    return _solve(lambda w: 1e3 * (w @ cov @ w), cols, ips, settings)


def equal_weight(train: pd.DataFrame, rf: float, ips: IPS, settings: Settings):
    """The portfolio closest to 1/N that still follows the IPS."""
    cols = list(train.columns)
    target = np.ones(len(cols)) / len(cols)
    return _solve(lambda w: np.sum((w - target) ** 2), cols, ips, settings)


STRATEGIES = {
    "Max Sharpe (shrunk)": (max_sharpe,
        "Seeks the best return per unit of risk, using shrunk estimates and a "
        "volatility cap so it does not chase last year's winner."),
    "Risk parity": (risk_parity,
        "Spreads risk, not dollars, as evenly as the policy allows, so no single "
        "asset dominates how the portfolio moves."),
    "Minimum variance": (min_variance,
        "Aims for the smoothest ride the policy allows, ignoring return forecasts."),
    "Constrained equal weight": (equal_weight,
        "The simplest diversified mix: as close to equal weights as the policy allows."),
}


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    name: str
    returns: pd.Series                  # daily, after trading costs
    weights: pd.DataFrame               # target weights at each rebalance
    turnover: pd.Series                 # one-way turnover at each rebalance
    costs: pd.Series                    # cost paid at each rebalance (fraction of wealth)
    diagnostics: dict = field(default_factory=dict)


def _hold(test: pd.DataFrame, weights: np.ndarray, wealth: float):
    """Buy at target weights, let them drift with prices. Returns wealth path
    and the drifted weights on the last day."""
    growth = (1 + test).cumprod().values
    values = wealth * growth * weights
    total = values.sum(axis=1)
    return pd.Series(total, index=test.index), values[-1] / total[-1]


def run_backtest(returns: pd.DataFrame, strategy: str, ips: IPS, settings: Settings,
                 fixed_weights: dict | None = None) -> BacktestResult:
    """Walk forward: train on the past `lookback_days`, hold for `rebalance_days`.

    If `fixed_weights` is given, the same weights are used every time
    (this is how the benchmark is run, with identical dates and costs).
    """
    cols = list(fixed_weights) if fixed_weights else ASSETS
    R = returns[cols]
    cash = returns[CASH_PROXY]
    fn = None if fixed_weights else STRATEGIES[strategy][0]

    wealth, drifted = 1.0, np.zeros(len(cols))
    paths, weight_rows, turnover, costs = [], {}, {}, {}

    for i in range(settings.lookback_days, len(R), settings.rebalance_days):
        test = R.iloc[i:i + settings.rebalance_days]
        if len(test) < 5:
            break
        if fixed_weights:
            target = np.array([fixed_weights[c] for c in cols])
        else:
            train = R.iloc[i - settings.lookback_days:i]
            rf = cash.iloc[i - settings.lookback_days:i].mean() * TRADING_DAYS
            target = fn(train, rf, ips, settings)

        trade = np.abs(target - drifted).sum()
        # The very first purchase is counted as a trade too.
        cost = trade * settings.cost_bps / 1e4
        wealth *= (1 - cost)

        date = test.index[0]
        weight_rows[date], turnover[date], costs[date] = target, trade / 2, cost
        path, drifted = _hold(test, target, wealth)
        paths.append((path, wealth / (1 - cost)))
        wealth = path.iloc[-1]

    # Stitch daily returns; the first day of each period includes its trading cost.
    daily = []
    for path, start in paths:
        r = path.pct_change()
        r.iloc[0] = path.iloc[0] / start - 1
        daily.append(r)
    daily = pd.concat(daily)

    return BacktestResult(
        name=strategy,
        returns=daily,
        weights=pd.DataFrame(weight_rows, index=cols).T,
        turnover=pd.Series(turnover),
        costs=pd.Series(costs),
    )


def run_all(returns: pd.DataFrame, ips: IPS = IPS(), settings: Settings = Settings()):
    """Every strategy plus the benchmark, on identical dates."""
    results = {name: run_backtest(returns, name, ips, settings) for name in STRATEGIES}
    results["70/30 benchmark"] = run_backtest(returns, "70/30 benchmark", ips,
                                              settings, fixed_weights=BENCHMARK)
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def metrics(r: pd.Series, bench: pd.Series | None = None,
            cash: pd.Series | None = None) -> dict:
    r = r.dropna()
    years = len(r) / TRADING_DAYS
    rf_daily = cash.reindex(r.index).fillna(0) if cash is not None else 0 * r
    excess = r - rf_daily
    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = excess[excess < 0].std() * np.sqrt(TRADING_DAYS)
    cagr = (1 + r).prod() ** (1 / years) - 1
    mdd = drawdown_series(r).min()
    monthly = (1 + r).resample("ME").prod() - 1

    out = {
        "Annual return": cagr,
        "Volatility": vol,
        "Sharpe ratio": excess.mean() * TRADING_DAYS / vol if vol else np.nan,
        "Sortino ratio": excess.mean() * TRADING_DAYS / downside if downside else np.nan,
        "Max drawdown": mdd,
        "Calmar ratio": cagr / abs(mdd) if mdd else np.nan,
        "Worst month": monthly.min(),
        "Positive months": (monthly > 0).mean(),
    }
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        active = r - b
        te = active.std() * np.sqrt(TRADING_DAYS)
        out["Beta to benchmark"] = np.cov(r, b)[0, 1] / b.var()
        out["Tracking error"] = te
        out["Information ratio"] = active.mean() * TRADING_DAYS / te if te else np.nan
        bm = (1 + b).resample("ME").prod() - 1
        up, down = bm > 0, bm < 0
        out["Upside capture"] = monthly[up].mean() / bm[up].mean()
        out["Downside capture"] = monthly[down].mean() / bm[down].mean()
    return out


def metrics_table(results: dict, cash: pd.Series) -> pd.DataFrame:
    bench = results["70/30 benchmark"].returns
    table = {}
    for name, res in results.items():
        m = metrics(res.returns, None if name == "70/30 benchmark" else bench, cash)
        m["Avg. turnover per rebalance"] = res.turnover.iloc[1:].mean()
        m["Trading costs per year"] = res.costs.sum() / (len(res.returns) / TRADING_DAYS)
        table[name] = m
    return pd.DataFrame(table)


def calendar_returns(r: pd.Series) -> pd.Series:
    yearly = (1 + r).resample("YE").prod() - 1
    yearly.index = yearly.index.year
    return yearly


def monthly_grid(r: pd.Series) -> pd.DataFrame:
    m = (1 + r).resample("ME").prod() - 1
    grid = pd.DataFrame({"year": m.index.year, "month": m.index.month, "r": m.values})
    return grid.pivot(index="year", columns="month", values="r")


def rolling_stats(r: pd.Series, cash: pd.Series, window: int = TRADING_DAYS):
    ex = r - cash.reindex(r.index).fillna(0)
    vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    sharpe = ex.rolling(window).mean() * TRADING_DAYS / vol
    return pd.DataFrame({"Rolling volatility": vol, "Rolling Sharpe": sharpe}).dropna()


def stress_table(results: dict) -> pd.DataFrame:
    rows = {}
    for label, (a, b) in STRESS_PERIODS.items():
        row = {}
        for name, res in results.items():
            seg = res.returns.loc[a:b]
            if len(seg) > 0:
                row[name] = (1 + seg).prod() - 1
        if row:
            rows[label] = row
    return pd.DataFrame(rows).T


def ips_check(res: BacktestResult, ips: IPS) -> pd.DataFrame:
    """Did the strategy follow every rule at every rebalance?"""
    W = res.weights
    cls = {c: [t for t in W.columns if UNIVERSE[t][1] == c] for c in ASSET_CLASSES}
    stocks, bonds, real = (W[cls[c]].sum(axis=1) for c in ASSET_CLASSES)
    max_exp = max(UNIVERSE[t][2] for t in W.columns)
    mdd = -drawdown_series(res.returns).min()
    rows = [
        ("Largest single holding", f"at most {ips.max_position:.0%}",
         f"{W.max(axis=1).max():.1%}", W.max(axis=1).max() <= ips.max_position + 1e-3),
        ("Stocks", f"{ips.min_stocks:.0%} to {ips.max_stocks:.0%}",
         f"{stocks.min():.1%} to {stocks.max():.1%}",
         stocks.min() >= ips.min_stocks - 1e-3 and stocks.max() <= ips.max_stocks + 1e-3),
        ("Bonds", f"at least {ips.min_bonds:.0%}", f"lowest {bonds.min():.1%}",
         bonds.min() >= ips.min_bonds - 1e-3),
        ("Real estate and gold", f"at most {ips.max_real:.0%}", f"highest {real.max():.1%}",
         real.max() <= ips.max_real + 1e-3),
        ("Expense ratios", f"at most {ips.max_expense:.2%}", f"highest {max_exp:.2%}",
         max_exp <= ips.max_expense + 1e-9),
        ("Long only, no leverage", "weights between 0% and 100%, summing to 100%",
         f"lowest {W.min().min():.1%}", W.min().min() >= -1e-3),
        ("Worst fall from a peak", f"at most {ips.max_drawdown:.0%}", f"{mdd:.1%}",
         mdd <= ips.max_drawdown),
    ]
    return pd.DataFrame(rows, columns=["Rule", "Limit", "Actual", "Pass"])


# ---------------------------------------------------------------------------
# Monte Carlo retirement projection
# ---------------------------------------------------------------------------

def monte_carlo(daily_returns: pd.Series, start_value: float, monthly_contribution: float,
                years: int, inflation: float, haircut: float = 0.0,
                n_sims: int = 5000, block: int = 12, seed: int = 7) -> dict:
    """Block-bootstrap the strategy's own monthly returns into many possible futures.

    Returns are converted to today's dollars (inflation removed) and can be
    reduced by `haircut` per year, because backtests usually flatter a strategy.
    Blocks of 12 months keep realistic runs of good and bad years together.
    """
    monthly = ((1 + daily_returns).resample("ME").prod() - 1).values
    real = (1 + monthly) / (1 + inflation) ** (1 / 12) - 1 - haircut / 12
    months = years * 12
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(months / block))
    starts = rng.integers(0, len(real) - block + 1, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    paths_r = real[(starts[:, :, None] + offsets).reshape(n_sims, -1)[:, :months]]

    # Final value is linear in the contribution: V = start * G + c * A.
    growth = np.cumprod(1 + paths_r, axis=1)
    G = growth[:, -1]
    A = (G[:, None] / growth).sum(axis=1)  # each contribution compounds to the end

    values = np.empty((n_sims, months + 1))
    values[:, 0] = start_value
    for m in range(months):
        values[:, m + 1] = values[:, m] * (1 + paths_r[:, m]) + monthly_contribution

    pct = np.percentile(values, [10, 25, 50, 75, 90], axis=0)
    return {
        "months": np.arange(months + 1),
        "percentiles": pd.DataFrame(pct.T, columns=["p10", "p25", "p50", "p75", "p90"]),
        "final": values[:, -1],
        "G": G,
        "A": A,
    }


def success_probability(mc: dict, goal: float) -> float:
    return float((mc["final"] >= goal).mean())


def contribution_for(mc: dict, goal: float, start_value: float, confidence: float) -> float:
    """Monthly saving needed so `confidence` of simulated futures reach the goal."""
    needed = (goal - start_value * mc["G"]) / mc["A"]
    return max(0.0, float(np.quantile(needed, confidence)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_returns(path="data/prices.csv") -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = [t for t in ALL_TICKERS if t not in prices.columns]
    if missing:
        raise ValueError(f"prices.csv is missing: {', '.join(missing)}")
    return prices[ALL_TICKERS].dropna().pct_change().dropna()
