"""
run_backtest.py
Step 2: run all four strategies and the 70/30 benchmark through the
walk-forward backtest, print a scorecard, and save results to data/.

Usage
    python run_backtest.py
"""

from pathlib import Path

import pandas as pd

import engine as E

DATA_DIR = Path("data")


def pct(x):
    return f"{x:.1%}"


def main():
    returns = E.load_returns(DATA_DIR / "prices.csv")
    ips, settings = E.IPS(), E.Settings()
    if (DATA_DIR / "DEMO").exists():
        print("*** Using SIMULATED prices. Results are illustrative only. ***\n")

    print("Running walk-forward backtests (train 3 years, rebalance quarterly)...")
    results = E.run_all(returns, ips, settings)
    cash = returns[E.CASH_PROXY]
    table = E.metrics_table(results, cash)
    first = results["70/30 benchmark"].returns.index

    print(f"\nOut-of-sample period: {first[0].date()} to {first[-1].date()}\n")
    show = table.loc[["Annual return", "Volatility", "Sharpe ratio", "Max drawdown",
                      "Calmar ratio", "Trading costs per year"]]
    fmt = show.copy().astype(object)
    for row in fmt.index:
        f = (lambda v: f"{v:.2f}") if "ratio" in row else \
            (lambda v: f"{v:.2%}") if "costs" in row else pct
        fmt.loc[row] = [f(v) for v in show.loc[row]]
    print(fmt.to_string())

    print("\nPolicy check (worst fall from a peak must stay under "
          f"{ips.max_drawdown:.0%}):")
    for name, res in results.items():
        if name == "70/30 benchmark":
            continue
        check = E.ips_check(res, ips)
        status = "all rules met" if check["Pass"].all() else \
            "BROKE: " + ", ".join(check.loc[~check["Pass"], "Rule"])
        print(f"  {name:<26} {status}")

    print("\nStress periods:\n")
    print(E.stress_table(results).map(pct).to_string())

    # Save everything the dashboard or a spreadsheet might want.
    pd.DataFrame({n: r.returns for n, r in results.items()}).to_csv(DATA_DIR / "backtest_returns.csv")
    table.to_csv(DATA_DIR / "performance_summary.csv")
    for name, res in results.items():
        if name != "70/30 benchmark":
            slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
            res.weights.round(4).to_csv(DATA_DIR / f"weights_{slug}.csv")
    save_chart(results, ips)
    print("\nSaved results to data/ and a summary chart to backtest.png")


def save_chart(results, ips):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Max Sharpe (shrunk)": "#1F6F6B", "Risk parity": "#4B6A9B",
              "Minimum variance": "#7A8C5A", "Constrained equal weight": "#8E6C9E",
              "70/30 benchmark": "#B38B45"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for name, res in results.items():
        wealth = 10_000 * (1 + res.returns).cumprod()
        lw = 2.4 if name == "70/30 benchmark" else 1.6
        ax1.plot(wealth, label=name, color=colors[name], lw=lw)
        ax2.plot(E.drawdown_series(res.returns), color=colors[name], lw=1.2)
    ax2.axhline(-ips.max_drawdown, color="#B5473D", ls="--", lw=1, label="Policy limit")
    ax1.set_title("Growth of $10,000, after trading costs", loc="left")
    ax2.set_title("Fall from previous peak", loc="left")
    ax1.legend(frameon=False, ncol=3, fontsize=9)
    ax2.legend(frameon=False, fontsize=9)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("backtest.png", dpi=150)


if __name__ == "__main__":
    main()
