"""
data_pull.py
Step 1: download adjusted daily prices for the universe, the benchmark ETF (VT)
and the cash proxy (BIL), then save them to data/prices.csv.

Usage
    python data_pull.py            real prices from Yahoo Finance (needs internet)
    python data_pull.py --demo     simulated prices, for trying the project offline
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ALL_TICKERS

DATA_DIR = Path("data")
START = "2013-07-01"   # BNDX, the youngest fund, launched in June 2013


def download() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ALL_TICKERS, start=START, auto_adjust=True, progress=False)
    prices = raw["Close"][ALL_TICKERS].dropna(how="all")

    short = [t for t in ALL_TICKERS if prices[t].first_valid_index() > pd.Timestamp(START) + pd.Timedelta(days=10)]
    if short:
        print(f"Note: shorter history for {', '.join(short)}; rows before they start are dropped.")
    return prices.dropna()


def simulate() -> pd.DataFrame:
    """Plausible fake prices with fat tails and two crashes. Not real data."""
    rng = np.random.default_rng(2024)
    dates = pd.bdate_range(START, "2026-09-25")
    #            VTI   VXUS  VIG   USMV  BND   BNDX  TIP   SHY   VNQ   IAU    VT    BIL
    mu = np.array([.11, .07, .10, .09, .025, .025, .03, .02, .07, .06, .095, .02]) / 252
    vol = np.array([.17, .17, .15, .12, .05, .045, .06, .015, .21, .15, .16, .003]) / np.sqrt(252)
    groups = [0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 0, 2]
    base = np.array([[1, .2, .05, .6, .05], [.2, 1, .5, .15, .25], [.05, .5, 1, .02, .1],
                     [.6, .15, .02, 1, .1], [.05, .25, .1, .1, 1]])
    corr = base[np.ix_(groups, groups)] * 0.9
    np.fill_diagonal(corr, 1)
    L = np.linalg.cholesky(corr)
    z = rng.standard_t(df=4, size=(len(dates), len(mu))) / np.sqrt(2)
    r = mu + (z @ L.T) * vol
    stock_like = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0], dtype=bool)
    for start, days, hit in [("2020-02-20", 23, -0.012), ("2022-01-04", 190, -0.0012)]:
        i = dates.get_indexer([pd.Timestamp(start)], method="bfill")[0]
        r[i:i + days, stock_like] += hit
        if start.startswith("2022"):
            r[i:i + days, 4:7] -= 0.0006   # bonds fell in 2022 too
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=ALL_TICKERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--demo", action="store_true", help="use simulated prices")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    demo_flag = DATA_DIR / "DEMO"
    if args.demo:
        prices = simulate()
        demo_flag.write_text("Simulated prices. Run `python data_pull.py` for real data.\n")
        print("Created SIMULATED prices (not real market data).")
    else:
        prices = download()
        demo_flag.unlink(missing_ok=True)

    prices.to_csv(DATA_DIR / "prices.csv")
    print(f"Saved {len(prices):,} days x {prices.shape[1]} tickers "
          f"({prices.index[0].date()} to {prices.index[-1].date()}) to data/prices.csv")


if __name__ == "__main__":
    main()
