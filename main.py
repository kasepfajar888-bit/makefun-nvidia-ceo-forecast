#!/usr/bin/env python3
"""Quantile forecaster for Polymarket's "Next CEO of NVIDIA?" markets.

Entry for make.fun's polymarket-nvidia-ceo-forecast-01. For every case
file in the input directory the program writes one forecast file: for
each market, and for each of the 240 minutes after the checkpoint, the
0.05 / 0.25 / 0.5 / 0.75 / 0.95 quantiles of that minute's Yes price.

How the forecast is built, per market:

* Median. The checkpoint price itself. On the horizons scored here
  these markets are a step process around a level set by the crowd;
  the last day of history shows no drift that survives contact with
  the next four hours, so the median stays where the price is.

* Spread. The empirical distribution of h-minute price changes over
  the last 24 hours of the case's own history, read on a grid of
  horizons and interpolated in sqrt(h). This is the market's own
  realised travel, jumps included, not a fitted normal curve: a
  market that moved in steps of a cent gets quantiles a cent wide,
  a quiet market gets narrow ones.

* Guards. A floor keeps a market that sat still for a day from
  forecasting a zero-width future (it can still jump), and a cap
  keeps one wild day in the history from blowing the forecast wider
  than the house baseline's fixed random walk, the yardstick every
  entry is measured against.

Standard library only, no network, no clock, no randomness: the same
case always produces the same file, byte for byte.
"""

import glob
import json
import math
import os

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
H_GRID = (1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 180, 240)
LOOKBACK = 1440        # minutes of history behind the spread
FLOOR_SIGMA = 0.0016   # narrowest allowed spread, per sqrt minute
BASE_SIGMA = 0.0025    # the house baseline's width; our cap


def quantile(sorted_vals, tau):
    """The tau quantile of an already sorted list, interpolated."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def grid_quantiles(prices):
    """Empirical quantiles of h-step price changes, one row per grid
    horizon. Rows a short history cannot fill are borrowed from the
    nearest filled row, rescaled by sqrt(h)."""
    seg = prices[-LOOKBACK:]
    n = len(seg)
    grid = []
    for h in H_GRID:
        if n - h < 30:
            grid.append(None)
        else:
            diffs = sorted(seg[i + h] - seg[i] for i in range(n - h))
            grid.append([quantile(diffs, t) for t in LEVELS])
    known = [i for i, row in enumerate(grid) if row is not None]
    if not known:
        return [[Z[k] * BASE_SIGMA * math.sqrt(h) for k in range(5)]
                for h in H_GRID]
    filled = []
    for i, h in enumerate(H_GRID):
        if grid[i] is not None:
            filled.append(grid[i])
        else:
            j = min(known, key=lambda k: abs(H_GRID[k] - h))
            scale = math.sqrt(h / H_GRID[j]) if H_GRID[j] else 1.0
            filled.append([v * scale for v in grid[j]])
    return filled


def forecast_market(prices, horizon):
    """The 240 x 5 quantile rows for one market."""
    if not prices:
        # A listed market always holds at least its checkpoint price,
        # but a crash here would void the whole case; stay safe.
        return [[0.5] * 5 for _ in range(horizon)]
    x = prices[-1]
    grid_q = grid_quantiles(prices)
    sqrt_grid = [math.sqrt(h) for h in H_GRID]
    rows = []
    gi = 0
    for minute in range(1, horizon + 1):
        while gi < len(H_GRID) - 2 and H_GRID[gi + 1] < minute:
            gi += 1
        s0, s1 = sqrt_grid[gi], sqrt_grid[gi + 1]
        sm = math.sqrt(minute)
        w = 0.0 if s1 == s0 else (sm - s0) / (s1 - s0)
        w = min(1.0, max(0.0, w))
        qs = []
        for k in range(5):
            if k == 2:
                d = 0.0  # the median does not chase drift
            else:
                d = grid_q[gi][k] * (1.0 - w) + grid_q[gi + 1][k] * w
                if Z[k] < 0:
                    d = min(d, Z[k] * FLOOR_SIGMA * sm)
                    d = max(d, Z[k] * BASE_SIGMA * sm)
                else:
                    d = max(d, Z[k] * FLOOR_SIGMA * sm)
                    d = min(d, Z[k] * BASE_SIGMA * sm)
            q = x + d
            qs.append(0.0 if q < 0.0 else 1.0 if q > 1.0 else q)
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append([round(v, 6) for v in qs])
    return rows


def run_case(path, out_dir):
    with open(path, "r", encoding="utf-8") as f:
        case = json.load(f)
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = [{"id": m["id"],
                  "q": forecast_market([float(p) for p in m["prices"]],
                                       horizon)}
                 for m in case["markets"]]
    case_id = case.get("case") or os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(out_dir, case_id + ".out"), "w",
              encoding="utf-8") as f:
        json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        run_case(path, out_dir)


if __name__ == "__main__":
    main()
