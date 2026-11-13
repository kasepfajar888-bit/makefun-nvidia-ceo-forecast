#!/usr/bin/env python3
"""Quantile forecaster for the Polymarket NVIDIA CEO markets.

The centre of every forecast is simply the last observed price.  How far
the other four quantiles sit away from that centre is learned, market by
market, from the market's own recent history: the empirical distribution
of price changes over a handful of horizons, measured on the last ten
hours of minutes, interpolated in square-root time, and finally squeezed
into a sane width band so that neither a dead-quiet stretch nor one wild
jump can produce absurd intervals.
"""

import json
import math
import os
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at LEVELS, as used by the house baseline.
Z_SCORES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)

HORIZON_GRID = (1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 180, 240)
WINDOW_MINUTES = 600
MIN_DIFFS = 30

SIGMA_FLOOR = 0.0005  # narrowest spread we are willing to report
SIGMA_CAP = 0.0028    # widest spread we are willing to report

_SQRT_GRID = tuple(math.sqrt(h) for h in HORIZON_GRID)


def _quantile_of_sorted(sorted_values, level):
    """One quantile of pre-sorted data, linear interpolation between
    the two neighbouring order statistics."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = level * (n - 1)
    low = int(math.floor(pos))
    high = low + 1
    if high >= n:
        return sorted_values[-1]
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def _grid_offsets(prices):
    """Empirical quantile offsets for every horizon in HORIZON_GRID.

    Returns a dict mapping horizon -> a list of five offsets (price
    change quantiles).  Horizons with too little data borrow the
    closest horizon that does have data, rescaled by sqrt time; if a
    market has too little data everywhere, a plain normal spread at
    the cap sigma is used for every horizon.
    """
    window = prices[-WINDOW_MINUTES:]
    rows = {}

    for step in HORIZON_GRID:
        diffs = [window[i + step] - window[i]
                 for i in range(len(window) - step)]
        if len(diffs) < MIN_DIFFS:
            rows[step] = None
            continue
        diffs.sort()
        rows[step] = [_quantile_of_sorted(diffs, level) for level in LEVELS]

    known = [step for step in HORIZON_GRID if rows[step] is not None]
    if not known:
        return {
            step: [z * SIGMA_CAP * math.sqrt(step) for z in Z_SCORES]
            for step in HORIZON_GRID
        }

    for step in HORIZON_GRID:
        if rows[step] is not None:
            continue
        source = min(known, key=lambda other: (abs(other - step), other))
        scale = math.sqrt(step / source)
        rows[step] = [value * scale for value in rows[source]]
    return rows


def _offsets_at(rows, minute):
    """Offsets for one forecast minute, interpolated in sqrt time."""
    if minute <= HORIZON_GRID[0]:
        return list(rows[HORIZON_GRID[0]])
    if minute >= HORIZON_GRID[-1]:
        return list(rows[HORIZON_GRID[-1]])

    # Locate the grid segment bracketing this minute.
    upper = 1
    while HORIZON_GRID[upper] < minute:
        upper += 1
    lower = upper - 1

    root = math.sqrt(minute)
    span = _SQRT_GRID[upper] - _SQRT_GRID[lower]
    weight = (root - _SQRT_GRID[lower]) / span
    blended = [
        a * (1.0 - weight) + b * weight
        for a, b in zip(rows[HORIZON_GRID[lower]], rows[HORIZON_GRID[upper]])
    ]
    return blended


def _checkpoint_price(prices):
    """A market's price at the checkpoint: its last recorded price.

    A market with no recorded price yet has no centre to anchor on;
    fall back to the mid-price rather than crashing the whole case.
    """
    return prices[-1] if prices else 0.5


def _forecast_market(prices, horizon):
    """The full quantile grid (horizon x 5) for a single market."""
    # The spread comes out of _grid_offsets, which degrades to the
    # normal fallback when there is no history to measure.
    last = _checkpoint_price(prices)
    rows = _grid_offsets(prices)

    forecast = []
    for minute in range(1, horizon + 1):
        root = math.sqrt(minute)
        offsets = _offsets_at(rows, minute)
        offsets[2] = 0.0  # the median stays on the last price

        values = []
        previous = 0.0
        for level_index, z in enumerate(Z_SCORES):
            offset = offsets[level_index]
            if z < 0.0:
                widest = z * SIGMA_CAP * root
                narrowest = z * SIGMA_FLOOR * root
                offset = min(max(offset, widest), narrowest)
            elif z > 0.0:
                narrowest = z * SIGMA_FLOOR * root
                widest = z * SIGMA_CAP * root
                offset = min(max(offset, narrowest), widest)
            else:
                offset = 0.0

            value = last + offset
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
            if value < previous:
                value = previous
            previous = value
            values.append(round(value, 6))
        forecast.append(values)
    return forecast


def _apply_coherence_shift(forecast, checkpoint, total):
    """Nudge one market's finished rows toward cross-market coherence.

    The markets of the event are mutually exclusive outcomes, so their
    prices should add up to one.  When the checkpoint prices instead
    add up to some total S0, each market is shifted by a share of the
    gap (1 - S0) / S0 proportional to its own checkpoint price, with
    the shift ramping in over the first hour of the horizon:
    delta(t) = (t / (t + 60)) * x * (1 - S0) / S0 at minute t.

    The shift is applied to the already-rounded quantile values; the
    shifted values are clamped to [0, 1] and forced non-decreasing,
    and deliberately not rounded a second time.
    """
    if total == 0.0:
        return forecast
    gap_share = (1.0 - total) / total
    shifted_forecast = []
    for index, row in enumerate(forecast):
        minute = index + 1
        delta = (minute / (minute + 60.0)) * checkpoint * gap_share
        shifted_row = []
        previous = 0.0
        for value in row:
            shifted = value + delta
            if shifted < 0.0:
                shifted = 0.0
            elif shifted > 1.0:
                shifted = 1.0
            if shifted < previous:
                shifted = previous
            previous = shifted
            shifted_row.append(shifted)
        shifted_forecast.append(shifted_row)
    return shifted_forecast


def _case_id(data, path):
    case = data.get("case")
    if case:
        return str(case)
    name = path.name
    return name[:-3] if name.endswith(".in") else path.stem


def _process_file(path, output_dir):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    horizon = int(data.get("horizon_minutes") or 240)
    markets = data.get("markets", [])

    # First pass: the case-level total of checkpoint prices, which the
    # coherence shift needs before any single market can be finished.
    histories = [market.get("prices") or [] for market in markets]
    checkpoints = [_checkpoint_price(prices) for prices in histories]
    total = sum(checkpoints)

    # Second pass: forecast each market on its own, then apply the
    # coherence shift that depends on that total.
    forecasts = []
    for market, prices, checkpoint in zip(markets, histories, checkpoints):
        rows = _forecast_market(prices, horizon)
        forecasts.append({
            "id": market["id"],
            "q": _apply_coherence_shift(rows, checkpoint, total),
        })

    out_path = output_dir / (_case_id(data, path) + ".out")
    text = json.dumps({"forecasts": forecasts}, separators=(",", ":"))
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(input_dir.glob("*.in")):
        _process_file(path, output_dir)


if __name__ == "__main__":
    main()
