#!/usr/bin/env python3
"""Lark forecaster for Polymarket's "Next CEO of NVIDIA?" event
(make.fun polymarket-nvidia-ceo-forecast-01, wallet w6 variant).

Reads one case from the input directory: every market of the event
with its Yes price at each minute up to the checkpoint. Writes, for
each market and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

The Lark variant is the family sceptic about the last print:

- Its centre is NOT quite the last price. One-minute spikes in
  these thin markets often half-retrace, so the centre shrinks the
  last price 10% of the way toward the market's own 60-minute mean:
  centre = 0.9 * last + 0.1 * mean(last 60). When the last print is
  in line with the hour, the shrink does nothing; when it is a
  spike, the whole quantile fan is re-centred slightly.
- The blend leans Gaussian (35% empirical / 65% Gaussian), since
  the empirical table is built from raw changes that include the
  very spikes the centre corrects for.
- Anchors are the rich set 1, 3, 5, 15, 30, 60, 120, 240 over the
  last 1440 minutes; sigma is the mean-absolute-change estimator
  floored at 0.0012 and capped at 0.004.
- The regime factor gets the widest clamp of the family,
  [0.6, 1.8]: a market that has truly woken up is allowed a much
  wider fan, and a dead one a much narrower one.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded
to 6 decimals. Deterministic, standard library only.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

FALLBACK_SIGMA = 0.0025
WINDOW = 1440            # minutes of history for vol estimates
ANCHORS = [1, 3, 5, 15, 30, 60, 120, 240]
EMP_WEIGHT = 0.35
SIGMA_FLOOR = 0.0012
SIGMA_CAP = 0.004
MIN_EMP_HISTORY = 300
SHRINK = 0.10            # centre = (1-SHRINK)*last + SHRINK*mean60
MEAN_SPAN = 60


def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def price6(v):
    return round(clamp01(v) * 1e6) / 1e6 + 0.0


def quantile_sorted(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def mean_abs_diff(prices):
    if len(prices) < 2:
        return 0.0
    total = 0.0
    for i in range(len(prices) - 1):
        total += abs(prices[i + 1] - prices[i])
    return total / (len(prices) - 1)


def centre_of(prices):
    last = prices[-1]
    span = prices[-MEAN_SPAN:] if len(prices) > MEAN_SPAN else prices
    hour_mean = sum(span) / len(span)
    return (1.0 - SHRINK) * last + SHRINK * hour_mean


def anchor_quantiles(hist):
    out = {}
    n = len(hist)
    for h in ANCHORS:
        if h >= n:
            continue
        changes = sorted(hist[i + h] - hist[i] for i in range(n - h))
        out[h] = [quantile_sorted(changes, tau) for tau in QUANTILES]
    return out


def empirical_row(anchors_q, h):
    keys = sorted(anchors_q)
    if not keys:
        return None
    if h <= keys[0]:
        return anchors_q[keys[0]]
    if h >= keys[-1]:
        return anchors_q[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= h <= hi:
            f = (math.sqrt(h) - math.sqrt(lo)) / (math.sqrt(hi) - math.sqrt(lo))
            return [anchors_q[lo][j] * (1.0 - f) + anchors_q[hi][j] * f
                    for j in range(len(QUANTILES))]
    return anchors_q[keys[-1]]


def market_sigma(prices):
    window = prices[-WINDOW:] if len(prices) > WINDOW else prices
    return min(max(1.2533 * mean_abs_diff(window), SIGMA_FLOOR), SIGMA_CAP)


def regime_scale(prices):
    if len(prices) <= 720:
        return 1.0
    overall = mean_abs_diff(prices)
    if overall <= 1e-9:
        return 1.0
    recent = mean_abs_diff(prices[-361:])
    return min(1.8, max(0.6, recent / overall))


def forecast_market(prices, horizon, sigma):
    c = centre_of(prices)
    window = prices[-WINDOW:] if len(prices) > WINDOW else list(prices)
    anchors_q = anchor_quantiles(window) if len(window) >= MIN_EMP_HISTORY else {}
    scale = regime_scale(prices)
    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        gauss = [c + z * sigma * sq for z in Z]
        emp = empirical_row(anchors_q, h)
        if emp is None:
            row = gauss
        else:
            row = [EMP_WEIGHT * (c + emp[j]) + (1.0 - EMP_WEIGHT) * gauss[j]
                   for j in range(len(QUANTILES))]
        rows.append(sorted(price6(c + (v - c) * scale) for v in row))
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text())
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"{path.name} asks for quantiles "
                             f"{case['quantiles']}, not {QUANTILES}")
        horizon = int(case["horizon_minutes"])
        parsed = []
        for market in case["markets"]:
            prices = [float(p) for p in market["prices"]]
            if not prices:
                raise ValueError(f"market {market['id']} has no prices")
            parsed.append((market["id"], prices))
        sigmas = {mid: market_sigma(prices) for mid, prices in parsed
                  if len(prices) >= MIN_EMP_HISTORY}
        if sigmas:
            ordered = sorted(sigmas.values())
            fallback_sigma = ordered[len(ordered) // 2]
        else:
            fallback_sigma = FALLBACK_SIGMA
        forecasts = []
        for mid, prices in parsed:
            sigma = sigmas.get(mid, fallback_sigma)
            forecasts.append({"id": mid,
                              "q": forecast_market(prices, horizon, sigma)})
        out = json.dumps({"forecasts": forecasts}) + "\n"
        (OUTPUT_DIR / f"{case_id}.out").write_text(out)


if __name__ == "__main__":
    main()
