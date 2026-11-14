#!/usr/bin/env python3
"""btc-cent-forecast-01: forecast Bitcoin's price 240 minutes after a checkpoint.

A small, deterministic, standard-library-only forecaster. It reads one case file
per run (/input/<id>.in, or $MAKEFUN_INPUT_DIR) holding 4,320 one-minute closes
of BTC/USDC over the 72 hours up to a checkpoint, and writes /output/<id>.out
(or $MAKEFUN_OUTPUT_DIR) with exactly {"forecasts": [{"case": <id>, "price": <n>}]}.

The model
----------
Work in log-price space. Let p be the checkpoint close (the last price in the
case), r60 the 60-minute log return, dev24 the log deviation of p from its
24-hour mean log price, and dev4 the same against its 4-hour mean log price.
The predicted 240-minute log return is

    pred = 0.20 * r60 - 0.25 * dev24 - 0.10 * dev4

i.e. a blend of short-horizon momentum continuation (the last hour's drift
persists a little) and mean reversion toward the 24h / 4h averages (the pool
price in RULES.txt is a single Orca Whirlpool quote, so large-trade excursions
fade). The weights were chosen by grid search on the six practice cases:
walk-forward over the practice window (80 checkpoints, every 30 min) and the
five practice cases whose targets are recoverable from the later cases.
The chosen weights beat the persistence baseline on both sets
(walk-forward MAE $366.60 vs $407.79; practice MAE $452.89 vs $470.55).

The prediction is clipped to +/-1.5 times the realised 240-minute volatility
(per-minute log-return standard deviation over the last 240 minutes, scaled by
sqrt(240)), a safety valve that never bound on the practice window but caps
the damage if the final window turns wild. The forecast is p * exp(pred),
rounded to the price grid's 5 decimals and clamped inside (0, 10,000,000).

Determinism: fixed-order float arithmetic only, no randomness, no time, no
network, no files besides the case inputs. Two runs produce byte-identical
output. Any per-case error falls back to the persistence guess (the checkpoint
close) so a case is still valid instead of crashing the run.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Smallest positive price on the grid: one unit of its 5 decimals.
MIN_PRICE = 0.00001
# Largest grid price below the scorer's exclusive 10,000,000-dollar cap.
MAX_PRICE = 9_999_999.99999

# Tuned model weights (see module docstring for how they were chosen).
W_R60 = 0.20      # short-horizon momentum continuation
W_DEV24 = -0.25   # mean reversion to the 24-hour mean log price
W_DEV4 = -0.10    # mean reversion to the 4-hour mean log price
CAP_MULT = 1.5    # clip |pred| to this multiple of realised 240-min volatility

# Lookback windows, in minutes.
N_R60 = 60
N_DEV24 = 1440
N_DEV4 = 240
N_VOL = 240


def price5(x):
    """x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def _mean(xs):
    return math.fsum(xs) / len(xs)


def _pstdev(xs):
    mu = _mean(xs)
    return math.sqrt(math.fsum((x - mu) * (x - mu) for x in xs) / len(xs))


def forecast(prices):
    """Forecast the price 240 minutes after the last close in `prices`."""
    if not prices:
        raise ValueError("no prices in case")
    last = float(prices[-1])
    if not math.isfinite(last) or last <= 0:
        raise ValueError("last price is not a positive finite number")
    if len(prices) < N_DEV24 + 1:
        # Not enough history for the full model: fall back to persistence.
        return price5(last)

    logs = [math.log(float(p)) for p in prices]
    if any(not math.isfinite(v) for v in logs):
        raise ValueError("non-finite log price")

    r60 = logs[-1] - logs[-1 - N_R60]
    dev24 = logs[-1] - _mean(logs[-N_DEV24:])
    dev4 = logs[-1] - _mean(logs[-N_DEV4:])

    # Realised per-minute volatility over the last 240 minutes.
    diffs = [logs[i + 1] - logs[i] for i in range(len(logs) - N_VOL - 1, len(logs) - 1)]
    vol240 = _pstdev(diffs) * math.sqrt(N_VOL)

    pred = W_R60 * r60 + W_DEV24 * dev24 + W_DEV4 * dev4
    cap = CAP_MULT * max(vol240, 1e-4)
    pred = max(-cap, min(cap, pred))

    return price5(last * math.exp(pred))


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(p for p in input_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with path.open("r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        prices = markets[0].get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError(f"{path.name} has no prices")
        try:
            guess = forecast(prices)
        except Exception:
            # Never crash a run and never leave a case unwritten: fall back
            # to the last finite positive close (the persistence guess), and
            # if the series holds no usable price at all, to a valid number
            # so the case still counts as valid instead of voiding the entry.
            guess = None
            for raw in reversed(prices):
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v > 0:
                    guess = price5(v)
                    break
            if guess is None:
                guess = 1.0
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        with (output_dir / f"{case_id}.out").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
