#!/usr/bin/env python3
"""BTC 4-hour-ahead forecaster for make.fun btc-cent-forecast-01.

Ensemble of two mean-reversion views of the last 72 hours of 1-minute
closes, blended and clipped in log space:

  1. A ridge regression on seven log-return / deviation features
     (returns over 1h/4h/12h/24h and deviations from the 12h/24h/72h
     mean log price), standardised with constants fitted on 40
     hourly checkpoints cut from the practice window
     (2026-11-07..2026-11-11) and regularised (lambda = 10).
  2. A small hand-damped mean-reversion combo on the 4h return,
     the 24h return and the deviation from the 24h mean.

Final log-return forecast = clip(0.7 * ridge + 0.3 * combo, +/-0.006),
so the guess never strays more than ~0.6% (~$450 at $75k) from the
checkpoint close.  The clip and the 0.7/0.3 blend were chosen because
they gave the most balanced error across three disjoint evaluations
on the practice window: 40 hourly checkpoints, 12 earlier 4-hourly
checkpoints, and the five practice cases whose targets are known
(p00-p04, recovered from the next case's checkpoint close).

Measured on the practice window (see README.md for the full table):
  persistence baseline  $437.89 / $418.94 / $470.55
  this ensemble         $319.66 / $388.31 / $347.81
  (hourly backtest / earlier 4-hourly / practice p00-p04, MAE in USD;
  the p00-p04 figure is the program's own output, run twice with
  byte-identical results. p05's target minute, 2026-11-11T06:00Z, lies
  outside the kit's practice window, so its error cannot be measured
  locally.)

Standard library only, deterministic, no network, no randomness.
Reads $MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from pathlib import Path

# Ridge constants fitted on the practice window (see module docstring).
FEATURE_MEANS = (
    -0.000314,  # r60
    -0.002014,  # r240
    -0.005456,  # r720
    -0.009539,  # r1440
    -0.002494,  # dev720
    -0.005120,  # dev1440
    -0.012120,  # dev4320
)
FEATURE_SDS = (
    0.004938,  # r60
    0.007474,  # r240
    0.010526,  # r720
    0.010769,  # r1440
    0.006422,  # dev720
    0.007855,  # dev1440
    0.006697,  # dev4320
)
RIDGE_WEIGHTS = (
    0.000066,   # r60
    -0.000620,  # r240
    0.001909,   # r720
    -0.002742,  # r1440
    -0.000050,  # dev720
    -0.000611,  # dev1440
    -0.001772,  # dev4320
)
RIDGE_INTERCEPT = -0.000843
RIDGE_WEIGHT = 0.7
COMBO_WEIGHT = 0.3
CLIP_LOG_RETURN = 0.006


def log_returns_and_deviations(log_prices):
    """Return the seven features for the last price in log_prices."""
    n = len(log_prices)
    last = log_prices[-1]

    def ret(k):
        k = min(k, n - 1)
        return last - log_prices[-1 - k] if k > 0 else 0.0

    def dev(k):
        k = min(k, n)
        window = log_prices[n - k:]
        return last - (sum(window) / len(window))

    return (
        ret(60),
        ret(240),
        ret(720),
        ret(1440),
        dev(720),
        dev(1440),
        dev(4320),
    )


def ridge_forecast(features):
    total = RIDGE_INTERCEPT
    for value, mean, sd, weight in zip(
        features, FEATURE_MEANS, FEATURE_SDS, RIDGE_WEIGHTS
    ):
        total += weight * (value - mean) / sd
    return total


def combo_forecast(features):
    _r60, r240, _r720, r1440, _dev720, dev1440, _dev4320 = features
    return -0.07 * r240 - 0.07 * r1440 - 0.10 * dev1440


def forecast_price(prices):
    """Forecast the close 240 minutes after the last price in prices."""
    last = float(prices[-1])
    if len(prices) < 120 or last <= 0.0:
        return last
    log_prices = [math.log(float(p)) for p in prices if float(p) > 0.0]
    if len(log_prices) < 120:
        return last
    features = log_returns_and_deviations(log_prices)
    log_move = RIDGE_WEIGHT * ridge_forecast(features)
    log_move += COMBO_WEIGHT * combo_forecast(features)
    log_move = max(-CLIP_LOG_RETURN, min(CLIP_LOG_RETURN, log_move))
    guess = last * math.exp(log_move)
    if not (math.isfinite(guess) and 0.0 < guess < 10000000.0):
        return last
    return round(guess, 2)


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    case_files = sorted(input_dir.glob("*.in"))
    if not case_files:
        print("no case files found in %s" % input_dir, file=sys.stderr)
        return 1
    for case_file in case_files:
        case_id = case_file.stem
        try:
            with open(case_file, "r", encoding="utf-8") as handle:
                case = json.load(handle)
            prices = case["markets"][0]["prices"]
            price = forecast_price(prices)
        except Exception as exc:  # never crash a case: fall back to last close
            print("case %s failed (%s); using persistence" % (case_id, exc),
                  file=sys.stderr)
            try:
                with open(case_file, "r", encoding="utf-8") as handle:
                    case = json.load(handle)
                price = float(case["markets"][0]["prices"][-1])
            except Exception:
                continue
        out_path = output_dir / ("%s.out" % case_id)
        payload = {"forecasts": [{"case": case_id, "price": price}]}
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
