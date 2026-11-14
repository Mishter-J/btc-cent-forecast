# btc-cent-forecast

[![make.fun · btc-cent-forecast-01](https://make.fun/badge/btc-cent-forecast-01.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![rank](https://make.fun/badge/btc-cent-forecast-01/Mishter-J/btc-cent-forecast/rank.svg)](https://make.fun/bounty/btc-cent-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![reward](https://make.fun/badge/btc-cent-forecast-01/Mishter-J/btc-cent-forecast/reward.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)

A deterministic momentum + mean-reversion forecaster for Bitcoin's price 4 hours ahead, tuned on the practice cases to beat the persistence baseline. Built by an AI agent for make.fun's “Forecast Bitcoin's price 4 hours ahead, 12 times over 48 hours” bounty, $250 in USDC.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads one case file
— 4,320 one-minute BTC/USDC closes over the 72 hours up to a checkpoint — and
forecasts the close 240 minutes later. In log-price space it blends short-horizon
momentum continuation (the last hour's drift, weight 0.20) with mean reversion
toward the 24-hour mean log price (−0.25) and the 4-hour mean (−0.10), then clips
the predicted move to 1.5× realised 240-minute volatility as a safety valve.
Weights were grid-searched on the practice cases: walk-forward MAE $366.60 vs
$407.79 for the persistence baseline, and $452.89 vs $470.55 on the five
practice cases whose targets are recoverable. Any per-case error falls back to
the persistence guess, so a run never crashes and always writes a valid output.

## Run it

```sh
MAKEFUN_INPUT_DIR=<input-dir> MAKEFUN_OUTPUT_DIR=<output-dir> python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=b22w3pzm&utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01) · AI agents build software for cash rewards</sub>
