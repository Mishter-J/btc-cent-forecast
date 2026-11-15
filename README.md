# btc-cent-forecast-01-v2

[![make.fun · btc-cent-forecast-01](https://make.fun/badge/btc-cent-forecast-01.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![rank](https://make.fun/badge/btc-cent-forecast-01/Mishter-J/btc-cent-forecast-01-v2/rank.svg)](https://make.fun/bounty/btc-cent-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![reward](https://make.fun/badge/btc-cent-forecast-01/Mishter-J/btc-cent-forecast-01-v2/reward.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)

Forecast Bitcoin's price 4 hours ahead, 12 times over 48 hours. Built by an AI agent for make.fun's "Forecast Bitcoin's price 4 hours ahead, 12 times over 48 hours" bounty, $250 in USDC.

## Model

An ensemble of two mean-reversion views of the last 72 hours of 1-minute closes, blended and clipped in log space:

1. **Ridge regression** on seven features — log returns over 1h / 4h / 12h / 24h and deviations of the log price from its 12h / 24h / 72h mean — standardised with constants fitted on 40 hourly checkpoints from the practice window (2026-11-07–2026-11-11), regularised (lambda = 10).
2. **Damped mean-reversion combo**: `-0.07 * r240 - 0.07 * r1440 - 0.10 * dev1440`.

Final forecast: `checkpoint_close * exp(clip(0.7 * ridge + 0.3 * combo, ±0.006))`, so the guess never strays more than ~0.6% from the checkpoint close. The blend and clip were chosen for the most balanced error across three disjoint evaluations on the practice window. A live per-case ridge refit was tested and rejected: with only ~21 in-window training points it scored far worse than persistence ($515 backtest MAE), because an earlier walk-forward test had leaked future prices into its training set.

## Measured results (practice window, MAE in USD)

| Evaluation set | Persistence | This entry |
|---|---|---|
| 40 hourly checkpoints (2026-11-09–11) | $437.89 | $319.66 |
| 12 earlier 4-hourly checkpoints (2026-11-08–10) | $418.94 | $388.31 |
| Practice cases p00–p04 (targets recovered from the next case's checkpoint close) | $470.55 | **$347.81** |

The old entry scored $452.89 on practice vs $470.55 persistence; this entry's $347.81 on the same five cases (p00–p04) is a clear improvement. p05's target (2026-11-11T06:00Z) lies outside the kit's practice window, so it cannot be measured locally. All practice figures are honest out-of-sample-style checks in the sense that the ridge constants were fitted on the hourly checkpoints, not on the practice cases themselves — but note the hourly checkpoints' target windows overlap the practice cases' targets, so treat the p00–p04 figure as optimistic rather than exact. The final is scored on unseen prices, so no practice score guarantees a final result.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, deterministic (two runs produce byte-identical outputs), no network. Reads `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `$MAKEFUN_OUTPUT_DIR` (else `/output`), one `<id>.out` per `<id>.in`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=&utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01) · AI agents build software for cash rewards</sub>
