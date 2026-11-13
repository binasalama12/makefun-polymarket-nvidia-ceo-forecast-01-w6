# polymarket-nvidia-ceo-forecast-01 entry — Lark (w6)

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of
that minute's price to the output directory.

## Model

The sceptic variant — it distrusts the last print:

- **Shrunk centre:** 0.9 × last price + 0.1 × the 60-minute mean.
  One-minute spikes in these thin markets often half-retrace; when
  the last print agrees with the hour, the shrink does nothing.
- **Spread = a 35/65 blend** leaning Gaussian: empirical quantiles
  of h-minute changes at anchors 1, 3, 5, 15, 30, 60, 120, 240
  over the last 1440 minutes (interpolated in sqrt(h)), blended
  with a Gaussian walk whose sigma comes from the mean absolute
  one-minute change (floor 0.0012, cap 0.004).
- **Volatility regime:** last 6 hours vs the whole history, with
  the family's widest clamp, [0.6, 1.8].
- A market with too little history borrows the median sigma of the
  case's established markets.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. Deterministic; well under a second per case.

Backtested on the kit's practice inputs (later cases as ground
truth), this scores a mean pinball loss of about 0.362 against the
house baseline's 0.399 on the same folds.

## License

MIT — see `LICENSE`.
