# Experiment Protocol

## Data Sources

- Official candidate data: `candidate/<target>/data/train.csv` and `candidate/<target>/data/test.csv`.
- BaoStock expanded data: `BaoStock_Datasets/data/full_hfq_tencent/` or the latest BaoStock subdirectory under `BaoStock_Datasets/data/`.
- Future diagnostic data: local BaoStock future files such as `baostock_future_post_2026-03-20_2026-07-03.csv`.

## Known Baselines

Use these as context, not as immutable truth:

- Formal one-shot self-score baseline: about `0.067080603329`.
- C1 lightweight rerank candidate: about `0.068956328886153`.
- C1 improves self-score by changing portfolio weighting/rerank only, not by retraining the Transformer.
- Extended-data retraining candidates previously improved some internal validation windows but did not improve official one-shot self-score.

## Scoring Definitions

Self-score:

```text
sum(weight_i * (open_end_i - open_start_i) / open_start_i)
```

For the local official-style test this is one output portfolio over one five-trading-day interval. It is useful for competition targeting but is easy to overfit.

Walk-forward:

- predict at multiple dates
- evaluate future five-trading-day open-to-open returns
- report mean, median, hit rate, worst 20 percent mean, min, max, and drawdown proxy
- use 0, 10, and 20 bps cost sensitivity when available

## Experiment Hygiene

- Copy a candidate into an isolated run directory before editing.
- Keep official candidate packages clean.
- Save commands, logs, configuration, result CSV, self-score output, and walk-forward output.
- Prefer deterministic seeds and record all deviations.
- If CUDA is available, record GPU information before training.

## Selection Discipline

- Do not select a model only because it maximizes one local self-score window.
- Flag candidates with high self-score but weak walk-forward as high-score unstable.
- Flag candidates with strong walk-forward but weak self-score as robust but not promotion-ready.
- Promote only after the user explicitly asks for promotion.
