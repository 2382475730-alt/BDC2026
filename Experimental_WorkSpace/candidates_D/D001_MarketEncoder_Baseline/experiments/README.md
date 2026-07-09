# D001 Experiments

This directory is reserved for D001 run outputs.

Do not write long-running experiment artifacts into `candidate/` or `BaoStock_Datasets/`. A future D001 run should create a timestamped subdirectory here and record:

- config snapshot;
- source data path, date range, and md5;
- environment details;
- training logs;
- checkpoint paths;
- fold predictions and metrics;
- registry update status.

Known runs:

- `round1_D001_market_only_top2_20260708/`: first trained D001 round under the former C-equivalent last-two-month validation split.
- `r2_D001_three_set_top2_20260708/`: R2 run target for the three-set T+5 protocol.
