# Three-candidate rolling validation run

This run directory contains isolated copies of `THU-BDC2026-main`,
`Candidate-C`, and `Candidate-D`. The protected packages under `candidate/`
remain unchanged.

The run completed on 2026-07-23. W1 and W2 each contain three independent clean
candidate copies, final checkpoints, logs, two byte-identical inference outputs,
and `prediction_lock.json` manifests. No prior-round checkpoint was reused.

- W1 shared input: 337,852 rows, 2021-10-12 through 2026-07-03, SHA256
  `DC76F0C68BA46C0D629AF37C5F7E280873D732702CB6CA1E375E89715946E07E`.
- W2 shared input: 339,352 rows, 2021-10-12 through 2026-07-10, SHA256
  `7E8CC2F5A6B685F21CAB17BA3A929D6DB8B15FFC2A5461C6CE1C3ED4B36D7F12`.
- All three candidate `train.csv` files match the round's shared hash.
- Candidate-D W2 used `raw_data_end=2026-07-10`,
  `train_sample_end=2026-07-03`, and 1,088 eligible samples.

The validation window is kept outside this project at:

`D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\BDC2026_Validation_20260706_20260718_Isolated`

The first-week feature rows were exposed only after all W1 prediction locks
existed. No 2026-07-13-or-later rows entered W2 training. Consolidated scores,
summaries, leakage audit, and the Chinese report are under `Evaluation/`; this
run remains research evidence and does not promote or modify `candidate/`.
