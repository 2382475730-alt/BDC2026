# TEST: D-series overnight gap feature

Research-only, untrained lineage derived from D01's frozen code snapshot
at commit f3d3aeee62aeab979d3a7aa1cf082953ece0b8f6. It is not a retained
champion or submission package.

D01's `intraday_return` and `open_close_spread` both equal
`(close - open) / open`. This experiment retains intraday return and
replaces the duplicate slot with `overnight_return = (open - previous
observed close) / previous observed close`, grouped by stock after sorting.
The first observation is missing and follows the existing dataset's missing
value handling. This is an explicit research choice, not a recovered original
specification. There are still 19 stock features.

The feature schema has changed. Do not use D01 or Candidate-D weights or
scalers with this lineage: train a new model from scratch and fit new scalers
on training samples. Resume rejects checkpoints without exactly matching
feature columns. No model has been trained or evaluated for investment or
competition performance here. Existing candidates and frozen snapshots stay
unchanged.

From the repository root, check feature behavior with:

```bash
python -m unittest discover -s Experimental_WorkSpace/candidates_D/TEST_overnight_gap_20261101/tests
```

For a future research run, supply a valid data path and isolated config to
`src/train.py`; record data provenance, splits, environment and validation
protocol as required by the project instructions. Do not promote automatically.
