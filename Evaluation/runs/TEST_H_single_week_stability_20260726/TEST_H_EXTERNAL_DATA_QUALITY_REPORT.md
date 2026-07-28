# TEST H External-Data Quality Gate

Status: `H0/H1 ENABLED; H2/H3/H4 BLOCKED_DATA_GATE`

## Dataset and grain summary

|Dataset|Grain|Rows|Key duplicates|Manifest hashes|
|---|---|---:|---:|---|
|Industry|stock|300|0|PASS|
|CFFEX|date|1331|0|PASS|
|Financial|stock + report period, latest revision only|7436|0|PASS|

## Branch decisions

|Branch|Decision|Reason|
|---|---|---|
|TEST_H0_CONTROL|ENABLED|Frozen D01 market snapshot reproduced exactly.|
|TEST_H1_TOPK_OBJECTIVE|ENABLED|Only the preregistered objective/checkpoint semantics change.|
|TEST_H2_FUTURES_INDUSTRY_RESIDUAL|BLOCKED_DATA_GATE|Industry is not historical point-in-time; small groups violate the >=5 stability rule; futures lacks spot-basis/annualized-term/risk-state fields and the prior consumer lag is 0.|
|TEST_H3_FINANCIAL_EVENT_RESIDUAL|BLOCKED_DATA_GATE|The source exposes latest revised values at one stock/report grain, not true versioned announcement snapshots. Conservative available_date prevents direct look-ahead but cannot recreate versions.|
|TEST_H4_CORPORATE_EVENT_RESIDUAL|BLOCKED_DATA_GATE|No compliant point-in-time increase/decrease/repurchase event dataset is present.|

## High-severity findings

- Industry: classification date 2026-07-26; 56/76 groups have fewer than five stocks, covering 40.0% of stocks.
- Futures: prior consumer lag=0; required corrected fields present={'spot_basis': False, 'annualized_term_structure': False, 'spot_basis_change': False, 'risk_state': False}.
- Financial: 83.0% of rows have update date after notice date; there is no version identifier and the grain is latest-revision-only.
- Corporate events: no compliant frozen dataset is present.

## Safe next action

Proceed only with H0/H1. Do not train H2-H4, do not access sealed labels, and do not reinterpret aggregate source validation passes as branch eligibility.

The full field coverage, age, robust-outlier, source URL/date, and hash evidence is in `TEST_H_EXTERNAL_DATA_QUALITY_AUDIT.json` and this executed notebook.
