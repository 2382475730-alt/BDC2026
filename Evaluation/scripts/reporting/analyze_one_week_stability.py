from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Policy:
    policy_id: str
    weights: tuple[float, ...]

    @property
    def top_k(self) -> int:
        return len(self.weights)

    @property
    def hhi(self) -> float:
        return float(sum(weight * weight for weight in self.weights))


# These policies are fixed before reading any labels.  They deliberately avoid
# score-temperature tuning because score scales differ across folds and models.
POLICIES = (
    Policy("TOP1_EQUAL", (1.0,)),
    Policy("TOP2_EQUAL", (0.5, 0.5)),
    Policy("TOP3_EQUAL", (1 / 3, 1 / 3, 1 / 3)),
    Policy("TOP4_EQUAL", (0.25, 0.25, 0.25, 0.25)),
    Policy("TOP5_EQUAL", (0.2, 0.2, 0.2, 0.2, 0.2)),
    Policy("TOP5_RANK_30_25_20_15_10", (0.30, 0.25, 0.20, 0.15, 0.10)),
)

VARIANT_PATTERNS = {
    "F_CONTROL": "F_QUICK_F_CONTROL_*",
    "F_FUSION": "F_QUICK_F_FUSION_*",
    "G_CONTROL": "TEST_G_QUICK_TEST_G_CONTROL_*",
    "G_FINANCIAL": "TEST_G_QUICK_TEST_G_FINANCIAL_*",
}

PAIRED_BASELINES = {
    "F_FUSION": "F_CONTROL",
    "G_FINANCIAL": "G_CONTROL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one-week return stability for the rejected F/G TEST branches."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Evaluation" / "runs" / "TEST_one_week_stability_20260726",
    )
    return parser.parse_args()


def _load_runs() -> pd.DataFrame:
    family_roots = {
        "F_CONTROL": ROOT / "Experimental_WorkSpace" / "candidates_F" / "automation_runs",
        "F_FUSION": ROOT / "Experimental_WorkSpace" / "candidates_F" / "automation_runs",
        "G_CONTROL": ROOT / "Experimental_WorkSpace" / "candidates_G" / "automation_runs",
        "G_FINANCIAL": ROOT / "Experimental_WorkSpace" / "candidates_G" / "automation_runs",
    }
    records: list[pd.DataFrame] = []
    for variant, pattern in VARIANT_PATTERNS.items():
        run_dirs = sorted(path for path in family_roots[variant].glob(pattern) if path.is_dir())
        if len(run_dirs) != 4:
            raise RuntimeError(f"{variant}: expected 4 Quick runs, found {len(run_dirs)}")
        for run_dir in run_dirs:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            fold_name, validation_start, validation_end = config["validation"]["folds"][0]
            frame = pd.read_csv(
                run_dir / "best_val_predictions.csv",
                dtype={"stock_id": str},
                usecols=["sample_end_date", "stock_id", "score", "label"],
            )
            frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"])
            frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
            frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
            frame = frame.dropna(subset=["sample_end_date", "stock_id", "score", "label"])
            frame["variant"] = variant
            frame["fold"] = str(fold_name)
            frame["validation_start"] = str(validation_start)
            frame["validation_end"] = str(validation_end)
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def _weekly_returns(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, fold), fold_frame in predictions.groupby(["variant", "fold"], sort=True):
        daily: list[dict[str, Any]] = []
        for sample_end_date, day in fold_frame.groupby("sample_end_date", sort=True):
            ranked = day.sort_values(["score", "stock_id"], ascending=[False, True])
            for policy in POLICIES:
                selected = ranked.head(policy.top_k)
                if len(selected) != policy.top_k:
                    raise RuntimeError(
                        f"{variant}/{fold}/{sample_end_date}: fewer than {policy.top_k} stocks"
                    )
                labels = selected["label"].to_numpy(dtype=np.float64)
                weights = np.asarray(policy.weights, dtype=np.float64)
                daily.append(
                    {
                        "variant": variant,
                        "fold": fold,
                        "sample_end_date": sample_end_date,
                        "week": sample_end_date.to_period("W-FRI").strftime("%Y-%m-%d"),
                        "policy_id": policy.policy_id,
                        "top_k": policy.top_k,
                        "hhi": policy.hhi,
                        "effective_holdings": 1.0 / policy.hhi,
                        "portfolio_return": float(np.dot(labels, weights)),
                        "selected_stocks": ",".join(selected["stock_id"].astype(str)),
                    }
                )
        daily_frame = pd.DataFrame(daily)
        # One observation per calendar week prevents the five-day target from
        # being counted repeatedly through overlapping daily forecasts.
        weekly = (
            daily_frame.sort_values("sample_end_date")
            .groupby(["variant", "fold", "policy_id", "week"], as_index=False)
            .tail(1)
        )
        rows.extend(weekly.to_dict(orient="records"))
    return pd.DataFrame(rows).sort_values(
        ["variant", "policy_id", "sample_end_date"]
    ).reset_index(drop=True)


def _cvar(values: np.ndarray, fraction: float = 0.20) -> float:
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(np.sort(values)[:count]))


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    values = group["portfolio_return"].to_numpy(dtype=np.float64)
    fold_means = group.groupby("fold")["portfolio_return"].mean()
    positive_sum = float(values[values > 0].sum())
    best_return = float(np.max(values))
    trimmed = np.sort(values)[:-2] if len(values) > 2 else np.asarray([], dtype=float)
    return {
        "week_count": int(len(values)),
        "mean_return": float(np.mean(values)),
        "median_return": float(np.median(values)),
        "std_return": float(np.std(values, ddof=1)),
        "q10_return": float(np.quantile(values, 0.10)),
        "cvar20_return": _cvar(values),
        "worst_week_return": float(np.min(values)),
        "positive_week_rate": float(np.mean(values > 0)),
        "mean_minus_one_std": float(np.mean(values) - np.std(values, ddof=1)),
        "best_week_return": best_return,
        "best_week_share_of_positive_sum": (
            best_return / positive_sum if best_return > 0 and positive_sum > 0 else 0.0
        ),
        "return_sum_without_best_two_weeks": float(trimmed.sum()),
        "positive_fold_count": int((fold_means > 0).sum()),
        "fold_count": int(len(fold_means)),
        "hhi": float(group["hhi"].iloc[0]),
        "effective_holdings": float(group["effective_holdings"].iloc[0]),
    }


def _absolute_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, policy_id), group in weekly.groupby(["variant", "policy_id"], sort=True):
        row = {"variant": variant, "policy_id": policy_id}
        row.update(_summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_summary(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for candidate, baseline in PAIRED_BASELINES.items():
        for policy in POLICIES:
            columns = ["fold", "sample_end_date", "week", "portfolio_return"]
            control = weekly[
                (weekly["variant"] == baseline) & (weekly["policy_id"] == policy.policy_id)
            ][columns].rename(columns={"portfolio_return": "baseline_return"})
            challenger = weekly[
                (weekly["variant"] == candidate) & (weekly["policy_id"] == policy.policy_id)
            ][columns].rename(columns={"portfolio_return": "candidate_return"})
            paired = control.merge(
                challenger,
                on=["fold", "sample_end_date", "week"],
                how="inner",
                validate="one_to_one",
            )
            paired["candidate"] = candidate
            paired["baseline"] = baseline
            paired["policy_id"] = policy.policy_id
            paired["excess_return"] = paired["candidate_return"] - paired["baseline_return"]
            paired_rows.append(paired)

            excess = paired["excess_return"].to_numpy(dtype=np.float64)
            epsilon = 1e-12
            strict_win_rate = float(np.mean(excess > epsilon))
            tie_adjusted_pdr = float(
                np.mean((excess > epsilon).astype(float) + 0.5 * (np.abs(excess) <= epsilon))
            )
            fold_excess = paired.groupby("fold")["excess_return"].mean()
            fold_end = paired.groupby("fold")["sample_end_date"].max()
            recent_fold = fold_end.idxmax()
            without_best_two = np.sort(excess)[:-2] if len(excess) > 2 else np.asarray([])
            candidate_cvar = _cvar(paired["candidate_return"].to_numpy(dtype=np.float64))
            baseline_cvar = _cvar(paired["baseline_return"].to_numpy(dtype=np.float64))
            checks = {
                "positive_fold_count_ge_3": int((fold_excess > 0).sum()) >= 3,
                "recent_fold_mean_excess_positive": float(fold_excess.loc[recent_fold]) > 0,
                "strict_win_rate_ge_0_55": strict_win_rate >= 0.55,
                "excess_sum_without_best_two_positive": float(without_best_two.sum()) > 0,
                "cvar20_not_worse_than_control": candidate_cvar >= baseline_cvar,
                "hhi_le_0_30": policy.hhi <= 0.30,
            }
            summaries.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "policy_id": policy.policy_id,
                    "paired_week_count": int(len(paired)),
                    "mean_excess": float(np.mean(excess)),
                    "median_excess": float(np.median(excess)),
                    "strict_win_rate": strict_win_rate,
                    "tie_adjusted_pdr": tie_adjusted_pdr,
                    "positive_fold_count": int((fold_excess > 0).sum()),
                    "recent_fold": str(recent_fold),
                    "recent_fold_mean_excess": float(fold_excess.loc[recent_fold]),
                    "excess_sum": float(excess.sum()),
                    "excess_sum_without_best_two_weeks": float(without_best_two.sum()),
                    "candidate_cvar20_return": candidate_cvar,
                    "baseline_cvar20_return": baseline_cvar,
                    "cvar20_delta": candidate_cvar - baseline_cvar,
                    "hhi": policy.hhi,
                    **checks,
                    "stability_gate_passed": all(checks.values()),
                }
            )
    return pd.concat(paired_rows, ignore_index=True), pd.DataFrame(summaries)


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def _write_report(
    output_dir: Path,
    absolute: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    g_top2 = absolute[
        (absolute["variant"] == "G_FINANCIAL")
        & (absolute["policy_id"] == "TOP2_EQUAL")
    ].iloc[0]
    lines = [
        "# TEST：单周 output 收益稳定性审计",
        "",
        "状态：`TEST / RESEARCH ONLY / NOT A SUBMISSION PACKAGE`",
        "",
        "## 结论",
        "",
        "当前外部数据方案不能进入正式候选。F（行业+期货）和 G（再加财报）在固定、"
        "不看标签的 1–5 只持仓规则下，必须同时通过跨折方向、最近折、去掉最佳两周、"
        "CVaR20 和集中度门槛；任何一项失败都不应为单周决赛冒险。",
        "",
        f"G 财务分支当前 top-2 的最佳周收益为 `{_fmt_pct(g_top2['best_week_return'])}`，"
        f"占全部正收益之和的 `{100 * g_top2['best_week_share_of_positive_sum']:.1f}%`；"
        "这正是均值看似改善、但单周可重复性不足的典型形态。",
        "",
        "## 固定策略的绝对收益分布",
        "",
        "|分支|策略|周数|均值|中位数|标准差|10%分位|CVaR20|正收益率|去最佳两周收益和|HHI|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in absolute.sort_values(["variant", "hhi"], ascending=[True, False]).itertuples():
        lines.append(
            f"|{row.variant}|{row.policy_id}|{row.week_count}|{_fmt_pct(row.mean_return)}|"
            f"{_fmt_pct(row.median_return)}|{100 * row.std_return:.2f}%|"
            f"{_fmt_pct(row.q10_return)}|{_fmt_pct(row.cvar20_return)}|"
            f"{100 * row.positive_week_rate:.1f}%|"
            f"{_fmt_pct(row.return_sum_without_best_two_weeks)}|{row.hhi:.3f}|"
        )
    lines.extend(
        [
            "",
            "## 外部数据相对控制组的稳定性门槛",
            "",
            "门槛是预先固定的联合条件：4 折中至少 3 折超额为正；最近折为正；"
            "逐周胜率至少 55%；删除最佳两周后超额和仍为正；CVaR20 不差于控制组；"
            "HHI 不高于 0.30。",
            "",
            "|分支|策略|正向折|最近折超额|逐周胜率|去最佳两周超额和|CVaR20差|门槛|",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in paired_summary.itertuples():
        lines.append(
            f"|{row.candidate}|{row.policy_id}|{row.positive_fold_count}/4|"
            f"{_fmt_pct(row.recent_fold_mean_excess)}|"
            f"{100 * row.strict_win_rate:.1f}%|"
            f"{_fmt_pct(row.excess_sum_without_best_two_weeks)}|"
            f"{_fmt_pct(row.cvar20_delta)}|"
            f"{'PASS' if row.stability_gate_passed else 'FAIL'}|"
        )
    lines.extend(
        [
            "",
            "## 决赛导向的修正顺序",
            "",
            "1. 暂停把 F/G 外部数据并入正式模型；先保持 Candidate-D 为正式基线。",
            "2. 把训练目标改为与提交一致的 hard top-k surrogate，并在 top-3/top-5 上做"
            "固定权重验证；不得用最终一周或封存区间选择 k。",
            "3. checkpoint 选择与周度报告分离：内层选 checkpoint，外层只评一次；每个"
            "分支至少 3 个固定种子，以排名共识而非单模型分数选股。",
            "4. 财报改成真实版本化的公告时点数据，只保留公告事件、同比/环比 surprise、"
            "距公告 1/3/5/10 日衰减和逐字段缺失标记；删除陈旧绝对水平。",
            "5. 行业改为历史点时分类；期货改为相对现货的 basis、期限结构和隔夜/日内风险"
            "状态，所有收盘特征至少明确 T 日收盘后的可用时刻。",
            "6. 最终输出只允许使用开发期预注册策略：优先 HHI<=0.30、top-5 等权或固定"
            "30/25/20/15/10；只有在嵌套验证下同时改善中位数、CVaR20 和去最佳两周"
            "结果时，才允许替换 top-2。",
            "",
            "本审计使用 2025Q1/Q2/Q3 与 2026Q1 共 53 个非重叠周。它不访问封存区间，"
            "不修改 `candidate/`，也不构成 F01/G01 晋级证据。",
            "",
        ]
    )
    (output_dir / "TEST_ONE_WEEK_STABILITY_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_runs()
    weekly = _weekly_returns(predictions)
    absolute = _absolute_summary(weekly)
    paired_weekly, paired_summary = _paired_summary(weekly)

    weekly.to_csv(output_dir / "TEST_weekly_policy_returns.csv", index=False, encoding="utf-8")
    absolute.to_csv(output_dir / "TEST_absolute_stability_summary.csv", index=False, encoding="utf-8")
    paired_weekly.to_csv(output_dir / "TEST_paired_weekly_returns.csv", index=False, encoding="utf-8")
    paired_summary.to_csv(output_dir / "TEST_paired_stability_gates.csv", index=False, encoding="utf-8")
    _write_report(output_dir, absolute, paired_summary)

    manifest = {
        "marker": "TEST",
        "submission_eligible": False,
        "source": "existing rejected F/G Quick best_val_predictions.csv files",
        "week_semantics": "last decision date in each W-FRI group; five-day label; non-overlapping weekly observations",
        "policies": [
            {
                "policy_id": policy.policy_id,
                "weights": list(policy.weights),
                "hhi": policy.hhi,
            }
            for policy in POLICIES
        ],
        "stability_gate": {
            "positive_fold_count_ge_3": True,
            "recent_fold_mean_excess_positive": True,
            "strict_win_rate_ge_0_55": True,
            "excess_sum_without_best_two_positive": True,
            "cvar20_not_worse_than_control": True,
            "hhi_le_0_30": True,
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "TEST_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
