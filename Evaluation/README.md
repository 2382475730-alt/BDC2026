# Evaluation 目录说明

更新时间：2026-07-08

`Evaluation/` 是本项目的统一评估工作区，用于集中管理 walk-forward、self-score、泄露审计、scoreboard 和评估报告相关脚本。

该目录不是提交包，不应被复制进最终比赛提交目录。

## 目录职责

```text
Evaluation/
  scripts/
    walk_forward/      # walk-forward 滚动评估脚本
    reporting/         # 评估报告生成脚本
    self_score/        # 后续统一 self-score 脚本预留区
    leakage_audit/     # 后续数据泄露审计脚本预留区
    scoreboard/        # 后续模型排行榜/对比脚本预留区
  configs/
    walk_forward/      # walk-forward 配置
    self_score/        # self-score 配置
    leakage_audit/     # 泄露审计配置
    scoreboard/        # 排行榜配置
  runs/                # 评估运行输出
  reports/             # 汇总报告输出
```

## 当前可用脚本

- `scripts/walk_forward/walk_forward_eval.py`
  用于对候选模型做泄露隔离的 walk-forward 滚动评估。
- `scripts/reporting/build_walk_forward_report.py`
  用于把多个 walk-forward 运行结果汇总成报告。
- `scripts/scoreboard/compare_d_vs_official_baseline.py`
  用于在 D001 当前验证标签上对比 D 系列当前模型与官方 baseline 冻结模型。

候选模型目录内的 `code/src/score_self.py`、`validation.py` 等文件是可复现包自身逻辑的一部分，暂时保留在各候选目录中，不在这里抽离。

## 推荐使用方式

从项目根目录运行评估脚本：

```powershell
$py = "python"
$eval = ".\Evaluation"

& $py "$eval\scripts\walk_forward\walk_forward_eval.py" `
  --candidate-dir ".\Experimental_WorkSpace\candidates_C\C1_Champion_RegimeGate" `
  --history-data ".\BaoStock_Datasets\data\full_hfq_baostock_20260703\baostock_future_post_2026-03-20_2026-07-03.csv" `
  --output-dir "$eval\runs\c1_formal_check" `
  --candidate-name C1 `
  --policy model_config `
  --window-mode frozen `
  --training-mode frozen_existing `
  --start-date 2026-03-23 `
  --model-asof-date 2026-03-20
```

生成汇总报告：

```powershell
& $py "$eval\scripts\reporting\build_walk_forward_report.py" `
  --runs-dir "$eval\runs" `
  --output "$eval\reports\walk_forward_report.md"
```

## 使用边界

- 不在这里保存训练出的模型权重。
- 不在这里修改 `candidate/` 下的提交包。
- 评估时可以使用未来区间收益做打分，但不能用未来收益训练模型、拟合 scaler、选择参数或构造推理规则。
- self-score 只作为单窗口诊断；模型泛化性主要看 walk-forward 和多窗口评估。

## D-Series Automation Scripts

`scripts/d_series/` contains shared D-series research-validation and Champion Gate utilities used by `Experimental_WorkSpace/candidates_D/run_d_auto_iteration.py`:

- `split_and_purge.py`
- `leakage_audit.py`
- `official_proxy_eval.py`
- `weekly_backtest.py`
- `metrics_v2.py`
- `paired_champion_gate.py`
- `experiment_registry.py`
- `production_refit.py`

These scripts are shared evaluation infrastructure. They do not store model weights, do not modify `candidate/`, and must call multi-week outputs validation/backtest rather than competition-style test.

