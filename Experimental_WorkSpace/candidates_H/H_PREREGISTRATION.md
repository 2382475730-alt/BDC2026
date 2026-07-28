# TEST H-Series 单周收益稳定性预注册

日期：2026-07-26  
状态：`TEST / RESEARCH ONLY / NOT SUBMISSION ELIGIBLE`

## 身份和不可变边界

- 正式对照：`D01_FrozenResearchChampion_20260709`。
- 正式提交对照：`candidate/Candidate-D/`，只做只读指纹，不运行探索训练。
- H 系列当前保留编号模型：`NONE`。
- 未经用户明确批准，不创建 `H01`，不向 `candidate/` 写入或复制产物。
- 所有 H 模型、日志、预测、注册表和报告必须包含 `TEST` 标记。
- 外部数据尚未与 2026-07-18 报备邮件逐项核对，只能用于 TEST。

## 数据、标签和时间

- 市场数据：
  `Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/data/train.csv`
- 标签：决策日 T 后，T+1 开盘买入、T+5 开盘卖出的收益。
- 训练起点：2022-01-04。
- 每个外层折前保留完整 T+5 purge；训练标签结束日必须早于首个外层决策日。
- 外层开发折：
  - 2025-01-03..2025-03-31
  - 2025-04-01..2025-06-30
  - 2025-07-01..2025-09-30
  - 2026-01-05..2026-03-27
- 封存标签：
  - 2025Q4（历史封存区）
  - 2026-04-03..2026-07-03（recent sealed）
- 封存标签在 finalist 策略锁生成前不可读取；若无 Serious PASS，则保持封存。

## 固定种子和预算

- D01 精确复现：20260708，12 epoch，原始 D01 配置。
- Quick：20260726、20260727，每个 inner/outer run 最多 3 epoch。
- Serious：在 Quick 基础上加入 20260728，每个 run 12 epoch。
- Quick 失败后不增加 epoch、不更换种子、不修改门槛挽救。

## 消融分支

1. `TEST_H0_CONTROL`
   - D01 数据、结构、特征、损失和优化器；
   - 只改变为本预注册的嵌套评估控制流程。
2. `TEST_H1_TOPK_OBJECTIVE`
   - 仅相对 H0 修改损失；
   - listwise 排序 + pairwise 边界；
   - hard-top-5 straight-through 组合收益；
   - downside softplus、batch CVaR20；
   - 正收益超过 8% 的 soft-cap 惩罚。
3. `TEST_H2_FUTURES_INDUSTRY_RESIDUAL`
   - 仅在历史点时行业分类和修正期货特征通过数据门后打开；
   - `final_rank = H1_rank + lambda * clip(aux_rank, -1, 1)`；
   - `lambda` 仅允许 0.00、0.05、0.10，并且只在 inner development 选择。
4. `TEST_H3_FINANCIAL_EVENT_RESIDUAL`
   - 仅在真实版本化公告快照可用后打开；
   - 最新修订值回填历史的数据不得训练此分支。
5. `TEST_H4_CORPORATE_EVENT_RESIDUAL`
   - 仅在合规、点时、可报备的增减持/回购数据可用后打开。

每个分支只改变一个主要变量。被数据门阻断的分支写明 `BLOCKED_DATA_GATE`，
不得用不合规替代数据强行训练。

## H1 固定损失

- listwise ranking：权重 1.00；
- pairwise boundary：权重 0.25，margin 0.05；
- hard-top-5 portfolio return：权重 0.05；
- downside：权重 0.08，temperature 0.02；
- CVaR20：权重 0.04；
- positive-tail soft-cap：权重 0.02，cap 0.08；
- hard top-k forward 必须与提交端 top-5 等权选择一致；
- 跨种子一致性只报告，不进入未来标签驱动的推理。

## 嵌套选择

- inner 窗口严格早于外层折，且与封存区不重叠；
- inner 最多训练 3 epoch，并按预注册的稳定性分数选择 epoch：
  `0.35*mean + 0.25*median + 0.20*CVaR20 + 0.10*q10 - 0.10*std`；
- outer 从同一固定种子重新初始化，只训练选定 epoch；
- outer 训练期间不查看 outer 指标，只在训练结束后评估一次；
- 样本物化白名单显式排除 2025Q4 与 2026-04-03..2026-07-03；
  价格历史仍可作为当时已知特征，但这些决策日的标签不进入样本对象；
- top-2/3/4/5 等权和固定 30/25/20/15/10 只作为预注册报告策略；
- Quick/Serious 的唯一主决策策略固定为 `TOP5_EQUAL`；其他策略仅作诊断，
  不得替代主策略通过门槛或用于事后挽救；
- 任何 top-k、权重或 lambda 都不得根据封存结果修改。

## Quick 硬门槛

- 至少 3/4 折相对同 seed、同 fold 的 H0/D01 控制为正；
- 最近折平均超额为正；
- 两个种子的结论方向一致；
- pooled 严格逐周胜率不低于 55%；
- 删除最佳两周后超额和仍为正；
- CVaR20 不差于控制；
- 最终输出策略 HHI 不高于 0.30；
- 最佳单周贡献不超过全部正收益之和的 20%。

任一关键门槛失败即标记 `REJECTED_AT_QUICK_GATE` 并停止该分支。

## Serious、策略锁和封存

- 最多两个 Quick PASS 分支进入 Serious；
- 三个固定种子、四折嵌套验证、12 epoch；
- 三个种子的平均超额均不为负；
- 至少两个种子独立达到 3/4 正向折；
- consensus 的中位数、CVaR20、去最佳两周超额均优于 D01；
- 最近折为正，严格逐周胜率不低于 55%；
- top-5 至少由 2/3 种子共同支持。

只有全部通过后才生成唯一 `TEST_H_FINALIST_LOCK.json`。封存评估只能由该锁
触发一次；否则不读取封存标签。
