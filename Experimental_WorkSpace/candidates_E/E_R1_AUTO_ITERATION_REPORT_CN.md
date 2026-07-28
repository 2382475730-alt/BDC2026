# BDC2026 E系列 R1 自动迭代最终中文报告

- 终态：**REJECT_E_ALPHA_CHAMPION**。无配置通过冻结的Quick→Serious Alpha全门禁；Recent Holdout保持sealed。
- 冠军：D02 继续为 Alpha Champion；Defensive Champion 仍为 `NONE`。
- 研究墙钟：22.188 小时（24 小时预算内）；成功训练单元 150，归档基础设施中断尝试 1。
- Serious / E4 / Recent Holdout / Production Refit / E01：均未触发；`candidate/`、D02、D03、D04 未写入。

## 1. 数据、标签与冻结协议

- 只读源：`c0_train_plus_baostock_future_20260703.csv`，50,670,372 bytes，MD5 `78759d892c95eb8ba547c2df9a8fca5a`，SHA256 `8e9cf37e5c53e4edafddc3f6f6b5843d4037376334e6f53cba068699095df1ad`；388,452 行、1,331 个交易日、300 只股票、2021-01-04..2026-07-03。
- 开发快照通过只解析日期字段的流式过滤产生，截止 2026-03-27；368,952 行、1,266 日、300 只，SHA256 `534443dab2a6cd3a491913e4c7fe7103c05a068de6c24c6b7aed23716ba6f115`。未解析 sealed 数值。
- 监督样本从 2022-01-04；标签固定为全局交易日 `open(T+5)/open(T+1)-1`。沿用 R5 四个 expanding folds、inner、T+5 purge 与 47 个开发周。
- 交易：决策日只用当日及以前数据，下周首个交易日开盘买、最后/T+5 开盘卖；净成本为买佣金 0.05% 加卖佣金 0.05% 与印花税 0.05%，卖出部分按实际卖出金额计。

## 2. E001 算法与自动迭代

- E1 去除 8 个 Factor 别名和 1 组完全重复特征，并引入 train-only 有限值 scaler、横截面百分位、60/120 日稳健 Z、5/20/60 日动量斜率、回撤修复、波动比、量价确认、市场宽度/离散度/换手/波动水平及变化。
- Stable Expert 读取 60/120 日；Recent Expert 读取 20 日；Market GRU 读取 60 日；门控严格满足 `score = stable_score + g * recent_residual`，再做跨股票注意力。
- 排序损失前向与推理同为硬 Top-2 等权，反向使用软边界直通梯度；近期权重只比较预注册的 252/126 交易日半衰期。
- 顺序完成 E0→E1→E2→E3。只有排序器先过 Quick 才允许 E4；因此 P1/P2/P3 只通过单元约束测试，没有被用来掩盖排序失败。

- 实施前由算法架构、特征与泄露、统计评估、比赛合规四个只读智能体分别提交机制假设、最小实验、失败条件和风险；主智能体裁决写入 `analysis_reviews/E_R1_ADVISORY_SYNTHESIS.md` 与预注册，智能体未写文件或占用 GPU。

## 3. E0、官方基线与比较口径

- Legacy 固定区间：官方 baseline gross proxy `0.012528499297`；D02 `0.037119835763`，D02 明确超过官方 baseline。
- E0 只读复现 D02 的 12 个 R5 控制 run，47 周完整，且在 E 内按实际卖出金额重算成本；未写 D 系列。
- E Quick 的 `median official-proxy` 是开发折诊断，不能冒充 Legacy 同区间的官方 baseline 复训结果。由于所有 E 配置还同时失败至少一个其他硬门禁，终态对“有效官方同协议改善”采取 fail-closed：**未成立**，不会声称进步或晋级。

## 4. Quick 结果与冻结门禁

|配置|均值净超额|PDR|非负折|F1|F2|F3|F4/2026Q1|去最佳2周|MDD比|CVaR比|median proxy|失败门禁|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|E1A_DEDUP_ST60|0.003843|0.426|3/4|0.016283|0.015049|0.002959|-0.020990|-0.203398|1.131|1.322|0.018217|pdr, f4, without_best2, mdd, cvar|
|E1B_CAUSAL_SCALE_ST60|0.003460|0.447|3/4|0.016283|0.015049|0.002959|-0.022624|-0.221372|1.226|1.322|0.016957|pdr, f4, without_best2, mdd, cvar|
|E1C_FULL_CAUSAL_ST60|-0.001742|0.511|1/4|-0.021209|0.026409|-0.001186|-0.011820|-0.275303|0.890|1.305|0.006219|official_baseline, mean_excess, folds, f4, without_best2, cvar|
|E2_MARKET_GRU60|0.003880|0.574|3/4|-0.016921|0.011365|0.003916|0.018366|-0.043692|0.644|0.896|0.016459|without_best2|
|E3_MIX_HL252|-0.005094|0.489|2/4|-0.027827|0.011258|-0.031033|0.030164|-0.573829|0.708|0.994|0.012132|official_baseline, mean_excess, pdr, folds, without_best2|
|E3_MIX_HL126|-0.003539|0.511|2/4|-0.029209|0.012935|-0.024385|0.029237|-0.500723|0.645|0.994|0.010504|official_baseline, mean_excess, folds, without_best2|

### seed 稳定性

|配置|seed 20260720|seed 20260721|seed 20260722|相对D02为正seed|
|---|---:|---:|---:|---:|
|E1A_DEDUP_ST60|-0.002459|-0.005155|0.019142|1/3|
|E1B_CAUSAL_SCALE_ST60|-0.002459|-0.006302|0.019142|1/3|
|E1C_FULL_CAUSAL_ST60|-0.003540|-0.002480|0.000795|1/3|
|E2_MARKET_GRU60|0.010562|-0.006284|0.007362|2/3|
|E3_MIX_HL252|0.002957|-0.023381|0.005141|2/3|
|E3_MIX_HL126|0.001355|-0.019257|0.007287|2/3|

### 验证 T+5 窗口平均收益（新增参考指标）

定义：每个验证决策日计算 `Σ weight × [open(T+5)/open(T+1)-1]`，同一 fold/date 先平均 3 seeds，再平均四折 208 个验证决策日。该值为毛收益，仅作事后报告参考，不改变原 Quick 门禁；实际成本周净指标仍是主指标。

|模型|T+5平均毛收益|相对D02|F1|F2|F3|F4/2026Q1|
|---|---:|---:|---:|---:|---:|---:|
|D02_R5_CONTROL|0.021352|0.000000|0.029933|0.002002|0.052219|-0.001676|
|E1A_DEDUP_ST60|0.022748|0.001396|0.027997|0.018908|0.054859|-0.014472|
|E1B_CAUSAL_SCALE_ST60|0.022932|0.001580|0.027997|0.018908|0.055605|-0.014541|
|E1C_FULL_CAUSAL_ST60|0.014817|-0.006535|0.014001|0.016760|0.037310|-0.011580|
|E2_MARKET_GRU60|0.022911|0.001559|0.019974|0.013262|0.048203|0.007555|
|E3_MIX_HL252|0.016978|-0.004374|0.004052|0.013703|0.026002|0.023205|
|E3_MIX_HL126|0.019413|-0.001939|0.003498|0.016641|0.027709|0.028921|

- E2 最接近 Quick：均值超额、PDR、3/4 折、F4、MDD、CVaR均通过，但去最佳两周后累计超额为 `-0.043692`，显示收益集中于极少数周，按预注册拒绝。
- 新增 T+5 参考指标最高的是 `E1B_CAUSAL_SCALE_ST60`（2.2932%），略高于 E2（2.2911%）和 D02（2.1352%）；但 E1B 的 F4 T+5 均值为负且原 Quick 多项门禁失败，因此不改变“E2 综合最佳、无人晋级”的裁决。
- E3-HL252/HL126 在 F4 均较强且风险较低，但 pooled 均值分别为 `-0.005094`/`-0.003539`，近期专家未带来稳定 Alpha。
- 无 Quick PASS，因此 Serious 5-seed、Alpha 精确重复、E4、finalist lock 和 Recent Holdout 均合法跳过。

## 5. 仓位、现金、成本与风险

|配置|周毛收益|周净收益|平均周成本|47周成本拖累|换手|股票仓位|现金|持股数|HHI|最大单股|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|E1A_DEDUP_ST60|0.015832|0.014316|0.001516|0.071244|0.511|1.000|0.000|2.0|0.500|0.500|
|E1B_CAUSAL_SCALE_ST60|0.015449|0.013933|0.001515|0.071226|0.514|1.000|0.000|2.0|0.500|0.500|
|E1C_FULL_CAUSAL_ST60|0.010242|0.008731|0.001510|0.070981|0.404|1.000|0.000|2.0|0.500|0.500|
|E2_MARKET_GRU60|0.015869|0.014353|0.001516|0.071246|0.479|1.000|0.000|2.0|0.500|0.500|
|E3_MIX_HL252|0.006886|0.005379|0.001507|0.070824|0.472|1.000|0.000|2.0|0.500|0.500|
|E3_MIX_HL126|0.008443|0.006934|0.001508|0.070897|0.447|1.000|0.000|2.0|0.500|0.500|

- P0 是排序控制，故 Top-2、满仓、现金 0、HHI 0.5；它不是 E 主策略。P1-P3 的 3-5 只、q∈[0.5,1]、投资部分单股≤35%、HHI≤0.34 已通过测试，但因排序未过门禁而未评估收益。
- Defensive Champion 也不成立：没有配置同时满足官方门槛、至少保留 D02 proxy 的 90%、且 MDD 与 CVaR 均下降至少 15%。E2 虽 MDD 降约 35.6%，但 proxy 保留率约 79.8%、CVaR 仅降约 10.4%。

## 6. 门控、过拟合与泄露判断

- `E3_MIX_HL252`：gate mean/p05/p50/p95 = 0.317/0.201/0.311/0.462；recent residual std=0.712。
- `E3_MIX_HL126`：gate mean/p05/p50/p95 = 0.311/0.200/0.304/0.459；recent residual std=0.733。
- E1A/E1B 主要依赖单一 seed 且 F4 转负；E2 收益集中于最佳两周；E3 对 seed 20260721 明显为负。证据支持“高方差/时间不稳定”，不支持稳定 Alpha 改善。
- 72/72 outer leakage audit 通过；真实 368,952 行截断等价在 2025-09-19、2026-03-20 均通过。9 项 unittest 全通过。
- 固定 300 股票池缺少历史成分变更，存在 survivor/universe 风险；本轮禁止使用成分身份特征，并在报告中保留该限制。
- 官方同协议 baseline 复训证据不足是比较限制；终态已 fail-closed，不把开发 proxy 误称为官方改善。

## 7. 产物、确定性与异常恢复

- 完整性：smoke 6/6、inner 72/72、outer 72/72；每个单元的周诊断、指标、leakage 和 manifest 均齐全，SHA manifest 验证结果=PASS。Smoke/inner明确标记为非门禁证据。
- Quick 144 个代码快照与最终源码一致：PASS，统一 SHA256 `b64289eaf3cfa3ce87478af4be1560eb1ea725d179e67bbfc29784ac18ed0d33`。
- `0x40010004` 外部终止尝试已完整归档；无模型异常、无 checkpoint。随后同代码/配置/seed/data 幂等重试成功，没有删掉失败证据。
- checkpoint 精确恢复、score contract 与 sealed lock 均由自动化测试覆盖；未发生 finalist 后解封。

## 8. 训练与预测耗时

- 150 个成功单元记录训练总计 387.6 分钟；单元最大 6.709 分钟，P95 5.294 分钟，均远低于 7 小时。
- 独立推理速度审计：E2 冻结 outer checkpoint 在完整开发快照上 25.066 秒，2 只股票、权重和 1.0、stderr 0，低于 4 分钟目标。该 CSV 明确标记为 speed-audit-only。
- 因无人晋级，Production Refit 未运行，所以不能把 Quick 时长冒充最终可提交训练程序的正式 7/8 小时认证。

## 9. 终态审计与冠军裁决

- 预注册哈希锁：PASS；sealed access count=0；E01=0；Serious=0；E4=0。
- 保护目录：PASS；candidate/D02/D03/D04 在控制器开始后均无文件 mtime 变化。
- 注册表终态对账：150 个成功 run ids，状态 `REJECT_E_ALPHA_CHAMPION`。
- 后台进程：无。
- 最终裁决：**REJECT_E_ALPHA_CHAMPION**。不创建 E01，不做 Production Refit，不生成比赛 `result.csv`，不修改 `candidate/`。D02 保持 Alpha Champion；Defensive Champion 保持 `NONE`。

## 10. 下一轮建议

1. 优先研究 E2 的收益集中机制：在不看 sealed 的前提下，用开发周预注册“时间块方差惩罚/去极端周鲁棒目标”，目标直接修复 without-best-2，而不是继续加指标。
2. E3 应先降低 Recent residual 容量或对 gate/residual 施加零均值与幅度约束；当前两种半衰期都表现出 seed 方差，不应扩展搜索。
3. 新一轮必须重新预注册、继续使用独立开发快照；Recent Holdout 仍保持一次性 sealed，不能依据本轮未读取区间调参。
4. 若未来候选通过 Quick，再做 4×5 Serious、精确重复、P1-P3 风险参数 inner-only 选择和一次性 holdout；此前不得生成 E01。

## 11. 关键证据文件

- `E_R1_PREREGISTRATION.md/json`、`manifests/`、`E_experiment_registry.json`
- `automation_runs/E_R1_CONTROLLER_STATE.json`、`QUICK_SUMMARY_*.json`、`QUICK_PAIRED_WEEKS_*.csv`
- `automation_runs/E_R1_RUN_COMPLETENESS_AUDIT.json`、`E_R1_CODE_SNAPSHOT_CONSISTENCY_AUDIT.json`、`E_R1_PROTECTED_SCOPE_AUDIT.json`
- `automation_runs/E_R1_TIMING_AUDIT.json`、`E_R1_PREREG_SEALED_TERMINAL_AUDIT.json`、`E_R1_FINAL_PROCESS_AUDIT.json`
- `automation_runs/E_R1_VALIDATION_T5_REFERENCE.json`、`E_R1_VALIDATION_T5_DAILY_SEED_AVERAGED.csv`
