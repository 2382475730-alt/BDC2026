# E 系列 R1 预注册

状态：`LOCKED_BEFORE_IMPLEMENTATION_AND_RETURN_ACCESS`

创建日期：2026-07-15  
研究墙钟上限：24 小时  
首版名称：`E001_RecentRegimeMixture_Ranker`

本文件在 E 系列代码复制、修改、训练以及任何开发收益读取之前冻结。
看到结果后不得放宽门禁、增加配置或改变选择顺序。机器可执行口径见
`E_R1_PREREGISTRATION.json`；其 SHA256 单独保存在
`E_R1_PREREGISTRATION.sha256`。

## 1. 核心机制与可证伪性

E 系列主假设不是增加一组技术指标，而是把排序信号拆成：

```text
ranking_scores = stable_score + gate(market_60d) * recent_residual
```

- Stable Expert 读取 60-120 日个股因果序列，建模长期稳定 Alpha。
- Recent Expert 读取最近 20 日，只学习低容量残差；训练半衰期只允许
  126 或 252 个交易日。
- Market Regime Encoder 读取最近 60 日市场宽度、离散度、换手和波动
  的水平与变化量，输出全市场标量 gate。
- 门控后的专家嵌入经过一层跨股票注意力；最终输出仍必须精确满足上式。
- 主排序目标采用 hard Top-2 forward / soft backward 的 straight-through
  Top-K aligned loss；训练与推理只使用 `ranking_scores`。

失败定义：若排序器在固定 P0 下未通过 Quick，不允许通过现金、Top-5
分散或风险优化掩盖排序失败。

## 2. 数据、标签与因果边界

- 唯一来源：`c0_train_plus_baostock_future_20260703.csv`，锁定
  MD5=`78759d892c95eb8ba547c2df9a8fca5a`。
- 原始历史：2021-01-04..2026-07-03；监督样本从 2022-01-04 开始。
- 标签通过全局交易日表显式连接，固定为
  `open(T+5)/open(T+1)-1`。T+1 或 T+5 无目标股票开盘价时丢弃样本，
  不得用“该股票下一次可交易的第 N 行”替代。
- Development 使用物理截断视图，数值截止 2026-03-27。
- 同日横截面特征只使用该决策日实际存在的锁定 300 股票池行；历史成分
  变化不可得，因此报告必须披露固定目标池的生存者偏差风险。
- scaler 仅在 inner/outer train 样本的有限值上拟合；先标准化、后填 0，
  并保存拟合日期、样本数和统计量哈希。

## 3. R5 固定验证协议

四个 development folds 共 47 个唯一周：

| Fold | train_end | inner validation | buy/sell evaluation | weeks |
| --- | --- | --- | --- | ---: |
| F1 | 2024-12-25 | 2024-11-28..12-25 | 2025-01-06..03-28 | 12 |
| F2 | 2025-03-26 | 2025-02-27..03-26 | 2025-04-07..06-27 | 12 |
| F3 | 2025-06-26 | 2025-05-29..06-26 | 2025-07-07..09-26 | 12 |
| F4 | 2025-12-23 | 2025-08-27..09-23 | 2026-01-05..03-27 | 11 |

训练样本必须满足 `label_end(T) < first_decision_date`。R5 给定的
`train_end` 已形成 T+5 的五交易日标签隔离；为保持 R5 精确边界，不再
额外重复扣除第二个五交易日窗口。manifest 分开记录 label purge 与
additional embargo，避免把二者混称。

Legacy 训练 2022-01-04..2024-12-25、验证
2025-01-03..2026-01-05，只用于官方 baseline 和 D02 固定协议复现，
不用于 E1-E4 反复调参。

Recent Sealed Holdout 的买卖窗口为 2026-04-07..2026-07-03，共 13 周。
没有 finalist lock 时，数值收益、组合指标和派生统计均禁止访问。

## 4. 预注册配置与顺序

每个阶段不超过三个配置，按以下顺序执行：

1. `E0_D02_EXACT`：复现 D02 Legacy 指标并复用/核验 R5 L0/F0 四折三
   seed 控制；不作为 challenger。
2. E1（累计式三配置）：
   - `E1A_DEDUP_ST60`：删除 8 个 Factor aliases 与重复
     `open_close_spread`，保持 60 日 stock encoder；
   - `E1B_CAUSAL_SCALE_ST60`：在 E1A 上改为 finite-only train scaler，
     加历史有效性掩码；
   - `E1C_FULL_CAUSAL_ST60`：在 E1B 上加入冻结定义的新特征，并把
     Beta/相关性修正为同频日收益。
3. `E2_MARKET_GRU60`：只在 E1C 上加入 60 日 Market GRU。
4. E3（两个配置）：
   - `E3_MIX_HL252`；
   - `E3_MIX_HL126`。
   两者除 Recent auxiliary loss 半衰期外完全相同：Stable 120 日、
   Recent 20 日、Market 60 日、全市场标量 gate、单层 cross-stock MHA。
5. E4：只有通过 Serious 的最佳 P0 排序器可以进入。冻结排名分数后比较
   P0 控制与 P1/P2/P3；Recent 只允许评估最终锁定的一种 ranker+policy。

每个配置先做单 seed smoke 和自动化测试，再做四折 x 3 seeds Quick。
最多两个 Quick 通过者进入四折 x 5 seeds Serious。若 E1、E2、E3 三个
假设族连续均无 Quick 通过者，立即停止。

## 5. 特征冻结定义

保留因果 OHLCV、1/5/10/20/60 日动量、日内收益、高低价幅度、成交量/
成交额比率、换手、5/20 日波动、下行波动、20/60 日回撤、ATR proxy、
同频 20 日 Beta 和相关性。

新增定义固定为：

- 横截面百分位：`rank(method="average", pct=True)`，NaN 不参与；
- 稳健 Z：`(x - rolling_median)/(1.4826*rolling_MAD+eps)`，60/120 日；
- 动量斜率：最近 5/20/60 个 `log(close)` 对 `[0..w-1]` 的带截距 OLS slope；
- 回撤修复：`drawdown_60(T)-drawdown_60(T-5)`；
- 波动比：`log((vol5+eps)/(vol20+eps))`；
- 量价确认：`return_5*log(mean_volume_5/(mean_volume_20+eps))`；
- 市场序列：breadth、日收益离散度、换手均值、日收益波动，以及各自
  固定五日变化量。

所有逐股特征、横截面特征、Stable/Recent 张量、Market 序列、mask 和
股票顺序都必须通过 full-vs-truncated 与 future-mutation 审计，容差
`1e-10`。

## 6. 收益、成本与统计聚合

官方毛收益和含成本净收益分开保存。每周强制在首个交易日开盘买入、
末个交易日开盘卖出，因此：

```text
buy_notional  = sum(weight_i)
gross         = sum(weight_i * stock_return_i)
sell_notional = sum(weight_i * (1 + stock_return_i))
cost          = 0.0005 * buy_notional + 0.0010 * sell_notional
net           = gross - cost
```

跨周组合重合不降低费用，因为周末已清仓；另报 composition one-way
turnover=`0.5*sum(abs(w_t-w_{t-1}))`。最大回撤从初始 NAV=1 开始。
CVaR95 取最差 `ceil(0.05*n)` 周，开发集 tail_count=3、Recent=1，报告
必须披露小样本不确定性。

三/五 seed 先在同一 `(fold, week)` 上取收益算术均值，形成 47 个唯一
市场周，再计算 pooled excess、PDR、去最佳两周、MaxDD 与 CVaR。seed
不作为独立市场样本。PDR 使用严格 `excess > 1e-12`，平局计非胜。

## 7. 不可放宽门禁

Quick：先超过同协议 official baseline；47 周平均净超额 >0；PDR>=0.50；
至少 3/4 folds 平均超额非负；F4 超额 >0；删除最大的两个日历周超额后
总和仍 >0；MaxDD 和 CVaR95 均不超过 fold-matched D02 的 1.05 倍；
所有泄露、确定性、score、成本、约束和恢复审计通过。

Alpha Champion：5 seeds 至少 4 个相对 D02 的 47 周平均净超额 >0；最差
seed >=-0.005；5-seed median gross official-proxy > D02；唯一周
PDR>=0.52；至少 3/4 folds 严格为正；F4 为正；去最佳两周后为正；
MaxDD、CVaR95 均不高于 D02；精确重复训练的 checkpoint、scaler 和规范化
预测哈希完全一致。

Defensive Champion：仍须超过 official baseline，gross official-proxy
至少保留 D02 的 90%，且净 MaxDD 和 CVaR95 均至少下降 15%。只能称
Defensive Champion，不能称整体或 Alpha Champion；本轮不因此创建 E01。

Recent：锁定唯一 finalist 后只解封一次。13 周平均净超额 >0、严格
PDR>=0.50（至少 7 周）、MaxDD 与 CVaR95 均不高于 D02。任一失败即拒绝，
不得据其结果修改模型、epoch、特征或策略。

## 8. E4 冻结策略

- P0：Top-2 等权满仓，仅排序控制，不受 E 主策略 3-5 只约束。
- P1：Top-4 bounded score-softmax，投资部分单股 <=35%，HHI<=0.34，q=1。
- P2：Top-10 候选中确定性选择 5 只，使用过去 60 日收缩协方差、标准化
  score、相关性、集中度和换手成本的固定目标；投资部分单股 <=35%、
  HHI<=0.34。
- P3：Top-4 bounded score-softmax + 仅用 inner/OOF 拟合的低容量置信度
  敞口头，q clip 到 [0.50,1.00]。

P1-P3 必须持有 3-5 只、非负、无做空和杠杆，现金不写入 CSV。参数只由
inner/development 决定。若 P1-P3 均不能保留 P0 的 Alpha 证据，不锁定
finalist。

## 9. 硬停止与产物

数据/源哈希不符、标签边界错误、scaler 越界、truncation diff>1e-10、
sealed 早读、结果后协议变更、确定性失败、连续三族 Quick 失败、24 小时
预算耗尽或任一程序训练预测超时，均触发硬停止。

每次 run 必须先登记，并保存代码快照、配置、seed、命令、环境、日志、
checkpoint、预测、周收益、指标、leakage audit、恢复命令和 artifact hash；
失败 run 不得删除。只有 `ACCEPT_AS_E_ALPHA_CHAMPION` 才创建 E01、执行
固定 epoch Production Refit 并生成比赛格式 `result.csv`。

