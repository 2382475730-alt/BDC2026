# 已执行模型检验报告：Candidate-D、H5 与同期基准

生成日期：2026-07-28  
报告范围：仅汇总截至生成时已经完成并留存证据的训练、回溯评分、Sentinel 检验和防泄露审计；不报告等待状态，不推测尚未发生的前瞻收益。

## 技术摘要

在 2026-07-06 至 2026-07-17 的两个已暴露周窗口中，Candidate-D 的单边 10 bps 成本后复合净收益为 **+9.1129%**，同期官方参考为 **-6.4334%**、Candidate-C 为 **-5.6872%**、TEST-H5 为 **-12.0785%**。Candidate-D 分别领先官方参考 **15.5463 个百分点**、领先 H5 **21.1915 个百分点**。

同期上证指数的已执行对比结果为 **-3.2029%**，Candidate-D 相对领先 **12.3158 个百分点**。该指数结果的机器可读回执未保存在本次工作区运行目录，因此只作为市场环境参照，不进入可复现硬门槛；可复现的内部等权基准为沪深 300 成分股等权组合，其单边 10 bps 后两周复合收益为 **-2.2837%**。

结论是：**在已经执行的同窗比较中，Candidate-D 是当前最强的保留候选，H5 没有形成替换依据。** 但“较稳定”只能理解为相对其它已测候选的表现更好，不能理解为已经证明绝对稳定。Candidate-D 只有 2 个周样本、正收益周占 50%，收益主要由第一周贡献，且每周仅持有 2 只股票、跨周换手率 100%。这些暴露窗口只允许形成诊断结论，不允许据此宣称未来收益、统计显著性或完成模型晋级。

## Candidate-D 在同窗候选中领先，但稳定性证据仍不足

主指标定义为：每周在 T+1 开盘买入、T+5 开盘卖出，现金收益为 0；在组合毛收益上扣除买入和卖出各 10 bps，并对两个独立周窗口进行复合。候选使用各自原生选股与权重规则。

| 模型或基准 | 两周复合收益 | 口径 | 相对 Candidate-D |
|---|---:|---|---:|
| Candidate-D | **+9.1129%** | 单边 10 bps 后净收益 | — |
| 同期上证指数 | -3.2029% | 已执行市场指数参照；工作区内缺少原始回执 | -12.3158 个百分点 |
| 沪深 300 成分股等权 | -2.2837% | 单边 10 bps 后净收益 | -11.3966 个百分点 |
| Candidate-C | -5.6872% | 单边 10 bps 后净收益 | -14.8001 个百分点 |
| THU-BDC2026-main | -6.4334% | 单边 10 bps 后净收益 | -15.5463 个百分点 |
| TEST-H5 | -12.0785% | 单边 10 bps 后净收益 | -21.1915 个百分点 |

逐周结果揭示了 Candidate-D 的优势和风险是同时存在的：

| 周窗口 | 买入—卖出 | Candidate-D | 官方参考 | Candidate-C | TEST-H5 | 等权基准 |
|---|---|---:|---:|---:|---:|---:|
| W1_20260703 | 2026-07-06—2026-07-10 | **+10.1752%** | -2.4564% | -2.7752% | +5.6793% | -0.9584% |
| W2_20260710 | 2026-07-13—2026-07-17 | **-0.9641%** | -4.0771% | -2.9950% | -16.8035% | -0.9384% |

Candidate-D 的最差周净收益为 -0.9641%，两周最大回撤为 0.9641%，都优于官方参考的 -4.0771% 和 6.4334%。不过 Candidate-D 的周收益波动率为 5.5696%，第一周到第二周下降 11.1393 个百分点；组合 HHI 为 0.5、平均有效持仓为 2、跨周换手率为 100%。因此，当前证据支持“相对领先、可继续保留”，不支持“收益过程已经稳定”。

## H5 同时未通过收益诊断和 Sentinel 晋级门

H5 使用与其它 candidates 完全相同的 W1/W2 冻结训练 CSV 字节，固定 seed=20260726、final epoch=1，从零训练，不使用验证集选模、早停、恢复训练或历史缓存命中。两轮预测在读取隔离验证收益前均完成双推理逐字节锁定。

两周回溯中，H5 第一周净收益为 +5.6793%，第二周为 -16.8035%，最终复合净收益为 -12.0785%。它落后官方参考 5.6451 个百分点，落后 Candidate-D 21.1915 个百分点。H5 的结果并不是被成本假设单独驱动：其两周毛收益同样为 -11.7004%。

独立的 Sentinel 检验也给出拒绝结论：

| Sentinel 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| 决策变化率 | 95.65%（22/23） | ≥20% 且每折有变化 | 通过 |
| 变化决策严格胜率 | 31.82% | >50% | **失败** |
| H5 CVaR20 | -8.8023% | 不劣于 control -8.7177% | **失败** |
| 归一化 rank5/rank6 间隔 | 0.0855 | 不劣于 H1 0.1898 | **失败** |
| Sentinel 总耗时 | 4.17 分钟 | ≤5 分钟 | 通过 |

H5 虽然改变了排序边界，但改变并没有稳定转化为更高收益，尾部风险和边界间隔也同时恶化。工作流按预注册门槛停在 Sentinel，未运行 Full Quick、Serious 或 sealed evaluation，未创建 H01，未修改任何正式 candidate 包。

## 同训练字节与预测先锁定使结果不存在已发现的数据泄露

三候选滚动检验的泄露审计为 `PASS`：

- W1 的三模型训练输入哈希均为 `DC76F0C68BA46C0D629AF37C5F7E280873D732702CB6CA1E375E89715946E07E`，数据截止 2026-07-03。
- W2 的三模型训练输入哈希均为 `7E8CC2F5A6B685F21CAB17BA3A929D6DB8B15FFC2A5461C6CE1C3ED4B36D7F12`，数据截止 2026-07-10。
- 两轮未来行计数均为 0；所有预测均先锁定；W2 训练输入不含第二周未来行。
- 验证文件保留在工作区外，SHA-256 为 `B21FB1BEEFAE407729C305E01E274695FE083C718B695F06A2146D7512B6AFC9`。
- 未根据 W1 结果修改 W2 超参数，未修改 candidate 包；手工单笔和复合收益复算均通过。

H5 两周诊断的 14/14 项泄露与执行审计全部通过：代码与配置冻结、W1/W2 双预测锁、哈希先验后仅解析验证文件一次、训练和推理配置不含验证路径、评分后无模型事件、正式候选和 H01/Serious/Finalist 路径均未被创建或修改。

已完成的 Candidate-D 前瞻控制器实现验证也支持后续无泄露执行能力：合并回归测试 44/44 通过，训练/推理沙箱不能访问 outcome 或 sealed vault，评分数据只能在卖出日门后进入工作区外密封库，预测必须先于买入开盘锁定，源码和数据计划受 SHA-256 强制门保护。由于该部分没有产生任何已完成的模型收益，它只证明执行机制，不进入本报告的收益结论。

## 结论边界与决策

本次已执行证据给出的决策为：

1. **保留 Candidate-D 作为当前 Alpha Champion/提交候选，不做降级。**
2. **关闭 H5 方向，不创建 H01，不将其结果用于候选晋级。**
3. 不因两周结果修改 Candidate-D、官方参考或 D01 研究快照；`candidate/` 保持不变。
4. 不把 +9.1129% 外推为未来收益。两周窗口已经暴露，且样本量、持仓集中度和跨周换手都不足以支持正式稳定性结论。
5. 如果后续继续研究，应优先检验 Candidate-D 的集中度与换手风险；任何结构、损失、标签、特征、数据、优化器或 checkpoint 选择变化都应建立新 lineage，并使用未消费的独立窗口预注册验证。

## 仍需回答但本报告不推测的问题

- Candidate-D 的优势能否在至少 8 个未消费独立窗口中维持？
- 第一周收益是否主要由个别股票或市场风格暴露驱动？
- 在不降低同期官方参考优势的前提下，能否降低 HHI、跨周换手和最差周损失？
- 同期上证指数的原始行情回执与计算脚本应否补录到 `Evaluation/runs/`，使市场基准也具备与模型结果相同的可复现等级？

## 证据索引

- `Evaluation/runs/tri_candidate_recent_rolling_20260722/comparison_summary.json`
- `Evaluation/runs/tri_candidate_recent_rolling_20260722/fold_scores.csv`
- `Evaluation/runs/tri_candidate_recent_rolling_20260722/leakage_audit.json`
- `Evaluation/runs/tri_candidate_recent_rolling_20260722/manual_score_check.json`
- `Evaluation/runs/tri_candidate_recent_rolling_20260722/verification.json`
- `Evaluation/runs/TEST_H5_predicted_boundary_20260727/TEST_H5_FINAL_REPORT.json`
- `Evaluation/runs/TEST_H5_predicted_boundary_20260727/TEST_H5_SENTINEL_GATE.json`
- `Evaluation/runs/TEST_H5_two_week_retrospective_20260727/TEST_H5_TWO_WEEK_SCORE.json`
- `Evaluation/runs/TEST_H5_two_week_retrospective_20260727/TEST_H5_LEAKAGE_AUDIT.json`
- `Evaluation/runs/TEST_H5_two_week_retrospective_20260727/TEST_H5_MANUAL_SCORE_CHECK.json`
- `Evaluation/runs/TEST_D_PROSPECTIVE_8W_20260727/IMPLEMENTATION_VERIFICATION.md`
- `Evaluation/runs/TEST_D_PROSPECTIVE_8W_20260727/TRAINING_SANDBOX_ACCESS_AUDIT_20260728.md`
- `Evaluation/runs/TEST_D_PROSPECTIVE_8W_20260727/integrity_audit.json`

