# candidates_G

`candidates_G/` 是加入点时财报特征后的独立研究谱系。所有输出均标记为
`TEST`，不属于可提交候选包。

## 研究变量

- `TEST_G_CONTROL`：行业 + 中金所股指期货，不使用财报；
- `TEST_G_FINANCIAL`：在完全相同的模型、损失、优化器、折和随机种子上增加
  点时财报特征。

训练入口为 `run_g_financial_fusion.py`。数据留在
`BaoStock_Datasets/data/fusion_research_20260726/`，训练产物留在本目录的
`automation_runs/`，目录和清单都包含 `TEST` 标记。

2026-07-26 四折 Quick 已完成并停止：只有 2/4 折为正，且删除最佳两周后的
累计超额为负。模型输出路径和 MD5 见 `TEST_STATUS.json`，完整结论见
`TEST_MODEL_CARD.md` 与 `G_experiment_registry.json`。

## 边界

- 行业映射是 2026-07-26 静态快照，不是历史点时分类；
- 新数据尚未与 2026-07-18 赛务报备邮件逐项核对；
- 不访问 2025Q4 与 2026-04-03..2026-07-03 封存区间；
- 未通过四折快速门时停止，不创建 `G01`，不进入 `candidate/`。
