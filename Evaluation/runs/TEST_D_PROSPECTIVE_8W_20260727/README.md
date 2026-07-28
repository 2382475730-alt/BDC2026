# TEST_D_PROSPECTIVE_8W_20260727

这里仅保存不含窗口收益的公开状态、密封评分回执和完整性审计。

- 训练、checkpoint 与预测锁位于
  `Experimental_WorkSpace/candidates_D/automation_runs/TEST_D_PROSPECTIVE_8W_20260727/`。
- 每周收益在达到预注册解封条件前只保存在工作区外的 sealed vault。
- 测试期个股与同期上证指数行情只在对应卖出日 09:31 后获取，查询边界
  固定为该窗口买入日至卖出日，并存入 sealed vault 的 `outcomes/Pxx/`。
- 决策日训练前缀与卖出日 outcome 由两个独立程序生成，后者禁止写回
  `BaoStock_Datasets` 的因果前缀、训练沙箱或 candidate 包。
- 当前两周回顾性结果不属于本检验的独立窗口。
- 本检验不会修改或晋级任何 `candidate/` 包。
