# E R1 只读分析智能体意见汇总

本文件由主智能体根据四个只读分析智能体的汇报整理。四个智能体均未写
文件、未启动训练、未占用 GPU。

## 算法架构

- 机制假设：长期 Stable Alpha 与市场门控的 Recent residual 可以降低
  老化；Recent head 零初始化、gate 初始约 0.2 可限制早期扰动。
- 最小实验：D02 exact、E1 因果去重、E2 Market 60 日时序、E3 的
  252/126 半衰期。
- 失败条件：P0 排序未改善、score contract 不精确、Recent 在近期折失效。
- 风险：双 encoder 过容量、平滑 Top-K 与 hard inference 不一致、Top-5
  通过牺牲 Alpha 换风险。

## 特征与泄露

- 机制假设：去除 8 个 Factor aliases 与一个重复价格特征，分离横截面
  percentile 和个股 robust Z，可减少重复加权与尺度漂移。
- 最小实验：全局交易日标签 join、finite-only scaler、全张量 truncation
  和 future-mutation。
- 失败条件：快照哈希不符、标签错位、scaler 越界、任一输入截断差异
  >1e-10、sealed 早读。
- 风险：固定 2026 目标股票池的历史生存者偏差、历史每日覆盖 272-300、
  新股 60/120 日历史不足、D02 Beta 使用不同频 market_return_5。

## 统计评估

- 机制假设：有效 Alpha 应在 47 个唯一市场周广泛改善，而非把 seed-week
  当独立样本或依赖两个极端周。
- 最小实验：fold/seed/week 三方对齐、Quick 4x3、Serious 4x5、唯一
  finalist exact repeat。
- 失败条件：缺周/重复键、F4 非正、PDR/trim/risk 任一门禁失败。
- 风险：CVaR 开发仅 3 周尾部、Recent 仅 1 周尾部；official baseline
  只有约 16.87% finite scores，冻结结果依赖原始 argsort NaN 语义。

## 比赛合规与风险

- 机制假设：所有特征只读决策日及以前数据，E0-E3 固定 P0，E4 仅作用
  于已通过排序门禁的最佳模型。
- 最小实验：离线/固定 seed 双跑、训练预测计时、CSV 约束、不可变目录
  前后指纹、sealed deny/resume 测试。
- 失败条件：不可复现、训练 >=8h、预测 >=5m、联网依赖、输出非法、
  candidate/D02-D04 发生任何变化。
- 风险：最终 Docker 需满足官方 10GB 非压缩镜像限制、指定 4060 8GB/
  16GB 环境、固定随机种子完整复训、离线运行和开源数据/模型报备规则。

## 主智能体裁决

采用低容量全市场标量 gate、hard-forward/soft-backward Top-2、47 唯一周
统计与实际买卖金额成本。E1 拆成三个累计配置，E2 一配置，E3 两个半衰
期。所有门禁、选择顺序和停止条件在结果读取前写入预注册 JSON。

