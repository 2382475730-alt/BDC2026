# C1 提交代码说明

## 环境配置

本项目按官方示例组织为一个可离线复现的提交目录。根目录包含 `Dockerfile`、`pyproject.toml`、`uv.lock`、`train.sh`、`test.sh`、`init.sh`，代码入口位于 `code/src/`。

主要依赖版本由 `uv.lock` 固定，核心依赖包括 Python 3.10-3.12、PyTorch、pandas、numpy、scikit-learn、TA-Lib、joblib、tqdm、tensorboardX。Docker 镜像构建后，复现训练和预测过程不需要联网。

## 数据

提交包内置 `data/train.csv` 和 `data/test.csv`。

`data/train.csv` 是本模型训练和最终预测使用的历史行情数据，覆盖约 2021-01-04 至 2026-03-19 的沪深 300 股票历史日线数据。字段包括股票代码、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌额、换手率、涨跌幅等。

`data/test.csv` 保留为官方目录规范要求的测试集文件，用于本地自测或复现人员检查目录完整性。正式推理脚本根据 `data/train.csv` 的最新交易日生成下一评估窗口的股票组合。

## 预训练模型

未使用外部预训练模型。`model/60_158+39/` 中保存的是由本提交包训练数据生成的模型产物，包括：

- `best_model.pth`
- `second_best_model.pth`
- `scaler.pkl`
- `config.json`
- `rerank_config.json`
- `final_score.txt`

这些文件可以通过运行 `bash train.sh` 从 `data/train.csv` 重新生成。

## 算法

### 整体思路介绍

C1 是一个 5 日收益导向的 A 股截面排序模型。模型先对每只股票构造过去 60 个交易日的特征序列，再用 Transformer 对单股时序模式和股票间截面关系进行建模，输出每只股票的排序分数。推理阶段在模型分数基础上进行风险和成交活跃度重排序，最终输出不超过 5 只股票及权重。

### 方法的创新点

1. 使用 158 个 alpha 风格量价特征加 39 个技术指标特征，覆盖动量、反转、波动、成交量、价格形态等信息。
2. 使用 Transformer 编码 60 日历史序列，并通过 CrossStockAttention 建模同一交易日股票之间的相对关系。
3. 损失函数同时使用 listwise 排序目标、pairwise 边界约束、top-5 样本加权和分数正则，使模型更关注组合前排股票。
4. C1 在 C0 Transformer 权重基础上加入成交量 regime gate：推理时计算最新截面 `_gate_volume_ratio_5 = volume / volume.shift(5)` 均值。当该均值不高于 1.111401 时，使用候选排序第 1、2、12、15、17 名，权重为 0.05、0.15、0.50、0.15、0.15；否则退回 C0 的 floor-capped softmax 权重，参数为 temperature=0.03、min_weight=0.05、max_weight=0.50。
5. `_gate_volume_ratio_5` 只作为 regime 判断变量，不参与模型训练标签，也不使用未来评估窗口收益，避免未来信息泄漏。

### 网络结构

核心模型定义在 `code/src/model.py`：

- 股票 ID embedding
- 线性特征投影
- PositionalEncoding
- TransformerEncoder
- FeatureAttention
- CrossStockAttention
- ranking layers 和 score head

输入张量形状为 `[batch, num_stocks, sequence_length, feature_dim]`，输出为 `[batch, num_stocks]` 的股票排序分数。

### 损失函数

训练脚本 `code/src/train.py` 使用 `WeightedRankingLoss`。主要组成包括：

- listwise softmax 排序损失
- top/bottom 边界 pairwise loss
- top-5 样本加权
- global cross entropy 辅助项
- score L2 正则
- 梯度裁剪和 early stopping

### 数据扩增

未使用外部数据扩增。训练脚本支持近期样本重复/加权配置，但当前 C1 提交配置保持与 C0 训练窗口一致，重点改动在推理期 regime gate 和组合权重逻辑。

### 模型集成

推理默认使用 `best_model.pth` 与 `second_best_model.pth` 的均值集成，并进行截面标准化。集成开关由 `model/60_158+39/rerank_config.json` 和 `code/src/config.py` 控制。

### 算法的其他细节

预测阶段先取模型排序前 150 只股票作为候选池，再叠加 `return_5`、`volume_change`、`volatility_10`、`atr_14`、`open_close_spread`、`volume_ratio` 等重排序因子。最终选择逻辑由 `weight_mode = regime_volume_gate_c69_else_c4` 控制。

## 训练流程

运行：

```bash
bash train.sh
```

训练流程如下：

1. 固定随机种子为 42。
2. 读取 `data/train.csv`。
3. 按股票分组构造 158+39 特征。
4. 构造未来 5 日开盘收益标签。
5. 按日期切分训练集和验证集。
6. 标准化特征并保存 `scaler.pkl`。
7. 构建按交易日组织的 ranking dataset。
8. 训练 Transformer 排序模型。
9. 根据稳健验证分数保存 `best_model.pth` 和 `second_best_model.pth`。
10. 保存 `config.json`、`final_score.txt` 和 `rerank_config.json`。

训练时间目标为官方要求的 8 小时以内。

## 推理流程

运行：

```bash
bash test.sh
```

推理流程如下：

1. 读取 `data/train.csv` 的最新交易日作为预测基准日。
2. 执行与训练一致的特征工程。
3. 加载 `scaler.pkl`、`best_model.pth` 和可选的 `second_best_model.pth`。
4. 对所有满足 60 日历史窗口的股票打分。
5. 根据 C1 的重排序和成交量 regime gate 逻辑选择最多 5 只股票。
6. 将结果写入 `output/result.csv`，列名固定为 `stock_id,weight`。

## 其他注意事项

1. 复现训练和预测过程不需要联网。
2. `output/result.csv` 的权重和不超过 1，股票数量不超过 5。
3. `data/train.csv`、`data/test.csv`、`model/`、`code/` 均已包含在提交包中，不依赖提交目录外的任何文件。
4. 本地可运行 `python code/src/validate_submission.py --write-manifest` 检查目录结构、入口脚本和输出格式。
