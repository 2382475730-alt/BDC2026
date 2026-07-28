# Candidate-I 代码说明

状态：`SUBMISSION_FORMAT_READY / RESEARCH_REJECTED`。

本目录是用户明确要求打包的独立提交格式 TEST，不替换
`candidate/Candidate-D/`。底层模型沿用 Candidate-D 的固定 12-epoch
生产重训权重；推理阶段增加严格 lag-1 的残差动量、量价确认和反转风险
重排，最终固定输出 top-5 等权组合。

原研究运行 `TEST_D_ALPHA_RESIDUAL_RERANK_20260728` 在读取任何校准或外层
outcome 前因 Candidate-D top-10 中存在不完整成交量确认因子而终止，状态为
`REJECTED_AT_DATA_GATE`。本包仅表示结构可提交和入口可复现，不表示该方法已
通过研究门禁、优于 Candidate-D 或优于官方 baseline。

## 环境配置

- Python 3.11-3.12
- PyTorch 2.10.0，CUDA 12.8
- NumPy 2.3.3
- pandas 2.3.2
- Docker 镜像名称：`bdc2026`

训练和推理入口均不包含网络访问。官方复现时只使用包内代码、数据、模型、
scaler 和固定配置。

## 数据

`data/train.csv` 是公开 BaoStock 行情的冻结快照，范围为
2021-01-04 至 2026-07-03，SHA-256：
`8e9cf37e5c53e4edafddc3f6f6b5843d4037376334e6f53cba068699095df1ad`。

`data/test.csv` 是包内推理输入切片。推理会按股票代码和日期与历史数据去重。
本包不使用行业、期货、财报、公告或任何外部辅助数据。

## 预训练模型

未使用外部预训练模型、词典或 embedding。`model/final_model.pth` 是
Candidate-D 生产程序以固定种子 `20260708`、固定 12 epoch 从包内
`data/train.csv` 训练得到的权重。残差重排本身不训练新权重。

## 算法

### 整体思路介绍

底层 Transformer 为 300 股生成完整横截面排名分数。对决策日 T，额外因子
只读取 T 之前的数据，最近可用日通常为 T-1。最终分数为：

```text
AlphaScore =
    D_rank_score
    + 0.05 * residual_momentum_rank
    + 0.05 * volume_confirmation_rank
    - 0.05 * reversal_risk_rank
```

上述 `(0.05, 0.05, 0.05)` 是预注册网格中的“各因子同时启用的最小对称非零”
打包配置，不由 outcome、测试周收益或 leaderboard 选择。

### 方法的创新点

- 使用 60 日市场残差回归构造 5/20 日及排除最近 5 日的中期动量；
- 结合 20 日突破、成交量和成交额确认；
- 对跳空回落、长上影和三日短期反转施加风险扣分；
- 因子严格 lag-1，top-k 和权重在推理前固定。

### 网络结构

- 个股 60 日序列 Transformer 编码器；
- 市场状态编码器；
- 横截面排名头；
- 仅在推理端执行的确定性残差重排器。

### 损失函数

底层模型沿用 Candidate-D 的 soft ranking、pairwise margin、组合收益奖励和
组合波动惩罚联合损失。残差重排不增加训练损失或可训练参数。

### 数据扩增

不使用随机数据扩增。训练标签是 T+1 开盘买入至 T+5 开盘卖出的收益。

### 模型集成

不使用模型集成。最终结果来自一个固定种子模型和一个固定确定性重排规则。

### 算法的其他细节

各原始因子先按横截面 2.5%/97.5% winsorize，再映射到 `[-1,1]` 并截断到
`[-0.95,0.95]`。成交量原始项缺失时沿用冻结 TEST 代码的横截面中性秩处理。
这是用户授权打包失败 TEST 的工程例外，也是该包不能声称研究晋级的原因。

最终固定选择 5 只股票，权重均为 0.20，总权重为 1，HHI 为 0.20；不会按
单周状态动态改变 top-k、lambda 或权重。

## 训练流程

```bash
bash init.sh
bash train.sh
```

`train.sh` 固定 `PYTHONHASHSEED=20260708`，运行
`code/src/train.py`，从 `data/train.csv` 训练 12 epoch，并写入：

- `model/final_model.pth`
- `model/scalers.json`
- `temp/training_history.csv`
- `temp/training_manifest.json`

训练过程不读取 `model/rerank_config.json`，因此底层模型可与 Candidate-D
生产重训逐张量比对。

## 推理流程

```bash
bash test.sh
```

`code/src/test.py`：

1. 合并并去重 `data/train.csv` 与 `data/test.csv`；
2. 加载固定 checkpoint 和 scaler，生成完整股票分数；
3. 只用决策日前行情生成 lag-1 残差因子；
4. 使用 `model/rerank_config.json` 的固定 lambda 重排；
5. 将 5 只等权股票以 UTF-8 写入 `output/result.csv`。

输出列严格为 `stock_id,weight`。

## 其他注意事项

- 研究状态为 `REJECTED_AT_DATA_GATE`，没有校准结果或 Quick 晋级证据；
- 包内当前 `data/test.csv` 的运行仅是已暴露的一次性诊断；
- 结构验证通过不等于收益超过官方 baseline；
- 不得将本说明解读为未来收益、榜单收益或获奖资格保证。

