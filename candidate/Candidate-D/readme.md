# Candidate-D 代码说明

状态：`SUBMISSION_CANDIDATE_READY`。固定种子全量复训、结果一致性、运行时间和提交结构校验均已完成。

## 环境配置

- Python 3.11-3.12
- PyTorch 2.10.0，CUDA 12.8
- NumPy 2.3.3
- pandas 2.3.2
- Docker 镜像名应为 `bdc2026`

训练和推理入口不包含网络下载行为；复现时仅使用包内数据、代码、配置、缩放器和模型。

## 数据

`data/train.csv` 是免费公开 BaoStock 行情整理出的冻结快照，范围为 2021-01-04 至 2026-07-03，SHA-256 为
`8e9cf37e5c53e4edafddc3f6f6b5843d4037376334e6f53cba068699095df1ad`。

`data/test.csv` 是训练快照最后五个交易日的包内推理输入切片。推理时会将其与历史数据按“股票代码、日期”去重合并，因此当前冻结结果不会重复计入行情；复现人员替换测试数据后，仍沿用同一入口。

## 预训练模型

未使用外部预训练模型、词典或 embedding。`model/final_model.pth` 是本包训练代码以固定种子 `20260708`、固定 12 个 epoch 从 `data/train.csv` 训练得到的生产重训权重。

## 算法

### 整体思路介绍

模型将个股 60 日序列、截面因子和市场状态分别编码，再输出横截面排序分数。最终选择分数最高的 2 只股票并等权配置。

### 方法的创新点

相对官方参考代码，本模型显式加入市场状态和横截面因子表征，并以排序、成对间隔、组合收益和组合波动组成的联合损失训练。

### 网络结构

- 个股时间序列 Transformer 编码器
- 市场状态编码器
- 可选因子分支（本冻结配置关闭）
- 横截面排序头

### 损失函数

总损失由 soft ranking、pairwise margin、组合收益奖励和组合波动惩罚构成。参数固定在 `model/config.json`。

### 数据扩增

不使用随机数据扩增。训练样本按交易日构造，标签为 T+1 开盘买入至 T+5 开盘卖出的收益。

### 模型集成

不使用模型集成；提交结果来自单个固定种子模型。

### 算法的其他细节

缩放器只在生产训练样本上拟合；生产重训不使用验证集，不进行未来信息选模。

## 训练流程

执行：

```bash
bash init.sh
bash train.sh
```

`train.sh` 固定 `PYTHONHASHSEED=20260708`，从 `data/train.csv` 构造 1083 个监督样本，训练 12 个 epoch，并写入：

- `model/final_model.pth`
- `model/scalers.json`
- `temp/training_history.csv`
- `temp/training_manifest.json`

训练不写出逐股票预测矩阵。

## 推理流程

执行：

```bash
bash test.sh
```

脚本读取 `data/train.csv` 与 `data/test.csv`，使用 `model/final_model.pth` 和 `model/scalers.json`，将不超过 5 只股票写到 UTF-8 编码的 `output/result.csv`。默认冻结配置输出 2 只等权股票。

## 其他注意事项

- 当前研究身份：`D01_FrozenResearchChampion_20260709`。
- 历史身份：`D02_FrozenResearchChampion_20260709`，仅用于审计映射。
- 研究包中的失败挑战者、预测矩阵、训练日志和自动化运行目录未进入本候选包。
- 2026-07-20 完成 12-epoch CUDA 全量复训：训练 3.782 分钟；59 组参数与来源模型完全一致，最大绝对差为 0；复现结果与来源 `result.csv` 逐字节一致；推理 23.4 秒。
- 同协议研究验证中，D Top-2 平均组合收益为 0.037120，官方 Top-5 为 0.012528；但更宽的研究稳健性门禁结论仍为 `REJECT`，因此不能把该比较表述成未来收益保证。
- 本包只表示结构与复现候选，不把一次性本地诊断描述为正式榜单收益保证。
