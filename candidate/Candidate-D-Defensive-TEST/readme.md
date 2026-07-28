# Candidate-D Defensive TEST 提交形态包

状态：`FORCED_TEST_PACKAGE_NOT_RESEARCH_GATE_APPROVED`。

本目录是按官方 baseline 目录与入口约定构造的独立、可运行包。它复用
Candidate-D 的冻结模型、scaler 和特征工程，但把原生 top-2 等权组合层替换为
complete-link 防御组合层。它不是 Defensive Champion，不会替换
`candidate/Candidate-D/`，也不代表 V2 已通过研究门禁。

## 算法

### 强制政策

- Policy ID：`EQ_V0.050_C0.150_T0.000_R0.100`
- 五股等权：`20/20/20/20/20`
- 波动率惩罚：`0.05`
- 相关簇惩罚：`0.15`
- 换手惩罚：`0.00`（单次提交没有前一期锁定持仓）
- 反转风险惩罚：`0.10`
- 市场状态乘数：固定 `1.00`（单次提交的保守中性回退）
- 相关性：决策日前 60 个交易日，至少 45 个共同观测，25% 向截面中位数收缩
- 缺失相关性：`0.65`
- 聚类：complete linkage，距离 `sqrt((1-rho)/2)`，切点 `rho=0.65`
- 扩池：`top-20 → top-50 → top-100 → 完整排名`
- 约束：五只不同股票、满仓、最大单股 30%、HHI 不超过 0.30、每簇最多两只

V2 的 288 项网格没有政策同时通过两个 seed 的预注册非劣门槛，因此原流程在
组合门禁拒绝，没有合法的全局政策锁。应用户明确要求，本包以原冻结排序规则对
所有结构合格政策进行同序决胜，强制选取上述政策。其 47 个唯一内层周的诊断为：
正收益率 72.34%、中位数 1.994%、CVaR20 -5.996%、最差周 -20.352%、平均超额
-0.301%；seed 20260726 平均超额 +0.587%，seed 20260727 为 -1.188%。

## 环境配置

- Python 3.11–3.12
- PyTorch 2.10.0 / CUDA 12.8
- NumPy 2.3.3
- pandas 2.3.2
- SciPy 1.16.2

推理不访问网络，仅使用包内数据、模型、scaler 和配置。

## 数据

`data/train.csv` 是 Candidate-D 的冻结 BaoStock 历史快照，
`data/test.csv` 是包内最后五个交易日的推理输入切片。推理入口按股票代码和日期
去重合并两者。防御特征严格使用决策日之前的数据。

## 预训练模型

不使用外部预训练模型。`model/final_model.pth` 是 Candidate-D 固定 seed
`20260708`、12 epoch 生产重训的冻结权重，本次没有重新训练。

## 训练流程

Linux / baseline 容器：

```bash
bash init.sh
bash train.sh
```

训练入口保持 Candidate-D 的固定生产重训协议；正常提交复现可直接使用包内冻结模型。

## 推理流程

Linux / baseline 容器：

```bash
bash test.sh
```

Windows 本地复现：

```powershell
python code/src/test.py --work-dir . --output output/result.csv
```

输出严格为：

```text
stock_id,weight
......
```

`output/result.csv` 恰好五行，权重和为 1。结构和选择诊断写入
`temp/defensive_diagnostics.json`，不进入官方提交结果。

## 其他注意事项

### 证据边界

- 来源实验：
  `Experimental_WorkSpace/candidates_D/automation_runs/TEST_D_DEFENSIVE_COMBO_V2_COMPLETE_LINK_20260728/`
- V2 状态：`REJECTED_AT_COMBINATION_GATE`
- 本包用途：强制的提交结构和端到端运行验证
- 禁止解读：正式晋级、未来收益保证、对 Candidate-D 的自动替换
