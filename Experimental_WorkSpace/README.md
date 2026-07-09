# Experimental_WorkSpace 目录说明

更新时间：2026-07-08

`Experimental_WorkSpace/` 是模型实验工作区。这里用于保存保留的 C 系列基准模型、后续新底层训练程序实验、共享训练环境和实验运行产物。

该目录不是提交区。准备提交时，应另行整理到 `candidate/` 下。

## 目录职责

```text
Experimental_WorkSpace/
  candidates_C/           # C 系列：同一底层训练程序 lineage 的基准模型
  candidates_D/           # D 系列：Market Intelligence 新底层训练程序 lineage
  shared_envs/            # 共享训练运行环境
  PYTORCH_ENVIRONMENT.md  # PyTorch/GPU 环境说明
```

目前实际存在的是 `candidates_C/`、`candidates_D/` 和 `shared_envs/`。`candidates_D/` 当前包含 D001 实验包；D001 round1 已训练评分，R2 正使用三集合 T+5 协议重新训练，尚未晋升为提交候选。

## candidates_C 子目录

`candidates_C/` 代表 C 系列候选模型集合。该集合中的模型来自同一底层训练程序路线，主要差异来自训练窗口、推理重排、集成或 clean 约束。

该目录下每个候选可以保留自己的完整可复现代码副本，例如：

```text
candidates_C/C0_BaseModel/code/src/train.py
candidates_C/C1_Champion_RegimeGate/code/src/train.py
candidates_C/C2_StrictClean_Baseline/code/src/train.py
```

当前 C 系列的 `train.py`、`model.py` 等底层训练程序文件是同一 lineage。`C1` 的 `predict.py`、`config.py` 这类推理期差异属于组合/重排逻辑变化，不单独构成新的底层训练程序。

当前只保留三个 C 系列基准：

- `candidates_C/C0_BaseModel/`
  原 C4。基础 Transformer 模型，作为最低保留基准。
- `candidates_C/C1_Champion_RegimeGate/`
  原 C77。当前 C 系列冠军基准，带 regime gate 的推理组合逻辑。
- `candidates_C/C2_StrictClean_Baseline/`
  原 C302。严格样本外 clean baseline，用于衡量泄露控制后的泛化表现。

## candidates_D 子目录

`candidates_D/` 代表新的 D 系列候选族，用于 Market Intelligence Ranking System 路线。该路线显式建模市场状态、因子暴露、个股时序行为和组合约束，不应和 C 系列 Transformer lineage 混放。

当前内容：

```text
candidates_D/D001_MarketEncoder_Baseline/
candidates_D/D_experiment_registry.json
```

`D001_MarketEncoder_Baseline/` 是当前 D 系列实验包，包含代码、配置和实验输出。后续训练前应先确认数据路径、数据来源、日期窗口、md5、验证折和 registry 条目。

## 新底层训练程序的放置方式

如果后续更换底层训练程序，不应继续混放在 `candidates_C/` 中，而应建立新的同级候选族目录，例如：

```text
Experimental_WorkSpace/candidates_D/
  D001_xxx/
  D002_xxx/
```

命名原则：

- `candidates_C/`：保留 C 系列，同一底层训练程序。
- `candidates_D/`：用于新的底层训练程序路线。
- 如果之后再出现第三条底层训练程序路线，可继续使用 `candidates_E/`。

需要新建候选族目录的情况：

- 修改 `train.py` 的训练流程、样本构造、验证选择逻辑或 checkpoint 保存标准。
- 修改 `model.py` 的模型结构。
- 修改 loss、label 构造、特征构造、dataset 构造、optimizer/scheduler 语义。
- 引入新的底层训练策略，且不只是推理期 rerank 或组合权重变化。

不需要新建候选族目录的情况：

- 只修改 `predict.py` 的推理重排、权重分配、regime gate。
- 只调整 `rerank_config.json` 或 `model/60_158+39/config.json` 中推理相关参数。
- 只做同一底层训练程序下的训练窗口、随机种子或 clean cutoff 对比。

新建候选族目录时，必须同步新增该目录的 `README.md`，说明：

- 该候选族的底层训练程序相对上一族改了什么。
- 训练数据和评估数据边界。
- 允许放入该候选族的模型类型。
- 不允许混入的旧候选或污染链。

## shared_envs 子目录

- `shared_envs/pytorch-cu128-py311/`
  GPU/PyTorch 训练环境。长时间训练和模型迭代优先复用这里的环境。

环境细节见：

```text
PYTORCH_ENVIRONMENT.md
```

## 使用边界

- 训练、调参、rerank 搜索、模型对比都应在这里进行。
- 不要直接修改 `candidate/` 提交包来做实验。
- 新模型必须记录训练程序、训练数据窗口、评估窗口、数据泄露审计和关键指标。
- 更换底层训练程序时，必须新建独立候选族目录，不要和 C 系列混放。
- 改变目录语义或新增候选族时，必须同步更新 `AGENTS.md`、`WORKSPACE_LAYOUT_20260705.md` 和相关 `README.md`。
- 若模型需要晋升为提交候选，先完成评估报告，再由用户明确要求后整理到 `candidate/`。

## D-Series Automation Runtime

D-series automated iteration is launched from `Experimental_WorkSpace/candidates_D/run_d_auto_iteration.py` and keeps generated experiment artifacts in `Experimental_WorkSpace/candidates_D/automation_runs/`.

Derived D-series data inputs with provenance manifests live in `Experimental_WorkSpace/candidates_D/data_snapshots/`. Canonical market-data source files remain under `BaoStock_Datasets/`; do not write model weights or submission packages to the data snapshot directory.

