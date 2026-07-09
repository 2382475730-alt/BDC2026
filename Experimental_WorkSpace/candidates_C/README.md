# candidates_C 目录说明

更新时间：2026-07-07

`candidates_C/` 是 C 系列候选模型族目录。该目录只放来自同一底层训练程序 lineage 的模型。

## 当前定位

C 系列当前保留：

```text
C0_BaseModel/
C1_Champion_RegimeGate/
C2_StrictClean_Baseline/
```

这些候选共享同一底层训练程序路线。当前核验结果显示，三者的核心 `train.py` 和 `model.py` 文件一致。

## 底层训练程序位置

底层训练程序没有单独抽成公共源码目录，而是随每个候选保留完整可复现代码副本：

```text
C0_BaseModel/code/src/train.py
C0_BaseModel/code/src/model.py

C1_Champion_RegimeGate/code/src/train.py
C1_Champion_RegimeGate/code/src/model.py

C2_StrictClean_Baseline/code/src/train.py
C2_StrictClean_Baseline/code/src/model.py
```

核心训练 lineage 主要由以下文件决定：

```text
code/src/train.py
code/src/model.py
code/src/featurework.py
code/src/utils.py
code/src/validation.py
code/src/validation_metrics.py
```

## 可以留在 candidates_C 的变化

- 同一训练程序下的训练窗口变化。
- 同一训练程序下的 clean cutoff 或 strict purge 对照。
- 同一训练程序下的随机种子、checkpoint 选择结果对照。
- 推理期 rerank、组合权重、regime gate、`predict.py` 或 rerank config 差异。

## 不应留在 candidates_C 的变化

以下变化代表新的底层训练程序路线，应放入新的同级候选族目录，例如 `Experimental_WorkSpace/candidates_D/`：

- 修改 Transformer 结构或 `model.py` 语义。
- 修改训练主流程、dataset 构造、样本权重机制、loss、label 构造。
- 修改特征构造语义。
- 修改 optimizer/scheduler 的核心训练语义。
- 修改验证指标和 checkpoint 选择语义，并把它作为新训练程序路线继续发展。

## 新候选族要求

如果创建 `candidates_D/` 或后续 `candidates_E/`：

- 必须新增该目录自己的 `README.md`。
- 必须说明相对 C 系列的底层训练程序改动。
- 必须说明训练数据窗口、评估窗口、泄露审计要求。
- 必须在 `AGENTS.md` 和 `WORKSPACE_LAYOUT_20260705.md` 中同步更新目录规则。

## 使用边界

- 不在本目录下放提交包。
- 不在本目录下混入不同底层训练程序 lineage 的模型。
- 不把未评估清楚的模型提升为候选提交包。
