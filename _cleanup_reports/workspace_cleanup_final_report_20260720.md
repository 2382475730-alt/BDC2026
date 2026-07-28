# BDC2026 工作区清理、重编号与候选包规范化完成报告

日期：2026-07-20

## 结论

本次重构已完成。工作区现在只保留一个 C 系列提交回退包、一个 D 系列
研究冠军、一个 D 系列提交候选，以及名称和内容均未改变的 THU 官方参考。

- 清理动作实际删除：`20,231,992,148` bytes（`18.842511 GiB`）。
- 在新增两个规范候选包、精简 registry 和保留审计材料后，受影响范围净减少：
  `20,120,112,608` bytes（`18.738315 GiB`）。
- 根 `.venv` 去重已执行，单独释放 `5,048,371,063` bytes
  （`4.701662 GiB`）。
- `candidate/THU-BDC2026-main/` 保持 58 个文件、`48,069,050` bytes；
  清理执行前后内容树 SHA256 均为
  `85a54a5f2cc04cccb2f90f8b2ee25e438e5a53b372fa3a5c192815a405ab44b9`，
  最终 Git 路径状态为空。

## 新旧路径映射

| 旧身份/路径 | 新身份/路径 | 处理 |
| --- | --- | --- |
| `candidate/THU-BDC2026-main/` | 不变 | 官方参考，受保护 |
| `candidate/C1_Submission_App_20260703/` | `candidate/Candidate-C/` | 规范化为 C 系列回退提交身份 |
| `Experimental_WorkSpace/candidates_D/D02_FrozenResearchChampion_20260709/` | `Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/` | 研究冠军重新编号；legacy ID 为 D02 |
| D01 的 `production_refit_20260703/` | `candidate/Candidate-D/` | 构建规范 D 系列候选包 |
| `candidate/Improvement_THU_original_candidate_20260705/` | 无 | 失败/冗余候选，结论留档后删除 |

`D01_FrozenResearchChampion_20260709/RENAMING_ALIAS.json` 保存 D02→D01
映射。下一个值得长期保留的 D 系列研究快照从 `D02_...` 开始；Smoke、
Quick、fold 和 seed 运行只使用 run ID。

## 删除明细

| 类别 | 动作数 | 删除 bytes | GiB |
| --- | ---: | ---: | ---: |
| D 自动化已完成运行子目录 | 274 | 12,637,323,979 | 11.769425 |
| 重复根 `.venv` | 1 | 5,048,371,063 | 4.701662 |
| E 自动化已完成运行子目录 | 152 | 1,883,103,387 | 1.753777 |
| C0/C1/C2 重复研究包 | 3 | 270,425,715 | 0.251854 |
| D001、D03、D04 与 D 缓存 | 4 | 211,433,631 | 0.196913 |
| 旧 C 候选源目录 | 1 | 57,743,836 | 0.053778 |
| E 派生数据快照 | 1 | 47,894,456 | 0.044605 |
| D01 内重复预测、日志、缓存等 | 15 | 40,269,709 | 0.037504 |
| 失败原候选 | 1 | 35,376,574 | 0.032947 |
| E 缓存 | 1 | 49,798 | 0.000046 |

详细到每个路径的 460 项执行记录见
`_cleanup_reports/workspace_cleanup_execution_20260720.json`。

## 清理前后空间

不可变清理前清单记录的受影响范围：

- 清理前：`25,388,208,881` bytes（`23.644612 GiB`）
  - `candidate/`：`141,189,460` bytes
  - `Experimental_WorkSpace/`：`20,198,648,358` bytes
  - 根 `.venv`：`5,048,371,063` bytes
- 清理后：`5,268,096,273` bytes（`4.906297 GiB`）
  - `candidate/`：`161,791,843` bytes
  - `Experimental_WorkSpace/`：`5,106,304,430` bytes
- 净减少：`20,120,112,608` bytes（`18.738315 GiB`）

按当前未触及范围重建的可比口径，整个工作区约从
`25,675,726,966` bytes（`23.912384 GiB`）降至终验时的
`5,555,614,358` bytes（`5.174069 GiB`）。

“实际删除”大于“净减少”，是因为新建了 Candidate-C、Candidate-D 和
轻量审计/映射文件。

## 最终核心目录树

```text
candidate/
├─ Candidate-C/
├─ Candidate-D/
└─ THU-BDC2026-main/

Experimental_WorkSpace/
├─ candidates_C/
│  └─ README.md
├─ candidates_D/
│  ├─ D01_FrozenResearchChampion_20260709/
│  │  ├─ code_snapshot/
│  │  ├─ data/
│  │  ├─ production_refit_20260703/
│  │  ├─ D01_ACTIVE_CONFIG.json
│  │  ├─ D01_SNAPSHOT_MANIFEST.json
│  │  └─ RENAMING_ALIAS.json
│  ├─ automation_runs/        # 只有根级汇总、状态、审计和最终报告
│  ├─ D_experiment_registry.json
│  ├─ README.md
│  └─ run_d_*.py
├─ candidates_E/             # 无保留权重；只有轻量代码和终局证据
└─ shared_envs/
   └─ pytorch-cu128-py311/
```

D/E `automation_runs/` 的直接子目录数均为 0；活动范围内 `.log` 数为 0。
除唯一共享环境和受保护官方目录外，`__pycache__`、`.pyc` 数均为 0。
共享环境中的解释器缓存属于已验证环境组成，不作为项目代码缓存删除。

## 保留模型和身份

1. `candidate/THU-BDC2026-main/`
   - 官方参考；名称和内容未修改。
2. `candidate/Candidate-C/`
   - C 系列回退提交包。
   - 原 C0/C1/C2 研究副本已删除。
3. `Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/`
   - 当前 D Alpha Champion；legacy ID 为 D02。
   - Defensive Champion 仍为 `NONE`。
4. `candidate/Candidate-D/`
   - 由 D01 固定 12 epoch production refit 构建。
   - 是提交身份，不是新训练程序。
5. E 系列
   - 当前没有保留编号模型；未来只有通过冻结和晋升闸门的模型才获得
     `E01_...`。

## Candidate-C 终验

- 编译：PASS。
- CUDA 推理：PASS。
- `validate_submission.py`：`ok=true`，全部 7 项检查通过。
- 包大小：`57,561,174` bytes；32 个文件。
- `temp/`：空。
- 输出：5 个唯一股票，非负权重，权重和 `1.0`。
- 一次性 self-score：`-0.03812298792188259`。

风险：该 self-score 为负，不能作为改进证据。运行时入口未发现联网行为；
旧 Candidate-C Dockerfile 在镜像构建阶段会下载 TA-Lib，若比赛构建阶段也
要求完全离线，应在正式提交前将依赖源内置。

## Candidate-D 终验

- 编译：PASS。
- CUDA 推理：PASS，`15.7619435` 秒，低于 5 分钟限制。
- 固定种子完整复训：PASS，12 epoch，`3.7817045` 分钟，低于 8 小时限制。
- 59/59 个模型 state tensor 完全一致，最大参数差 `0.0`。
- 模型 state SHA256：
  `c22f866f4753d9deabdc900ad598a326497350b5b19c33972e25145dc3e87830`。
- production 模型文件 SHA256：
  `bd85f504599d1995bd19f254c78136be1cf42e74a349f32ed6bdec5ccb0ef86d`。
- scaler SHA256：
  `3d647f554d5e09c50e56488931deabd54ff08750edb645f265ff91d00bc62b38`。
- 训练数据 SHA256：
  `8e9cf37e5c53e4edafddc3f6f6b5843d4037376334e6f53cba068699095df1ad`。
- 结果 SHA256：
  `6a0a63f641317d5e41afa6c9fc2a7d627cac9dae444b4d086838c3d9315f1622`。
- 上述模型、scaler、数据和结果与 D01 production refit 源逐项一致。
- `validate_submission.py`：`ok=true`，全部 7 项检查通过。
- 包大小：`56,161,619` bytes；29 个文件。
- `temp/`：空。
- 输出：2 个唯一股票，各 `0.5`，权重和 `1.0`。
- 绝对工作区路径：0；运行时联网调用：0。
- 状态：`SUBMISSION_CANDIDATE_READY`。

研究证据必须带上以下限制：在 243 个样本、2025-01-03 至 2026-01-05
的记录协议下，D Top-2 平均组合收益 `0.0371198358` 高于官方 Top-5 的
`0.0125284993`，但官方有限分数比例只有 `0.1687243`，且更广泛的
Champion Gate 因稳健性、多种子和成本条件失败而为 `REJECT`。因此
Candidate-D 的“可提交/可复现”已验证，但不能把它描述为所有研究闸门均通过。

## 代码、文档和路径终验

- 89 个非共享环境/非官方 Python 文件以内存编译方式检查：0 个语法错误。
- 9 个关键 JSON/manifest 解析：0 个错误。
- D-series unit tests：30 项运行，26 PASS，4 SKIP，0 FAIL。
- 4 个 SKIP 仅依赖已按计划删除的 R5 run-local 失败挑战者源码；
  锁定审计与最终报告仍保留，当前 D01、分割、封存和共享策略测试均通过。
- 活动 README、布局文档、registry、canonical manifest 中旧 canonical
  路径命中：0。
- 旧候选、C0/C1/C2、D001、旧 D02 目录、D03、D04、根 `.venv` 和
  `tmp/` 均不存在。
- 新的 Candidate-C、Candidate-D、D01 和共享环境均存在。
- `git status` 下官方 THU 路径无修改或未跟踪文件。

冻结的 `D02_SNAPSHOT_MANIFEST.json`、旧实验报告和锁定审计仍保留历史 ID
及原路径，这是刻意保存的不可变 provenance，不属于活动 canonical path。

## 环境去重

唯一环境：

```text
Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/
```

终验版本：Python 3.11.9、PyTorch 2.10.0+cu128、CUDA 12.8、
pandas 2.3.2、NumPy 2.3.3、scikit-learn 1.7.2、joblib 1.5.2、
tqdm 4.67.1；CUDA 可用，GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU；
`pip check` 为 PASS。

## 审计文件

- 清理前清单（绝对路径）：
  `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\_cleanup_reports\workspace_cleanup_renumber_20260717.json`
- 执行记录：
  `_cleanup_reports/workspace_cleanup_execution_20260720.json`
- 退役模型结论：
  `_cleanup_reports/retired_model_conclusions_20260720.md`
- 候选终验：
  `_cleanup_reports/candidate_validation_20260720.json`

没有未完成的强制验证。剩余事项均是上述研究稳健性或 Candidate-C
构建期依赖风险，不影响本次“清理、重新编号、候选包规范化”任务本身完成。
