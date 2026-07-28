# Candidate-D 前瞻检验训练沙箱数据访问审计

审计时间：2026-07-28  
状态：`PASS`  
范围：`THU-BDC2026-main` 与 `Candidate-D` 的冻结训练、推理入口及控制器
子进程边界。

## 官方参考

- 训练入口：`candidate/THU-BDC2026-main/code/src/train.py`。
- 唯一市场输入为配置中的 `./data/train.csv`。
- 所谓验证集由同一因果训练前缀的最后一个月在进程内切分，不读取外部
  validation 文件；其日期不晚于当前决策日。
- 推理入口：`candidate/THU-BDC2026-main/code/src/predict.py`。
- 推理同样只读取 `./data/train.csv`，以该文件最大日期作为决策日期，
  不读取 `test.csv`、outcome 或 sealed vault。

## Candidate-D

- 训练入口：`candidate/Candidate-D/code/src/train.py`。
- 训练命令显式使用沙箱内 `data/train.csv` 和运行时 `model/config.json`；
  固定 12 epoch，不读取验证集、不做事后 checkpoint 选择。
- 推理入口：`candidate/Candidate-D/code/src/test.py`。
- 控制器的冻结命令显式传入
  `--history data/train.csv --test-data ""`，因此 `test_path=None`，
  `load_input` 只读取 history。
- `model/config.json` 虽保留 submission 包兼容字段
  `data.test_csv=data/test.csv`，但冻结训练入口不读取该字段，推理入口由
  空的命令行参数覆盖。
- `score_self.py` 和 `submission_spec.py` 中存在 `test.csv` 字样，但它们
  不属于前瞻协议的训练或推理入口，控制器不会调用。

## 控制器边界

- 每个候选均普通复制到独立 round staging 目录。
- 复制后清空 `output/`、`temp/` 和旧模型产物，删除
  `data/test.csv`，再把同一个 `shared_train.csv` 普通复制为各候选的
  `data/train.csv`。
- 复制后和训练前均重新核对三份训练文件 SHA-256 完全相同，并拒绝任何
  残留 `data/test.csv`。
- 训练/推理子进程只继承 CUDA、系统、临时目录和可执行路径所需的最小
  环境变量；不会继承 sealed vault、outcome、Codex 工作区或用户目录变量。
- 子进程命令出现 `sealed_scores`、`outcome`、`validation_csv` 或
  `score_vault` 时立即拒绝。
- 训练与推理工作目录固定为各候选 round 沙箱；冻结候选源树哈希在准备前
  及完整性审计时重复校验。

## 结论

实际调用链只具备当前决策日因果训练前缀的程序化读取路径。测试期 outcome
只在卖出日后由控制器外层评分阶段读取，不进入候选工作目录、训练命令、
推理命令或其环境变量。本审计未运行模型、未读取测试 outcome、未访问密封
收益，也未修改任何候选包。
