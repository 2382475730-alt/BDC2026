# PyTorch 环境使用说明

本实验目录只保留一个本机 CUDA PyTorch 环境。后续训练、评估、搜索脚本都应复用这个环境，不要在每个 candidate 或 submodel 目录下复制 `.venv`。

## 唯一环境位置

环境目录：

```powershell
D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\08_experiment_generalization_selfscore_20260701\shared_envs\pytorch-cu128-py311
```

Python 可执行文件：

```powershell
D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\08_experiment_generalization_selfscore_20260701\shared_envs\pytorch-cu128-py311\Scripts\python.exe
```

当前验证版本：

- Python: `3.11.9`
- PyTorch: `2.10.0+cu128`
- CUDA runtime: `12.8`
- `torch.cuda.is_available()`: `true`

## 推荐调用方式

在任意 candidate 目录中运行训练或评估时，使用共享入口脚本：

```powershell
& "D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\08_experiment_generalization_selfscore_20260701\scripts\torch_python.ps1" -X utf8 code/src/train.py --work-dir .
```

如果只需要拿到 Python 路径：

```powershell
$TORCH_PYTHON = & "D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\08_experiment_generalization_selfscore_20260701\scripts\torch_python.ps1"
& $TORCH_PYTHON -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

已有脚本如果提供 `--python` 参数，应传入上面的 `$TORCH_PYTHON`，例如：

```powershell
$EXP = "D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\08_experiment_generalization_selfscore_20260701"
$TORCH_PYTHON = & "$EXP\scripts\torch_python.ps1"
& $TORCH_PYTHON "$EXP\scripts\run_rerank_search.py" --candidate-dir "$EXP\candidates\C96_final_candidate" --python $TORCH_PYTHON --logs-dir "$EXP\night_search_20260702\logs" --summary "$EXP\night_search_20260702\summary.json"
```

## 后续规则

- 不要在 `candidates\*\.venv` 或 `candidates\*\submodels\*\.venv` 下创建新的本地环境。
- 不要把 `shared_envs\pytorch-cu128-py311` 复制进 candidate 目录。
- 如果新增依赖，先确认是否所有训练都需要；需要时安装到共享环境一次即可。
- 推荐安装命令：

```powershell
$TORCH_PYTHON = & "D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\08_experiment_generalization_selfscore_20260701\scripts\torch_python.ps1"
& $TORCH_PYTHON -m pip install <package>
```

Dockerfile 里的 `/app/.venv` 是容器内部路径，和本机共享环境不是一回事，不需要改。
