# Shared PyTorch Environment

Updated: 2026-07-20

The canonical GPU environment is:

```text
Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/
```

The duplicate root `.venv` was deleted only after the shared environment passed
the following checks:

- Python `3.11.9`
- PyTorch `2.10.0+cu128`
- CUDA runtime `12.8`
- CUDA available: `true`
- GPU detected: NVIDIA GeForce RTX 5060 Laptop GPU
- pandas `2.3.2`
- NumPy `2.3.3`
- scikit-learn `1.7.2`
- `pip check`: no broken requirements
- imports for torch, pandas, numpy, sklearn, joblib, and tqdm

Use the environment's Python directly from PowerShell:

```powershell
$python = ".\Experimental_WorkSpace\shared_envs\pytorch-cu128-py311\Scripts\python.exe"
& $python --version
```

Do not create another workspace-level `.venv` unless a future dependency change
cannot be satisfied safely by this environment. If the environment changes,
record the change and re-run dependency, CUDA, candidate inference, and relevant
unit checks.

The environment is experiment infrastructure. It must not be copied into a
submission package.
