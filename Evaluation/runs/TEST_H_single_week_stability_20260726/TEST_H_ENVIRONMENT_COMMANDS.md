# TEST H-Series Environment and Commands

Status: `TEST / RESEARCH ONLY / REJECTED_AT_QUICK_GATE`

## Environment

- Python: `3.11.9` via `Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/Scripts/python.exe`
- PyTorch: `2.10.0+cu128`; CUDA runtime `12.8`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`; driver `592.01`
- NumPy/Pandas/scikit-learn: `2.3.3` / `2.3.2` / `1.7.2`
- Shared environment was reused; no project dependency environment was created or changed.

## Commands

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/Scripts/python.exe' `
  -m unittest discover -s Experimental_WorkSpace/candidates_H/tests -v

& 'Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/Scripts/python.exe' -B `
  Experimental_WorkSpace/candidates_H/code/run_h_stability.py --stage smoke

& 'Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/Scripts/python.exe' -B `
  Experimental_WorkSpace/candidates_H/code/run_h_stability.py --stage quick
```

The Quick command is resumable. One host-side termination occurred after 10/16
outer runs; the rerun verified and skipped completed run directories, then
trained only missing runs.

## Test coverage

Python stdlib `trace --count --missing` over the 16-unit-test suite recorded
`417` hit and `91` missed executable lines across the new H loss/protocol
modules: **82.09% line coverage**.
