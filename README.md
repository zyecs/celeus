# CELEUS: Certifiable and Efficient Evaluation via E-Processes

Reference implementation accompanying an anonymous NeurIPS submission.

## Install

```
pip install -e .
```

## Run

```
python scripts/run_paper_experiment.py --stage <STAGE> --config configs/paper_experiment.yaml
```

Stages: `compute-rn`, `disk-audit`, `smoke`, `ce-sweep`, `main-accuracy`,
`main-ce`, `beta-sweep`, `oracle-accuracy`, `acquisition-sweep`,
`hparam-sweep`, `wallclock`, `merge-cells`, `merge-oracle-accuracy`,
`analyze`.

## Layout

- `src/save/core` — estimator, confidence sequence, state
- `src/save/acquisition` — acquisition functions (uniform, residual, entropy)
- `src/save/allocation` — stratum allocation rules
- `src/save/inference` — Hoeffding/Bernstein/bootstrap CI methods
- `src/save/baselines` — Cer-Eval and e-value baselines
- `src/save/analysis` — diagnostic metrics and replay
- `src/save/paper_experiment` — paper experiment harness
- `scripts/paper_experiment` — figure/table/sweep scripts
- `configs` — YAML configs (paths must be set by user)

## Data and results

Pool fixtures and result archives are not included. Set `paths.data_root`
in your config (or the `SAVE_DATA_ROOT` env var) before running
data-dependent stages.
