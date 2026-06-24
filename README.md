# CELEUS: Certifiable and Efficient Evaluation via E-Processes

Reference implementation accompanying paper accepted to the ICML'26 Hypothesis Workshop.

> **Note on package name.** The Python package is imported as `save` for
> historical reasons. All exposed APIs, CLI flags, and config keys retain that
> namespace; the project is referred to as **CELEUS** throughout the paper.

## Installation

Python ≥ 3.10 is required. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Runtime dependencies (`numpy`, `scipy`, `pyyaml`, `torch`, `matplotlib`,
`pytest`) are pinned in `pyproject.toml`. No GPU is required for the estimator
or the analysis pipeline; surrogate scoring assumes pre-computed logits stored
on disk (see **Data** below).

## Quick start (smoke test)

After installation, the `smoke` stage runs a single (dataset × pair × loss)
cell across four methods with three seeds and verifies that the saved
trajectories satisfy the protocol invariants:

```bash
python scripts/run_paper_experiment.py \
    --stage smoke \
    --config configs/paper_experiment.yaml
python scripts/paper_experiment/verify_smoke.py
```

Expected runtime: ~5 minutes on a single CPU core. A passing smoke test
indicates that the install, data path, and YAML schema are all correct.

## Data

The pool fixtures (target / surrogate logits and ground-truth labels for SST-2,
MMLU, and AG-News) are **not** shipped with the code. Two equivalent ways to
register their location:

1. Set `paths.data_root` in your config (see `configs/paper_experiment.yaml`).
2. Export `SAVE_DATA_ROOT=/abs/path/to/data` before invoking the runner.

Each pool is consumed as a `.npz` archive following the schema in
`src/save/paper_experiment/pool_loader.py`. Reviewer instructions for obtaining
the archives are in the supplementary material; the loader fails fast with a
descriptive error if any expected file is missing.

## Reproducing the paper

Every paper artefact is produced by `scripts/run_paper_experiment.py --stage X`
followed by an analysis script under `scripts/paper_experiment/`. The mapping
below is exact; figure numbers refer to the camera-ready manuscript layout.

| Paper artefact                           | Stage(s)                               | Analysis script                                       |
|------------------------------------------|----------------------------------------|-------------------------------------------------------|
| Pool-Hoeffding floor table (§3)          | `compute-rn`                           | `render_mae_axis.py`                                  |
| RQ1 — sample-efficiency at the stop time | `main-accuracy`, `main-ce`             | `plot_rq1_efficiency.py`, `render_per_cell_efficiency.py` |
| RQ3 — estimation error vs. budget        | `main-accuracy`, `main-ce`             | `plot_rq3_estimation_error.py`                        |
| RQ4 — unbiasedness of the e-estimator    | `main-accuracy`, `main-ce`             | `plot_rq4_unbiasedness.py`                            |
| RQ5 — signal MSE under the deployed acq. | `main-accuracy`, `main-ce`             | `plot_rq5_signal_mse.py`                              |
| RQ6 — variance reduction vs. baselines   | `main-accuracy`, `main-ce`             | `plot_rq6_variance.py`                                |
| RQ7 — complementarity of components      | `main-accuracy`                        | `compute_rq7_{predictors,outcomes}.py`, `plot_rq7_complementarity.py` |
| Cross-entropy strategy selection (App.)  | `ce-sweep`                             | `render_ce_appendix.py`                               |
| Acquisition-strategy ablation (App.)     | `acquisition-sweep`                    | `plot_acq_strategies_width.py`, `render_acquisition_appendix.py` |
| β-floor sensitivity (App.)               | `beta-sweep`                           | `render_beta_appendix.py`                             |
| Hyper-parameter sensitivity (App.)       | `hparam-sweep`                         | `render_hparam_appendix.py`                           |
| Wall-clock comparison (App.)             | `wallclock`                            | `render_wallclock_appendix.py`                        |
| Surrogate on/off ablation (App.)         | `main-accuracy`                        | `render_surrogate_onoff.py`                           |

A typical end-to-end reproduction is:

```bash
# 1. Pre-flight checks
python scripts/run_paper_experiment.py --stage disk-audit  --config configs/paper_experiment.yaml
python scripts/run_paper_experiment.py --stage compute-rn  --config configs/paper_experiment.yaml

# 2. Strategy selection on cross-entropy (writes ce_sweep_winner.json)
python scripts/run_paper_experiment.py --stage ce-sweep    --config configs/paper_experiment.yaml

# 3. Main campaign
python scripts/run_paper_experiment.py --stage main-accuracy --config configs/paper_experiment.yaml
python scripts/run_paper_experiment.py --stage main-ce       --config configs/paper_experiment.yaml

# 4. Ablations
python scripts/run_paper_experiment.py --stage beta-sweep        --config configs/paper_experiment.yaml
python scripts/run_paper_experiment.py --stage acquisition-sweep --config configs/paper_experiment.yaml
python scripts/run_paper_experiment.py --stage hparam-sweep      --config configs/paper_experiment.yaml
python scripts/run_paper_experiment.py --stage wallclock         --config configs/paper_experiment.yaml

# 5. Aggregate and analyse
python scripts/run_paper_experiment.py --stage merge-cells --config configs/paper_experiment.yaml
python scripts/run_paper_experiment.py --stage analyze     --config configs/paper_experiment.yaml
```

The `merge-cells` and `analyze` stages depend only on on-disk trajectories, so
the campaign can be parallelised across (cell × seed-chunk) tasks and merged
afterwards. Reproduction bundles for each figure can be exported with
`scripts/paper_experiment/build_reproduction_bundle.py`.

### Hardware and wall-clock budget

The estimator and analysis pipeline are CPU-only. The compared methods are
M1 (CELEUS, adaptive bounds + surrogate), M3 (CELEUS, no surrogate), M4 (naive
e-value baseline), and M5 (Cer-Eval baseline); per-trajectory wall-clock varies
by method, and Cer-Eval (M5) dominates total cost.

The full paper campaign comprises ≈ 50 seeds × {accuracy, CE} × 30 cells ×
4 methods. Trajectories are independent and the harness can be parallelised
across cells and seed chunks.

## Method overview (code ↔ paper)

| Module                          | Paper section                              |
|---------------------------------|--------------------------------------------|
| `src/save/core/estimator.py`    | E-process estimator (§3, Alg. 1)            |
| `src/save/core/confidence.py`   | Anytime-valid CS construction (§3.2)        |
| `src/save/inference/`           | Hoeffding / Bernstein / bootstrap baselines (§4) |
| `src/save/acquisition/`         | Acquisition functions: uniform, residual magnitude / variance, self / surrogate entropy (§5, Remarks 1–2) |
| `src/save/allocation/`          | Stratum allocation rules: uniform, proportional, Neyman (§5.3) |
| `src/save/surrogate*.py`        | Surrogate scoring under Strategies S1–S5 (§5.1) |
| `src/save/baselines/cereval.py` | Cer-Eval reproduction (§6, M5)              |
| `src/save/baselines/evalue.py`  | Naive e-value baseline (§6, M4)             |
| `src/save/paper_experiment/`    | Experiment harness, RNG streams, cell schema |

## Repository layout

```
configs/                          # YAML configs (paper + LLM-pair experiments)
src/save/                         # estimator, baselines, harness (importable as `save`)
scripts/run_paper_experiment.py   # single CLI entry-point for all stages
scripts/paper_experiment/         # plot, render, build, verify, compute scripts
```

`results/`, `logs/`, and other run-time artefacts are git-ignored.

## Configuration

All runtime knobs live in YAML; the `protocol`, `methods`, and per-stage
sections in `configs/paper_experiment.yaml` are documented inline and are the
single source of truth. Code paths read config, never magic numbers. To run a
modified protocol (e.g. tighter ε or a different α-split) it suffices to copy
the config and pass `--config path/to/your.yaml`.

## Determinism

Each (cell × seed) trajectory uses an independent `numpy.random.Generator`
seeded by `(seed_main, dataset_id, surrogate_id, target_id, loss)` via
`src/save/paper_experiment/rng_streams.py`. Trajectories are bit-reproducible
on a fixed CPU+BLAS combination. Numerical drift across BLAS implementations
is bounded by 1e-6 in the reported metrics.

## License

The code is released under the MIT License for the purpose of NeurIPS review;
see `LICENSE` for the full text once the non-anonymous version is published.
Pool fixtures and pretrained model logits inherit the licenses of their
respective sources.
