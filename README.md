# CELEUS: Certifiable and Efficient LLM Evaluation via E-Processes

Code for the NeurIPS 2026 paper *CELEUS: Certifiable and Efficient LLM Evaluation via E-Processes*.

CELEUS evaluates an LLM on a fixed pool of examples by revealing target losses one at a time. It samples
examples with an uncertainty-guided rule, corrects a cheap surrogate score with the revealed losses, and
maintains an anytime-valid e-process confidence interval for the pool risk. Evaluation stops once the
interval is narrower than a target width.

## Installation

Python 3.10 or later. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pandas            # used by the figure and table scripts
```

The experiment harness needs NumPy 1.25 or later (it uses `numpy.random.Generator.spawn`).
No GPU is needed: the method and the analysis run on CPU from precomputed model outputs.

Run the tests:

```bash
pytest tests
```

`tests/regression` replays every method on fixed synthetic pools and compares the trajectories bit for bit
with a stored fixture. `tests/test_smoke_stage.py` runs the `smoke` stage end to end on a synthetic data root.

## Data

Model outputs are not included in this repository. The loader expects PyTorch tensors in this layout:

```
<data_root>/<dataset>/all_set_targets.pt                                 # integer labels
<data_root>/<dataset>/<model>/all_set_scores.pt                          # logits, shape (N, C)
<data_root>/<dataset>/<model>/all_set_per_sample_accuracy.pt             # 0/1 correctness, shape (N,)
<data_root>/<dataset>/<model>/all_set_per_sample_cross_entropy_loss.pt   # shape (N,)
<data_root>/<dataset>/<model>/all_set_embeddings_layerm2_last.pt         # only for Cer-Eval
```

Datasets are `sst2`, `mmlu` and `agnews`. Model directories are `llama2_7b`, `llama3_8b`, `llama3_70b`,
`Mixtral_8x7b`, `deepseek_67b` and `qwen25_72b`. Set `paths.data_root` in `configs/paper_experiment.yaml`
(default `./data`); `celeus.analysis.replay` reads `CELEUS_DATA_ROOT` instead.

## Quick start

The smoke stage runs one dataset and model pair (SST-2, Llama-2-7B to Mixtral-8x7B) for the four methods
M1 to M4 with three seeds. Each `--index` is one method:

```bash
for i in 0 1 2 3; do
    python scripts/run_paper_experiment.py --stage smoke --index $i
done
python scripts/paper_experiment/verify_smoke.py
```

To use the method on your own losses, see `celeus.benchmark.run_benchmark` and
`celeus.paper_experiment.runners.run_celeus_method_for_seed`; `tests/regression/cases.py` shows a minimal call
on a synthetic pool.

## Methods

| ID | Name in the paper | Setting |
|----|-------------------|---------|
| M1 | CELEUS | surrogate correction, uncertainty-guided sampling, per-round scaling bounds |
| M2 | CELEUS with fixed scaling bounds | as M1 with one global pair of bounds |
| M3 | CELEUS without surrogate | surrogate set to zero, uniform sampling |
| M4 | E-value baseline | uniform sampling, plain e-process |
| M5 | Cer-Eval | baseline of Wang et al. (2025) |

All settings are in `configs/paper_experiment.yaml`. Code reads parameters from this file; to change the
protocol, copy it and pass `--config path/to/copy.yaml`.

## Reproducing the paper

Experiments are split into cells (dataset, model pair, loss, method) and seed chunks so that each
`--index` can run as one SLURM array task. `--dry-run` prints the assignment without running it.

```bash
CFG=configs/paper_experiment.yaml
python scripts/run_paper_experiment.py --stage disk-audit --config $CFG
python scripts/run_paper_experiment.py --stage compute-rn --config $CFG
python scripts/run_paper_experiment.py --stage ce-sweep   --config $CFG --index <i>   # writes ce_sweep_winner.json
python scripts/run_paper_experiment.py --stage main-accuracy --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage main-ce       --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage oracle-accuracy --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage cereval-fresh --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage beta-sweep    --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage hparam-sweep  --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage wallclock     --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage acquisition-sweep --loss accuracy --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage paper-acq-sweep   --loss accuracy --config $CFG --index <i>
python scripts/run_paper_experiment.py --stage merge-cells           --config $CFG
python scripts/run_paper_experiment.py --stage merge-oracle-accuracy --config $CFG
python scripts/paper_experiment/build_paper_outputs.py                     # main figures and tables
python scripts/paper_experiment/build_paper_outputs.py --part acquisition  # acquisition appendix
```

Run a stage without `--index` to print how many tasks it has. Trajectories are written to
`results/paper_experiment/`; figures and tables to `results/paper_outputs/`.

| Paper output | Stages | Script (`scripts/paper_experiment/`) |
|---|---|---|
| Width vs. evaluated samples, 0-1 loss (main text) | `main-accuracy`, `oracle-accuracy` | `run_rq1_efficiency.py` |
| Bias, signal MSE and variance (main text) | `main-accuracy`, `oracle-accuracy` | `regen_plots_only.py`, `plot_rq4_rq5_rq6_combined.py` |
| Width vs. evaluated samples, cross-entropy (App.) | `ce-sweep`, `main-ce` | `run_rq1_efficiency.py --loss cross_entropy` |
| Coverage and samples-to-width, cross-entropy (App.) | `ce-sweep`, `main-ce` | `render_ce_appendix.py` |
| Acquisition-strategy comparison (App.) | `acquisition-sweep`, `paper-acq-sweep` | `render_acquisition_appendix.py`, `plot_acq_strategies_width.py` |
| Sensitivity to β (App., Figure 6) | `beta-sweep` | `render_beta_appendix.py` |
| Hyperparameter sensitivity (App., Table 8) | `hparam-sweep` | `render_hparam_appendix.py` |
| Per-round wall-clock cost (App., Table 9) | `wallclock` | `render_wallclock_appendix.py` |
| Width per model pair (App.) | `main-accuracy` | `plot_per_pair_efficiency.py` |
| Per-cell results and surrogate on/off (App.) | `main-accuracy`, `main-ce` | `render_per_cell_efficiency.py`, `render_per_cell_mechanism.py`, `render_surrogate_onoff.py` |
| Surrogate and acquisition complementarity (App.) | `main-accuracy` | `compute_rq7_predictors.py`, `compute_rq7_outcomes.py`, `plot_rq7_complementarity.py` |

`build_paper_outputs.py` runs all of these in order. To run one script on its own, start it from the
repository root with `PYTHONPATH=.:src`. Not in this repository: the synthetic experiment of
Table 1 (left), the open-ended safety evaluation (App. L.1) and the break-even timing study (App. L.3).

## Repository layout

```
configs/paper_experiment.yaml     experiment settings (single source of truth)
src/celeus/core/                  estimator, e-process confidence sequence, run state
src/celeus/acquisition/           sampling rules (residual magnitude, uniform, entropy-based)
src/celeus/baselines/             e-value baseline (M4) and Cer-Eval (M5)
src/celeus/paper_experiment/      experiment harness: config, cells, RNG streams, stages
src/celeus/analysis/              replay and metrics for the bias/MSE/variance figures
scripts/run_paper_experiment.py   command-line entry point for all stages
scripts/paper_experiment/         figure, table and summary scripts
tests/                            regression and smoke tests
```

## Reproducibility

Each seed spawns four independent NumPy generators (`celeus.paper_experiment.rng_streams`), so a given
config and seed give the same trajectory on the same platform. The regression test checks this for every
method.

## Citation

```bibtex
@inproceedings{celeus2026,
  title     = {{CELEUS}: Certifiable and Efficient {LLM} Evaluation via E-Processes},
  author    = {Zhou, Zhijian and Ye, Zesheng and Chen, Zhaorun and Li, Bo and Liu, Feng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT License; see `LICENSE`. Model outputs and datasets keep the licenses of their sources.
