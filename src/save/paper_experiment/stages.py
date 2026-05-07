"""Stage-dispatch entry points called by scripts/run_paper_experiment.py.

Each function takes ``(cfg: PaperExperimentConfig, index: int | None, dry_run: bool)``
and returns a POSIX exit code (0 = success, non-zero = fatal).
"""
from __future__ import annotations

from .config import PaperExperimentConfig


def _todo(name: str):
    def inner(*, cfg: PaperExperimentConfig, index: int | None, dry_run: bool) -> int:
        raise NotImplementedError(f"stage {name!r} not yet implemented")
    return inner


def run_analyze(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    import subprocess
    from pathlib import Path

    repo = Path(cfg.paths["repo_root"])
    py = cfg.paths["python"]

    scripts = [
        "scripts/paper_experiment/build_summary.py",
        "scripts/paper_experiment/check_missing_cells.py",
        "scripts/paper_fig_1.py",
        "scripts/paper_fig_2.py",
        "scripts/paper_fig_3.py",
        "scripts/paper_fig_4.py",
        "scripts/paper_fig_5.py",
        "scripts/paper_fig_6.py",
        "scripts/paper_fig_7.py",
        "scripts/paper_fig_8.py",
        "scripts/paper_fig_supp_trajectory.py",
        "scripts/paper_tables.py",
        "scripts/paper_experiment/build_reproduction_bundle.py",
    ]
    env = {"PYTHONPATH": str(repo / "src")}
    for rel in scripts:
        path = repo / rel
        if not path.exists():
            print(f"skip (missing): {rel}")
            continue
        if dry_run:
            print(f"would run: {rel}")
            continue
        print(f"run: {rel}")
        rc = subprocess.call([py, str(path)], cwd=str(repo), env=env)
        if rc != 0:
            print(f"FAILED: {rel} (rc={rc})")
            return rc
    return 0


def run_ce_sweep(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    """Stage 2: execute one CE-sweep sub-cell, OR (index == -1) select the winner.

    CE sweep uses only 5 seeds per cell (spec §5); one SLURM task per cell
    handles all 5 — no chunking — so assignment count = cell count = 90.
    """
    from pathlib import Path

    from .ce_sweep_select import load_ce_sweep_groups, select_ce_winner, write_winner
    from .cell_runner import run_subcell
    from .hostguard import ensure_compute_node
    from .index_maps import ce_sweep_cells, pick_cell
    from .pool_loader import load_pool_for_cell

    cells = ce_sweep_cells(cfg)
    if index is None:
        print(f"ce-sweep has {len(cells)} cells (+ index=-1 for selection)")
        return 0

    if index == -1:
        out_root = Path(cfg.paths["out_root"])
        groups = load_ce_sweep_groups(out_root, cells)
        if not groups:
            print("no ce-sweep cells found on disk yet — run the sweep first")
            return 1
        winner = select_ce_winner(
            groups,
            min_stop_rate=float(cfg.ce_sweep["selection"]["min_stop_rate"]),
            required_miscoverages=int(cfg.ce_sweep["selection"]["required_miscoverages"]),
            tie_break_pct=float(cfg.ce_sweep["selection"]["tie_break_pct"]),
        )
        write_winner(winner, out_root / "ce_sweep_winner.json")
        print(f"wrote ce_sweep_winner.json: {winner}")
        return 0

    cell = pick_cell(cells, index)
    if dry_run:
        print(f"DRY-RUN ce-sweep cell: {cell}")
        return 0

    ensure_compute_node()
    # CE sweep is always CE loss → pass cfg.ce_nll_filter to pool loader.
    # Eagerly load the pool once so T_max reflects filtered pool.N.
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=cell.dataset, surrogate=cell.surrogate, target=cell.target,
        loss="cross_entropy", surrogate_type=cell.surrogate_type,
        ce_nll_filter=cfg.ce_nll_filter,
    )
    T_max = int(pool.N)
    run_subcell(
        out_base=Path(cfg.paths["out_root"]),
        method_id="M1", dataset=cell.dataset,
        surrogate=cell.surrogate, target=cell.target, loss="cross_entropy",
        seeds=list(cfg.seeds_ce_sweep), T_max=T_max,
        pool_factory=lambda: pool,
        epsilon=cfg.protocol["epsilon"],
        alpha_1=cfg.protocol["alpha_1"], alpha_2=cfg.protocol["alpha_2"],
        theta=cfg.protocol["theta"], c_betting=cfg.protocol["c_betting"],
        c_fixed=cfg.protocol["c_fixed"], cs_grid_size=cfg.protocol["cs_grid_size"],
        beta_min=cell.beta_min,
        surrogate_type=cell.surrogate_type,
        adaptive_bounds=True, uniform_acquisition=False,
        monitor_to_T_max=cfg.protocol["monitor_to_T_max"],
        cereval_m_init=cfg.methods["M5"]["m_init"],
        chunk=0, kind="ce_sweep",
        surrogate_type_for_path=cell.surrogate_type,
    )
    return 0


def run_compute_rn(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    from pathlib import Path

    from .compute_rn import write_rn_csv

    data_root = Path(cfg.paths["data_root"])
    out_path = Path(cfg.paths["out_root"]) / "R_N.csv"

    targets = sorted({p["target"] for p in cfg.all_pairs})
    combos = [
        (d, t, l)
        for d in cfg.datasets
        for t in targets
        for l in cfg.losses
    ]
    if dry_run:
        print(f"would compute {len(combos)} R_N values to {out_path}")
        return 0
    write_rn_csv(
        data_root=data_root,
        datasets_pairs_losses=combos,
        out_path=out_path,
        ce_nll_filter=cfg.ce_nll_filter,
    )
    print(f"wrote {len(combos)} rows to {out_path}")
    return 0


def run_disk_audit(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    from pathlib import Path

    from .disk_audit import (
        AuditFailure, check_bytes_free, check_executable, check_inodes_free,
        check_no_uncommitted_src_deletions, check_readable,
        write_probe_and_read_back,
    )

    out_root = Path(cfg.paths["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        check_bytes_free(out_root, int(cfg.disk_audit["min_bytes_free"]))
        check_inodes_free(out_root, int(cfg.disk_audit["min_inodes_free"]))
        check_readable(Path(cfg.paths["data_root"]))
        check_executable(Path(cfg.paths["python"]))
        write_probe_and_read_back(Path(cfg.paths["repo_root"]) / cfg.disk_audit["probe_path"])
        check_no_uncommitted_src_deletions(Path(cfg.paths["repo_root"]))
    except AuditFailure as e:
        print(f"DISK AUDIT FAILED: {e}")
        return 2
    print("DISK AUDIT PASSED")
    return 0


def _resolve_method_config(
    cfg: PaperExperimentConfig, method_id: str, loss: str
) -> dict:
    """Return per-method runtime config for one (method, loss) pair."""
    m = cfg.methods[method_id]
    if method_id == "M5":
        return {"m_init": int(m["m_init"])}
    if loss == "accuracy":
        return {
            "uniform_acquisition": bool(m["uniform_acquisition"]),
            "adaptive_bounds": bool(m["adaptive_bounds"]),
            "surrogate_type": str(m["surrogate_type_accuracy"]),
            "beta_min": float(m["beta_min_accuracy"]),
        }

    surrogate_type = m["surrogate_type_ce"]
    beta_min = m["beta_min_ce"]
    if surrogate_type is None or beta_min is None:
        from json import loads
        from pathlib import Path

        winner_path = Path(cfg.paths["out_root"]) / "ce_sweep_winner.json"
        if not winner_path.exists():
            raise RuntimeError(
                f"CE loss requires ce_sweep_winner.json at {winner_path}; "
                "run --stage ce-sweep first."
            )
        winner = loads(winner_path.read_text())
        surrogate_type = surrogate_type or winner["surrogate_type"]
        beta_min = beta_min if beta_min is not None else winner["beta_min"]
    return {
        "uniform_acquisition": bool(m["uniform_acquisition"]),
        "adaptive_bounds": bool(m["adaptive_bounds"]),
        "surrogate_type": str(surrogate_type),
        "beta_min": float(beta_min),
    }


def _main_cell_dry_run(cell) -> None:
    print(
        f"DRY-RUN cell: method={cell.method_id} dataset={cell.dataset} "
        f"surrogate={cell.surrogate} target={cell.target} loss={cell.loss}"
    )


def _oracle_cell_dry_run(cell, chunk: int, seeds: list[int]) -> None:
    print(
        f"DRY-RUN oracle-accuracy cell: dataset={cell.dataset} "
        f"surrogate={cell.surrogate} target={cell.target}"
    )
    print(f"  chunk={chunk} seeds={list(seeds)}")


def _run_main_like_subcell(
    cfg: PaperExperimentConfig,
    cell,
    seeds: list[int],
    chunk: int,
    dry_run: bool,
    kind: str = "main",
) -> int:
    """v2: execute one (cell x seed-chunk) sub-cell."""
    if dry_run:
        _main_cell_dry_run(cell)
        print(f"  chunk={chunk} seeds={list(seeds)}")
        return 0

    from pathlib import Path

    from .cell_runner import run_subcell
    from .hostguard import ensure_compute_node
    from .pool_loader import load_pool_for_cell

    ensure_compute_node()

    method_cfg = _resolve_method_config(cfg, cell.method_id, cell.loss)

    if cell.method_id == "M5":
        # Cer-Eval needs embeddings for stratification. Its algorithm ignores
        # surrogate scores, but the loader still needs a concrete score file.
        if cell.loss == "accuracy":
            surrogate_type_for_load = "remark2_strategy4"
        else:
            from json import loads

            winner_path = Path(cfg.paths["out_root"]) / "ce_sweep_winner.json"
            winner = loads(winner_path.read_text())
            surrogate_type_for_load = winner["surrogate_type"]
    else:
        surrogate_type_for_load = method_cfg["surrogate_type"]

    need_embeddings = cell.method_id == "M5"
    # CE loss → pass cfg.ce_nll_filter; accuracy → pass None (filter is CE-only).
    filter_arg = (
        cfg.ce_nll_filter if cell.loss == "cross_entropy" else None
    )
    # Eagerly load pool so T_max reflects the filtered count (matches pool.N).
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss=cell.loss,
        surrogate_type=surrogate_type_for_load,
        load_embeddings=need_embeddings,
        ce_nll_filter=filter_arg,
    )
    T_max = int(pool.N)
    pool_factory = lambda: pool
    run_subcell(
        out_base=Path(cfg.paths["out_root"]),
        method_id=cell.method_id,
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss=cell.loss,
        seeds=list(seeds),
        T_max=T_max,
        pool_factory=pool_factory,
        epsilon=cfg.protocol["epsilon"],
        alpha_1=cfg.protocol["alpha_1"],
        alpha_2=cfg.protocol["alpha_2"],
        theta=cfg.protocol["theta"],
        c_betting=cfg.protocol["c_betting"],
        c_fixed=cfg.protocol["c_fixed"],
        cs_grid_size=cfg.protocol["cs_grid_size"],
        beta_min=method_cfg.get("beta_min", 0.4),
        surrogate_type=surrogate_type_for_load,
        adaptive_bounds=method_cfg.get("adaptive_bounds", False),
        uniform_acquisition=method_cfg.get("uniform_acquisition", False),
        monitor_to_T_max=cfg.protocol["monitor_to_T_max"],
        cereval_m_init=cfg.methods["M5"]["m_init"],
        chunk=chunk,
        kind=kind,
    )
    return 0


def run_smoke(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    """Stage 1 smoke: one cell x 3 seeds x 4 methods = 4 SLURM tasks."""
    from pathlib import Path

    from .cell_runner import merge_subcells
    from .index_maps import pick_assignment, smoke_cells

    cells = smoke_cells(cfg)
    seeds = list(cfg.seeds_smoke)
    if index is None:
        print(f"smoke has {len(cells)} cells; pass --index 0..{len(cells) - 1}")
        return 0
    asn = pick_assignment(cells, seeds, index, chunk_size=len(seeds))
    rc = _run_main_like_subcell(
        cfg,
        asn.cell,
        seeds=list(asn.seeds),
        chunk=asn.chunk_index,
        dry_run=dry_run,
    )
    if rc != 0 or dry_run:
        return rc

    merge_subcells(
        out_base=Path(cfg.paths["out_root"]),
        kind="main",
        method_id=asn.cell.method_id,
        dataset=asn.cell.dataset,
        surrogate=asn.cell.surrogate,
        target=asn.cell.target,
        loss=asn.cell.loss,
        expected_seeds=seeds,
        delete_subcells=True,
    )
    return 0


def run_main_accuracy(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    from .index_maps import chunked_assignments, main_cells_for_loss, pick_assignment

    cells = main_cells_for_loss(cfg, loss="accuracy", include_cereval=False)
    if index is None:
        total = len(chunked_assignments(cells, cfg.seeds_main))
        print(f"main-accuracy has {total} (cell x seed-chunk) assignments")
        return 0
    asn = pick_assignment(cells, list(cfg.seeds_main), index)
    return _run_main_like_subcell(
        cfg,
        asn.cell,
        seeds=list(asn.seeds),
        chunk=asn.chunk_index,
        dry_run=dry_run,
    )


def run_main_ce(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    from .index_maps import chunked_assignments, main_cells_for_loss, pick_assignment

    cells = main_cells_for_loss(cfg, loss="cross_entropy", include_cereval=False)
    if index is None:
        total = len(chunked_assignments(cells, cfg.seeds_main))
        print(f"main-ce has {total} (cell x seed-chunk) assignments")
        return 0
    asn = pick_assignment(cells, list(cfg.seeds_main), index)
    return _run_main_like_subcell(
        cfg,
        asn.cell,
        seeds=list(asn.seeds),
        chunk=asn.chunk_index,
        dry_run=dry_run,
    )


def run_cereval_fresh(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    """Stage 0: M5 fresh sweep over both losses.

    chunk_size=1: one seed per task. M5's per-seed wallclock on the larger
    pools (mmlu N=115k, agnews N=128k) approaches the 24h SLURM cliff when
    bundled three-to-a-task, and a mid-run TIMEOUT loses the whole subcell
    (write is atomic at end of the seed loop). Splitting per seed eliminates
    the cliff and increases queue parallelism.
    """
    from .index_maps import chunked_assignments, main_cells_for_loss, pick_assignment

    cells = (
        main_cells_for_loss(
            cfg, loss="accuracy", include_cereval=True, only_cereval=True
        )
        + main_cells_for_loss(
            cfg, loss="cross_entropy", include_cereval=True, only_cereval=True
        )
    )
    if index is None:
        total = len(chunked_assignments(cells, cfg.seeds_main, chunk_size=1))
        print(f"cereval-fresh has {total} (cell x seed-chunk) assignments")
        return 0
    asn = pick_assignment(cells, list(cfg.seeds_main), index, chunk_size=1)
    return _run_main_like_subcell(
        cfg,
        asn.cell,
        seeds=list(asn.seeds),
        chunk=asn.chunk_index,
        dry_run=dry_run,
    )


def run_beta_sweep(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    """Stage 5: M1 with sweep over beta (seed-chunked)."""
    from pathlib import Path

    from .cell_runner import run_subcell
    from .hostguard import ensure_compute_node
    from .index_maps import beta_sweep_cells, chunked_assignments, pick_assignment
    from .pool_loader import load_pool_for_cell

    cells = beta_sweep_cells(cfg)
    if index is None:
        total = len(chunked_assignments(cells, cfg.seeds_beta_sweep))
        print(f"beta-sweep has {total} (cell x seed-chunk) assignments")
        return 0
    asn = pick_assignment(cells, list(cfg.seeds_beta_sweep), index)
    cell = asn.cell

    if dry_run:
        print(
            f"DRY-RUN beta-sweep cell: dataset={cell.dataset} "
            f"surrogate={cell.surrogate} target={cell.target} "
            f"loss={cell.loss} beta={cell.beta_min} "
            f"chunk={asn.chunk_index} seeds={list(asn.seeds)}"
        )
        return 0

    ensure_compute_node()
    mcfg = _resolve_method_config(cfg, method_id="M1", loss=cell.loss)
    surrogate_type = mcfg["surrogate_type"]

    # Filter active for CE cells only (beta-sweep spans both losses).
    filter_arg = (
        cfg.ce_nll_filter if cell.loss == "cross_entropy" else None
    )
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss=cell.loss,
        surrogate_type=surrogate_type,
        ce_nll_filter=filter_arg,
    )
    T_max = int(pool.N)

    run_subcell(
        out_base=Path(cfg.paths["out_root"]),
        method_id="M1",
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss=cell.loss,
        seeds=list(asn.seeds),
        T_max=T_max,
        pool_factory=lambda: pool,
        epsilon=cfg.protocol["epsilon"],
        alpha_1=cfg.protocol["alpha_1"],
        alpha_2=cfg.protocol["alpha_2"],
        theta=cfg.protocol["theta"],
        c_betting=cfg.protocol["c_betting"],
        c_fixed=cfg.protocol["c_fixed"],
        cs_grid_size=cfg.protocol["cs_grid_size"],
        beta_min=cell.beta_min,
        surrogate_type=surrogate_type,
        adaptive_bounds=True,
        uniform_acquisition=False,
        monitor_to_T_max=cfg.protocol["monitor_to_T_max"],
        cereval_m_init=cfg.methods["M5"]["m_init"],
        chunk=asn.chunk_index,
        kind="beta_sweep",
    )
    return 0


def run_oracle_accuracy(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    from pathlib import Path

    from .cell_runner import run_subcell
    from .hostguard import ensure_compute_node
    from .index_maps import chunked_assignments, oracle_accuracy_cells, pick_assignment
    from .pool_loader import load_pool_for_cell

    cells = oracle_accuracy_cells(cfg)
    if index is None:
        total = len(chunked_assignments(cells, cfg.seeds_main))
        print(f"oracle-accuracy has {total} (cell x seed-chunk) assignments")
        return 0

    asn = pick_assignment(cells, list(cfg.seeds_main), index)
    cell = asn.cell
    if dry_run:
        _oracle_cell_dry_run(cell, asn.chunk_index, list(asn.seeds))
        return 0

    ensure_compute_node()
    method_cfg = _resolve_method_config(cfg, "M1", "accuracy")
    surrogate_type = "remark2_oracle_strategy4"
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss="accuracy",
        surrogate_type=surrogate_type,
        ce_nll_filter=None,
    )
    T_max = int(pool.N)
    run_subcell(
        out_base=Path(cfg.paths["out_root"]),
        method_id="ORACLE_ACC",
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss="accuracy",
        seeds=list(asn.seeds),
        T_max=T_max,
        pool_factory=lambda: pool,
        epsilon=cfg.protocol["epsilon"],
        alpha_1=cfg.protocol["alpha_1"],
        alpha_2=cfg.protocol["alpha_2"],
        theta=cfg.protocol["theta"],
        c_betting=cfg.protocol["c_betting"],
        c_fixed=cfg.protocol["c_fixed"],
        cs_grid_size=cfg.protocol["cs_grid_size"],
        beta_min=method_cfg["beta_min"],
        surrogate_type=surrogate_type,
        adaptive_bounds=method_cfg["adaptive_bounds"],
        uniform_acquisition=method_cfg["uniform_acquisition"],
        monitor_to_T_max=cfg.protocol["monitor_to_T_max"],
        cereval_m_init=cfg.methods["M5"]["m_init"],
        chunk=asn.chunk_index,
        kind="oracle_accuracy",
    )
    return 0


def run_acquisition_sweep(
    *, cfg: PaperExperimentConfig, loss: str, scope: str = "legacy",
    index: int | None, dry_run: bool
) -> int:
    """Stage 6 / 6b: acquisition-strategy sweep, seed-chunked.

    The CLI selects ``--loss`` so accuracy and CE share one stage but get
    independent SLURM arrays. ``oracle_accuracy`` is *not* re-emitted here —
    the accuracy oracle is reused at render time from
    ``trajectories/oracle_accuracy/`` (Strategy 4 oracle, written by
    ``oracle-accuracy``).

    ``scope``: 'legacy' (the original 6-cell sweep) or 'v0502' (the 12-cell
    sweep on the four paper_pairs). The legacy scope is the default
    so that the existing ``--stage acquisition-sweep`` callsite continues
    to work without changes.
    """
    from pathlib import Path

    from .cell_runner import run_subcell
    from .hostguard import ensure_compute_node
    from .index_maps import (
        AcquisitionSweepCell, acquisition_sweep_cells,
        chunked_assignments, pick_assignment,
    )
    from .pool_loader import load_pool_for_cell

    cells = acquisition_sweep_cells(cfg, loss=loss, scope=scope)
    seeds = (list(cfg.acquisition_sweep["seeds"]) if scope == "legacy"
             else list(cfg.seeds_main))
    if index is None:
        total = len(chunked_assignments(cells, seeds))
        print(f"acquisition-sweep ({loss}, {scope}) has {total} (cell × seed-chunk) assignments")
        return 0

    asn = pick_assignment(cells, seeds, index)
    cell: AcquisitionSweepCell = asn.cell
    if dry_run:
        print(
            f"DRY-RUN acquisition-sweep ({loss}, {scope}) cell={cell} "
            f"chunk={asn.chunk_index} seeds={list(asn.seeds)}"
        )
        return 0

    ensure_compute_node()
    # CE loss → pass cfg.ce_nll_filter; accuracy → pass None (filter is CE-only).
    filter_arg = (
        cfg.ce_nll_filter if loss == "cross_entropy" else None
    )
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss=loss,
        surrogate_type=cell.surrogate_type,
        ce_nll_filter=filter_arg,
    )
    T_max = int(pool.N)
    run_subcell(
        out_base=Path(cfg.paths["out_root"]),
        method_id="M1", dataset=cell.dataset,
        surrogate=cell.surrogate, target=cell.target, loss=loss,
        seeds=list(asn.seeds), T_max=T_max,
        pool_factory=lambda: pool,
        epsilon=cfg.protocol["epsilon"],
        alpha_1=cfg.protocol["alpha_1"], alpha_2=cfg.protocol["alpha_2"],
        theta=cfg.protocol["theta"],
        c_betting=cfg.protocol["c_betting"], c_fixed=cfg.protocol["c_fixed"],
        cs_grid_size=cfg.protocol["cs_grid_size"],
        beta_min=float(cfg.acquisition_sweep["beta_min"]),
        surrogate_type=cell.surrogate_type,
        adaptive_bounds=True, uniform_acquisition=False,
        monitor_to_T_max=cfg.protocol["monitor_to_T_max"],
        cereval_m_init=cfg.methods["M5"]["m_init"],
        chunk=asn.chunk_index, kind="acquisition_sweep",
        surrogate_type_for_path=cell.surrogate_type,
    )
    return 0


def run_hparam_sweep(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    """Stage 7 (§6.5 item #4): hyperparameter OAT sweep, seed-chunked.

    Each YAML config entry overrides the protocol kwargs ``alpha_1``, ``theta``,
    ``c_betting``, ``c_fixed`` for one (cell, config) trajectory. The deployed
    accuracy strategy ``remark2_strategy4`` is fixed across configs — this
    stage isolates the recipe knobs from the surrogate choice. Accuracy-only.
    """
    from pathlib import Path

    from .cell_runner import run_subcell
    from .hostguard import ensure_compute_node
    from .index_maps import (
        HparamSweepCell, hparam_sweep_cells,
        chunked_assignments, pick_assignment,
    )
    from .pool_loader import load_pool_for_cell

    cells = hparam_sweep_cells(cfg)
    seeds = list(cfg.hparam_sweep["seeds"])
    if index is None:
        total = len(chunked_assignments(cells, seeds))
        print(f"hparam-sweep has {total} (cell x seed-chunk) assignments")
        return 0

    asn = pick_assignment(cells, seeds, index)
    cell: HparamSweepCell = asn.cell
    if dry_run:
        print(
            f"DRY-RUN hparam-sweep cell={cell} "
            f"chunk={asn.chunk_index} seeds={list(asn.seeds)}"
        )
        return 0

    ensure_compute_node()
    # Hparam sweep is accuracy-only; no CE NLL filter applies.
    loss = str(cfg.hparam_sweep["loss"])
    # Surrogate type is fixed at the deployed accuracy strategy. The hparam
    # sweep isolates the recipe knobs from the surrogate choice (which is
    # varied separately in §6.5 item #3 / acquisition_sweep).
    surrogate_type = "remark2_strategy4"
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss=loss,
        surrogate_type=surrogate_type,
        ce_nll_filter=None,
    )
    T_max = int(pool.N)
    run_subcell(
        out_base=Path(cfg.paths["out_root"]),
        method_id="M1", dataset=cell.dataset,
        surrogate=cell.surrogate, target=cell.target, loss=loss,
        seeds=list(asn.seeds), T_max=T_max,
        pool_factory=lambda: pool,
        epsilon=cfg.protocol["epsilon"],
        # Per-config overrides: this stage *replaces* protocol α_1, θ,
        # c_betting, c_fixed with the matching configs[i] entry.
        alpha_1=cell.alpha_1,
        alpha_2=cfg.protocol["alpha_2"],
        theta=cell.theta,
        c_betting=cell.c_betting, c_fixed=cell.c_fixed,
        cs_grid_size=cfg.protocol["cs_grid_size"],
        beta_min=float(cfg.hparam_sweep["beta_min"]),
        surrogate_type=surrogate_type,
        adaptive_bounds=True, uniform_acquisition=False,
        monitor_to_T_max=cfg.protocol["monitor_to_T_max"],
        cereval_m_init=cfg.methods["M5"]["m_init"],
        chunk=asn.chunk_index, kind="hparam_sweep",
        config_name_for_path=cell.config_name,
    )
    return 0


def run_wallclock(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    """Stage 8 (§6.5 item #5, Task 9): Cer-Eval wall-clock instrumentation.

    Mirrors ``run_oracle_accuracy`` but uses ``method_id="M5"`` (Cer-Eval) and
    the wallclock seed list. ``chunk_size=1`` so each (cell, seed) is one
    SLURM task — Cer-Eval is 4–5h per seed, which is too slow for the default
    chunk_size of 5. 4 cells × 5 seeds = 20 SLURM tasks.
    """
    from pathlib import Path

    from .cell_runner import run_subcell
    from .hostguard import ensure_compute_node
    from .index_maps import (
        WallclockCell, chunked_assignments, pick_assignment, wallclock_cells,
    )
    from .pool_loader import load_pool_for_cell

    cells = wallclock_cells(cfg)
    seeds = list(cfg.wallclock["seeds"])
    if index is None:
        # chunk_size=1 → one task per (cell, seed).
        total = len(chunked_assignments(cells, seeds, chunk_size=1))
        print(f"wallclock has {total} (cell x seed-chunk) assignments")
        return 0

    asn = pick_assignment(cells, seeds, index, chunk_size=1)
    cell: WallclockCell = asn.cell
    if dry_run:
        print(
            f"DRY-RUN wallclock cell={cell} "
            f"chunk={asn.chunk_index} seeds={list(asn.seeds)}"
        )
        return 0

    ensure_compute_node()
    # Cer-Eval needs embeddings for stratification. Surrogate scores are
    # ignored by the algorithm itself but the loader still needs a concrete
    # score file — pick the deployed accuracy strategy to match
    # ``cereval-fresh`` so the input pool is identical.
    surrogate_type_for_load = "remark2_strategy4"
    pool = load_pool_for_cell(
        data_root=Path(cfg.paths["data_root"]),
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        loss=cell.loss,
        surrogate_type=surrogate_type_for_load,
        load_embeddings=True,
        ce_nll_filter=None,  # accuracy-only; CE NLL filter does not apply.
    )
    # Per-round timing only needs samples — we don't need to run Cer-Eval to
    # completion of the full pool. Cap T_max via cfg.wallclock.t_max_cap to
    # keep per-cell wall-clock bounded; falls back to pool.N if cap is unset.
    cap = cfg.wallclock.get("t_max_cap")
    T_max = int(pool.N) if cap is None else min(int(pool.N), int(cap))
    run_subcell(
        out_base=Path(cfg.paths["out_root"]),
        method_id="M5", dataset=cell.dataset,
        surrogate=cell.surrogate, target=cell.target, loss=cell.loss,
        seeds=list(asn.seeds), T_max=T_max,
        pool_factory=lambda: pool,
        epsilon=cfg.protocol["epsilon"],
        alpha_1=cfg.protocol["alpha_1"], alpha_2=cfg.protocol["alpha_2"],
        theta=cfg.protocol["theta"],
        c_betting=cfg.protocol["c_betting"], c_fixed=cfg.protocol["c_fixed"],
        cs_grid_size=cfg.protocol["cs_grid_size"],
        beta_min=0.4,  # unused by M5 but required by the runner signature.
        surrogate_type=surrogate_type_for_load,
        adaptive_bounds=False, uniform_acquisition=False,
        monitor_to_T_max=cfg.protocol["monitor_to_T_max"],
        cereval_m_init=cfg.methods["M5"]["m_init"],
        chunk=asn.chunk_index, kind="wallclock",
    )
    return 0


def run_merge_oracle_accuracy(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    from pathlib import Path

    from .cell_runner import merge_subcells
    from .index_maps import oracle_accuracy_cells

    cells = oracle_accuracy_cells(cfg)
    out_root = Path(cfg.paths["out_root"])
    targets = [(cell, list(cfg.seeds_main)) for cell in cells]

    if index is None:
        print(f"merge-oracle-accuracy has {len(targets)} cells to consolidate")
        return 0

    if not 0 <= index < len(targets):
        raise IndexError(
            f"--index {index} out of range for {len(targets)} oracle merge targets"
        )
    cell, expected_seeds = targets[index]
    if dry_run:
        print(f"DRY-RUN merge (oracle_accuracy) for cell {cell}")
        return 0

    merge_subcells(
        out_base=out_root,
        kind="oracle_accuracy",
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        surrogate_type="remark2_oracle_strategy4",
        expected_seeds=expected_seeds,
        delete_subcells=True,
    )
    return 0


def run_merge_cells(
    *, cfg: PaperExperimentConfig, index: int | None, dry_run: bool
) -> int:
    """Consolidate sub-cells into bundled cell .npz files."""
    from pathlib import Path

    from .cell_runner import merge_subcells
    from .index_maps import (
        acquisition_sweep_cells, beta_sweep_cells, ce_sweep_cells,
        hparam_sweep_cells, main_cells_for_loss, wallclock_cells,
    )

    out_root = Path(cfg.paths["out_root"])

    targets = []
    for loss in cfg.losses:
        for cell in main_cells_for_loss(cfg, loss=loss, include_cereval=True):
            targets.append(("main", cell, list(cfg.seeds_main)))
    for cell in ce_sweep_cells(cfg):
        targets.append(("ce_sweep", cell, list(cfg.seeds_ce_sweep)))
    for cell in beta_sweep_cells(cfg):
        targets.append(("beta_sweep", cell, list(cfg.seeds_beta_sweep)))
    # §6.5 acquisition-sweep: enumerate per-loss so the merge index range
    # extends naturally past beta_sweep without disturbing earlier ranges.
    # v0502: enumerate both scopes when v0502 cells are populated.
    acq_seeds_legacy = list(cfg.acquisition_sweep["seeds"])
    if acq_seeds_legacy:
        for loss in cfg.losses:
            for cell in acquisition_sweep_cells(cfg, loss=loss, scope="legacy"):
                targets.append(("acquisition_sweep", cell, acq_seeds_legacy))

    # v0502 cells use seeds_main (50 seeds). Gate on directory presence,
    # NOT on out_root path string (Opus MAJOR #2):
    v0502_subcells = out_root / "_subcells" / "acquisition_sweep"
    v0502_cells_dir = out_root / "trajectories" / "acquisition_sweep"
    v0502_has_data = (
        (v0502_subcells.exists() and any(v0502_subcells.iterdir()))
        or (v0502_cells_dir.exists()
            and any(v0502_cells_dir.glob("cell__acquisition_sweep__*__remark2_strategy2__*.npz")))
    )
    if v0502_has_data:
        acq_seeds_v0502 = list(cfg.seeds_main)
        for loss in cfg.losses:
            for cell in acquisition_sweep_cells(cfg, loss=loss, scope="v0502"):
                targets.append(("acquisition_sweep", cell, acq_seeds_v0502))
    # §6.5 hparam-sweep: extends index range past acquisition_sweep.
    hp_seeds = list(cfg.hparam_sweep["seeds"])
    if hp_seeds:
        for cell in hparam_sweep_cells(cfg):
            targets.append(("hparam_sweep", cell, hp_seeds))
    # §6.5 wallclock (Task 9): extends index range past hparam_sweep.
    wc_seeds = list(cfg.wallclock["seeds"])
    if wc_seeds:
        for cell in wallclock_cells(cfg):
            targets.append(("wallclock", cell, wc_seeds))

    if index is None:
        # Compute range boundaries from the constructed targets list so the
        # printout stays correct under cereval_scope changes.
        n_main = sum(1 for t in targets if t[0] == "main")
        n_ce = sum(1 for t in targets if t[0] == "ce_sweep")
        n_beta = sum(1 for t in targets if t[0] == "beta_sweep")
        n_acq = sum(1 for t in targets if t[0] == "acquisition_sweep")
        n_hp = sum(1 for t in targets if t[0] == "hparam_sweep")
        n_wc = sum(1 for t in targets if t[0] == "wallclock")
        main_end = n_main - 1
        ce_end = n_main + n_ce - 1
        beta_end = n_main + n_ce + n_beta - 1
        acq_end = n_main + n_ce + n_beta + n_acq - 1
        hp_end = n_main + n_ce + n_beta + n_acq + n_hp - 1
        wc_end = n_main + n_ce + n_beta + n_acq + n_hp + n_wc - 1
        print(
            f"merge-cells has {len(targets)} cells to consolidate "
            f"(main=0..{main_end}, ce_sweep={n_main}..{ce_end}, "
            f"beta_sweep={n_main + n_ce}..{beta_end}, "
            f"acquisition_sweep={n_main + n_ce + n_beta}..{acq_end}, "
            f"hparam_sweep={n_main + n_ce + n_beta + n_acq}..{hp_end}, "
            f"wallclock={n_main + n_ce + n_beta + n_acq + n_hp}..{wc_end})"
        )
        return 0

    if not 0 <= index < len(targets):
        raise IndexError(
            f"--index {index} out of range for {len(targets)} merge targets"
        )
    kind, cell, expected_seeds = targets[index]
    if dry_run:
        print(f"DRY-RUN merge ({kind}) for cell {cell}")
        return 0

    kwargs = dict(
        out_base=out_root,
        kind=kind,
        dataset=cell.dataset,
        surrogate=cell.surrogate,
        target=cell.target,
        expected_seeds=expected_seeds,
        delete_subcells=True,
    )
    if kind == "main":
        kwargs["method_id"] = cell.method_id
        kwargs["loss"] = cell.loss
    elif kind == "ce_sweep":
        kwargs["surrogate_type"] = cell.surrogate_type
        kwargs["beta_min"] = cell.beta_min
    elif kind == "beta_sweep":
        kwargs["loss"] = cell.loss
        kwargs["beta_min"] = cell.beta_min
    elif kind == "acquisition_sweep":
        kwargs["loss"] = cell.loss
        kwargs["surrogate_type"] = cell.surrogate_type
    elif kind == "hparam_sweep":
        kwargs["loss"] = cell.loss
        kwargs["config_name"] = cell.config_name
    elif kind == "wallclock":
        kwargs["method_id"] = "M5"
        kwargs["loss"] = cell.loss
    merge_subcells(**kwargs)
    return 0
