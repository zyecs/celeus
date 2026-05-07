"""
SAVE reporting — comparison plots and summary tables.

"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from save.benchmark import BenchmarkResult


def plot_comparison(result: "BenchmarkResult", output_path: str) -> None:
    """
    Generate CI width comparison plot: SAVE vs all baselines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_t = result.save_trajectory
    base_t = result.baseline_trajectory

    save_labels = save_t["total_labels"]
    save_width = save_t["pop_upper"] - save_t["pop_lower"]
    base_labels = base_t["total_labels"]
    base_width = base_t["pop_upper"] - base_t["pop_lower"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: CI width over labels
    ax1.plot(save_labels, save_width, color="tab:blue", label="SAVE", linewidth=1.5)
    ax1.plot(base_labels, base_width, color="tab:orange", label="E-value baseline (uniform)", linewidth=1.5, linestyle="--")

    # Cer-Eval baseline (if available)
    if result.cereval_baseline_trajectory is not None:
        ce_t = result.cereval_baseline_trajectory
        ce_labels = ce_t["total_labels"]
        ce_width = ce_t["pop_upper"] - ce_t["pop_lower"]
        ax1.plot(ce_labels, ce_width, color="tab:green", label="Cer-Eval baseline", linewidth=1.5, linestyle="--")

    eps = result.config.get("save_config", {}).get("epsilon", 0.02)
    ax1.axhline(y=eps, color="red", linestyle=":", linewidth=1, label=f"ε = {eps}")
    ax1.set_xlabel("Labels acquired")
    ax1.set_ylabel("CI width")
    ax1.set_title("CI Width Comparison")
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Right panel: SAVE CI band with true risk
    ax2.fill_between(
        save_labels,
        save_t["pop_lower"],
        save_t["pop_upper"],
        alpha=0.3,
        color="tab:blue",
        label="SAVE 95% CI",
    )
    ax2.plot(save_labels, save_t["R_hat"], color="tab:blue", linewidth=1, label="R̂ (SAVE)")
    ax2.axhline(y=result.true_risk, color="black", linestyle="-", linewidth=1, label=f"True R = {result.true_risk:.3f}")
    ax2.set_xlabel("Labels acquired")
    ax2.set_ylabel("Risk estimate")
    ax2.set_title("SAVE Confidence Sequence")
    ax2.legend()

    rho = result.surrogate_correlation
    dataset = result.config.get("dataset", "?")
    target = result.config.get("target_model", "?")
    surrogate = result.config.get("surrogate_model", "?")
    fig.suptitle(f"{dataset} | target: {target} | surrogate: {surrogate} | ρ = {rho:.3f}", fontsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def write_summary_table(results: list["BenchmarkResult"], output_path: str) -> None:
    """Write summary CSV and print table to stdout.

    The dedup key now includes ``loss_type`` so accuracy and
    cross-entropy runs with otherwise-identical configs coexist in one
    CSV. Legacy rows that pre-date the column default to ``"accuracy"``.
    """
    headers = [
        "dataset", "target_model", "surrogate_model", "surrogate_type",
        "loss_type",
        "true_R", "rho",
        "beta_min", "mode", "acquisition", "grid_size", "cs_range", "seed",
        "save_estimate", "save_ci_mid", "save_width", "save_labels",
        "baseline_ci_mid", "baseline_width", "baseline_labels",
        "cereval_ci_mid", "cereval_width", "cereval_labels",
        "efficiency_ratio",
    ]

    rows = []
    for r in results:
        eff_ratio = r.baseline_labels / r.save_labels if r.save_labels > 0 else float("inf")
        sc = r.config.get("save_config", {})
        loss_type = r.config.get("loss_type", "accuracy")
        beta_min = sc.get("beta_min", 0.05)
        fixed_horizon = sc.get("fixed_horizon", False)
        mode = "fixed_horizon" if fixed_horizon else "anytime"
        acquisition = "uniform" if r.config.get("uniform_acquisition", False) else "active"
        seed = sc.get("seed", 42)
        # Use runtime-computed values from benchmark [Stage 11 Fix 2]
        cs_range = r.runtime_cs_range if r.runtime_cs_range is not None else 1.0
        grid_size = r.runtime_grid_size if r.runtime_grid_size is not None else sc.get("cs_grid_size", 2000)
        rows.append({
            "dataset": r.config.get("dataset", "?"),
            "target_model": r.config.get("target_model", "?"),
            "surrogate_model": r.config.get("surrogate_model", "?"),
            "surrogate_type": r.config.get("surrogate_type", "?"),
            "loss_type": loss_type,
            "true_R": f"{r.true_risk:.4f}",
            "rho": f"{r.surrogate_correlation:.4f}",
            "beta_min": f"{beta_min}",
            "mode": mode,
            "acquisition": acquisition,
            "grid_size": grid_size,
            "cs_range": f"{cs_range:.1f}",
            "seed": seed,
            "save_estimate": f"{r.save_estimate:.4f}",
            "save_ci_mid": f"{r.save_ci_mid:.4f}",
            "save_width": f"{r.save_width:.4f}",
            "save_labels": r.save_labels,
            "baseline_ci_mid": f"{r.baseline_ci_mid:.4f}",
            "baseline_width": f"{r.baseline_width:.4f}",
            "baseline_labels": r.baseline_labels,
            "cereval_ci_mid": f"{r.cereval_baseline_ci_mid:.4f}" if r.cereval_baseline_ci_mid is not None else "—",
            "cereval_width": f"{r.cereval_baseline_width:.4f}" if r.cereval_baseline_width is not None else "—",
            "cereval_labels": r.cereval_baseline_labels if r.cereval_baseline_labels is not None else "—",
            "efficiency_ratio": f"{eff_ratio:.2f}",
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Append to existing CSV if present; avoid duplicate rows. The
    # ``or "accuracy"`` fallback on loss_type and the ``or "legacy"``
    # fallback on surrogate_type normalise legacy rows (missing column
    # or empty string) so dedup matches against freshly-built rows of
    # the same type — without the fallbacks, every run would silently
    # double-append.
    existing_keys = set()
    if os.path.isfile(output_path):
        with open(output_path, newline="") as f:
            reader = csv.DictReader(f)
            for erow in reader:
                key = (erow.get("dataset"), erow.get("target_model"),
                       erow.get("surrogate_model"),
                       erow.get("surrogate_type") or "legacy",
                       erow.get("loss_type") or "accuracy",
                       erow.get("beta_min", ""),
                       erow.get("mode", ""), erow.get("acquisition", "active"),
                       erow.get("seed", ""))
                existing_keys.add(key)
    new_rows = [
        r for r in rows
        if (r["dataset"], r["target_model"], r["surrogate_model"],
            r["surrogate_type"] or "legacy",
            r["loss_type"],
            str(r["beta_min"]), r["mode"], r["acquisition"],
            str(r["seed"])) not in existing_keys
    ]
    # Schema-migration guard: if the existing CSV header does not include every
    # column in ``headers``, rewrite the whole file with the current schema
    # rather than appending misaligned rows.  Missing values in old rows are
    # filled with "" so legacy data is preserved intact.
    needs_rewrite = False
    existing_rows: list[dict] = []
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_file_headers = reader.fieldnames or []
            # Ordered comparison, NOT set equality. csv.DictWriter writes
            # rows in the order of ``fieldnames``, so if the existing file
            # has the same columns in a DIFFERENT order, appending would
            # still produce misaligned rows. Use list equality to catch
            # both missing columns AND reordered columns.
            if list(headers) != list(existing_file_headers):
                needs_rewrite = True
                existing_rows = list(reader)
    if needs_rewrite:
        # Atomic rewrite: write to a sibling temp file and os.replace()
        # into the final location so a crash mid-write cannot lose the
        # legacy CSV. On first rewrite we also drop a .bak.pre-stage13
        # backup alongside summary.csv so the operator has a safety
        # net (see stage-12 migrate_summary_csv.py for the precedent).
        backup_path = output_path + ".bak.pre-stage13"
        if not os.path.isfile(backup_path):
            # Copy bytes (not the in-memory rows) so the backup preserves
            # the ORIGINAL legacy schema verbatim, not the migrated form.
            import shutil
            shutil.copy2(output_path, backup_path)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for erow in existing_rows:
                writer.writerow({h: erow.get(h, "") for h in headers})
            writer.writerows(new_rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    else:
        write_header = not os.path.isfile(output_path) or os.path.getsize(output_path) == 0
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
    print(f"  Summary saved ({len(new_rows)} new rows): {output_path}")

    # Print compact table
    compact_headers = [
        "dataset", "save_width", "save_labels",
        "baseline_width", "cereval_width",
        "efficiency_ratio",
    ]
    fmt = "{:<15} {:>12} {:>12} {:>14} {:>13} {:>10}"
    print()
    print(fmt.format(*compact_headers))
    print("-" * 95)
    for row in rows:
        print(fmt.format(*[str(row[h]) for h in compact_headers]))
