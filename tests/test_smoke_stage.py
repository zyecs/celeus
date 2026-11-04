"""End-to-end check of the ``smoke`` stage on a small synthetic data root.

The real pool fixtures are not shipped with the code, so this test writes
random logits in the expected layout ({data_root}/{dataset}/{model}/*.pt),
points a copy of the paper config at them, and runs the four smoke cells
through ``scripts/run_paper_experiment.py``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
N_ITEMS = 400


def _write_model(dirpath: Path, logits: np.ndarray, labels: np.ndarray) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    scores = torch.from_numpy(logits).float()
    torch.save(scores, dirpath / "all_set_scores.pt")
    log_probs = torch.log_softmax(scores, dim=1)
    correct = (scores.argmax(dim=1).numpy() == labels).astype(np.float32)
    torch.save(torch.from_numpy(correct), dirpath / "all_set_per_sample_accuracy.pt")
    ce = -log_probs[torch.arange(len(labels)), torch.from_numpy(labels)]
    torch.save(ce, dirpath / "all_set_per_sample_cross_entropy_loss.pt")


@pytest.fixture()
def smoke_config(tmp_path: Path) -> Path:
    cfg = yaml.safe_load((REPO / "configs" / "paper_experiment.yaml").read_text())
    smoke = cfg["smoke"]
    rng = np.random.default_rng(0)
    n_classes = cfg["datasets"][smoke["dataset"]]["classes"]
    labels = rng.integers(0, n_classes, size=N_ITEMS)
    data_root = tmp_path / "data"
    dataset_dir = data_root / smoke["dataset"]
    dataset_dir.mkdir(parents=True)
    torch.save(torch.from_numpy(labels), dataset_dir / "all_set_targets.pt")
    signal = np.eye(n_classes)[labels] * 2.0
    for model, noise in ((smoke["target"], 1.0), (smoke["surrogate"], 1.5)):
        logits = signal + noise * rng.normal(size=(N_ITEMS, n_classes))
        _write_model(dataset_dir / model, logits, labels)

    cfg["paths"] = {"repo_root": str(REPO), "data_root": str(data_root),
                    "out_root": str(tmp_path / "results")}
    cfg["datasets"][smoke["dataset"]]["N"] = N_ITEMS
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_smoke_stage_runs(smoke_config: Path) -> None:
    cfg = yaml.safe_load(smoke_config.read_text())
    n_cells = len(cfg["smoke"]["methods"])
    for index in range(n_cells):
        proc = subprocess.run(
            [sys.executable, "scripts/run_paper_experiment.py", "--stage", "smoke",
             "--config", str(smoke_config), "--index", str(index)],
            cwd=REPO, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    cells = sorted((Path(cfg["paths"]["out_root"])).rglob("*.npz"))
    assert len(cells) >= n_cells, [str(c) for c in cells]
