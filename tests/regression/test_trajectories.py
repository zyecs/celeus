"""Fixed-seed trajectories must match the stored fixture bit for bit."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from cases import run_all_cases  # noqa: E402

FIXTURE = Path(__file__).with_name("trajectories.npz")


@pytest.fixture(scope="module")
def current():
    return run_all_cases()


def test_same_cases(current):
    with np.load(FIXTURE) as ref:
        assert sorted(ref.files) == sorted(current)


def test_bitwise_identical(current):
    with np.load(FIXTURE) as ref:
        for key in ref.files:
            expected, actual = ref[key], np.asarray(current[key])
            assert expected.dtype == actual.dtype, key
            assert expected.shape == actual.shape, key
            assert expected.tobytes() == actual.tobytes(), key
