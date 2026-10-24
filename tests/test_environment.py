"""Smoke test: the environment assembles and core libraries import.

This confirms the container is sound before any uplift model is built
on top of it. It does not test the analysis, which comes in later steps.
"""

def test_core_imports():
    import numpy, pandas, sklearn, scipy, matplotlib  # noqa: F401

def test_causalml_imports():
    import causalml  # noqa: F401

def test_config_values():
    from src import config
    assert config.TRAIN_SIZE == 1_000_000
    assert config.TEST_SIZE == 1_000_000
    assert config.N_SUBSAMPLES == 50
    assert config.SUBSAMPLE_SIZE == 100_000
    assert config.TAU_THRESHOLD == 0.7
    assert len(config.MODELS) == 4