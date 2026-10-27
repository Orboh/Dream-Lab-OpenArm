"""Test setup: the safety guard and the arm lock are off by default so the older driver tests run as before; the new tests turn them on."""

import pytest


@pytest.fixture(autouse=True)
def _legacy_motion_by_default(monkeypatch):
    """The older tests were written for direct, unsmoothed commands; the smooth-motion tests switch smoothing on per arm (cfg.smooth)."""
    import openarm_v2 as drv

    monkeypatch.setattr(drv.Arm, "smooth_default", False, raising=False)
    monkeypatch.setattr(drv.Arm, "sync_default", True, raising=False)


@pytest.fixture(autouse=True)
def _guard_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "0")
    monkeypatch.setenv("OPENARM_ZONES", str(tmp_path / "zones.yaml"))
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(tmp_path / "active_arms.json"))

@pytest.fixture(autouse=True)
def _fresh_arm_pair():
    """The arm-to-arm record lives for the whole process: no pose may carry over from one test's arms to the next (log-034)."""
    import openarm_pair

    openarm_pair.reset()
    yield
    openarm_pair.reset()

import os  # noqa: E402

os.environ.setdefault("OPENARM_MAPPING_FILE", "/nonexistent/openarm_model_mapping")   # the tests never read the real ~/.dreamlabs record
