"""Test setup: the safety guard and the arm lock are off by default so the older driver tests run as before; the new tests turn them on."""

import pytest


@pytest.fixture(autouse=True)
def _guard_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "0")
    monkeypatch.setenv("OPENARM_ZONES", str(tmp_path / "zones.yaml"))
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(tmp_path / "active_arms.json"))
