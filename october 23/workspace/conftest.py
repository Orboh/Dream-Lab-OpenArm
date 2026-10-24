"""Test setup: the driver folder (zones, guard, poses) is on the path; nothing here touches hardware."""

import pathlib
import sys

DRIVER = pathlib.Path(__file__).resolve().parent.parent / "driver"
for path in (DRIVER, pathlib.Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import os  # noqa: E402

os.environ.setdefault("OPENARM_MAPPING_FILE", "/nonexistent/openarm_model_mapping")   # the tests never read the real ~/.dreamlabs record
