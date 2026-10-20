"""Importing `orboh.sim` makes MuJoCo render off-screen through EGL, on the NVIDIA GPU when there is one.

MuJoCo reads these variables when it is first imported, so this has to run before `import mujoco`:
  * MUJOCO_GL defaults to `egl`. The default GLFW backend can read the desktop framebuffer instead of the simulated scene:
    on 2026-10-20 a simulation-only run sent screenshots of the desktop to Dream-1 as "camera frames".
  * With Mesa's EGL (the default when several vendors are installed) rendering falls back to the CPU: ~150 ms per frame, and a
    render blocks the physics step, so the web console's live views made the control loop overrun its 250 ms watchdog and the
    simulated arm never moved. Pointing EGL at the NVIDIA vendor file makes a frame ~2 ms.
Opt out with ORBOH_SOFTWARE_GL=1, or set MUJOCO_GL / __EGL_VENDOR_LIBRARY_FILENAMES yourself. If no context can be created the
renders fail, `MujocoSim.render` returns None and callers use a grey frame.
"""

import os
from pathlib import Path

NVIDIA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

os.environ.setdefault("MUJOCO_GL", "egl")
if (os.environ["MUJOCO_GL"] == "egl" and not os.environ.get("ORBOH_SOFTWARE_GL")
        and NVIDIA_EGL_VENDOR.exists() and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ):
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(NVIDIA_EGL_VENDOR)
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")  # the only device left once the vendor list is the NVIDIA one
