"""Tests for collect.py: the command line, the refusals, the phase from the output, and keys typed into a real pseudo-terminal."""

import pathlib
import sys
import textwrap

import pytest

import collect

FAKE = textwrap.dedent('''
    import os, select, sys, termios, tty
    tty.setcbreak(0)
    print("[collect-bi] mode=dry", flush=True)
    print("INFO Waiting: press 'a' to return both arms", flush=True)
    while True:
        if not select.select([0], [], [], 0.1)[0]:
            continue
        ch = os.read(0, 1)
        if ch == b"a":
            print("'a' key pressed. Aligning the arm to the start pose...", flush=True)
            print("INFO Arm returned. Restore the scene, then press the right arrow", flush=True)
        elif ch == b"\\x1b":
            seq = os.read(0, 8) if select.select([0], [], [], 0.2)[0] else b""
            if seq == b"[C":
                print("Right arrow key pressed. Exiting loop...", flush=True)
                print("INFO Recording episode 3", flush=True)
            elif seq == b"":
                print("Escape key pressed. Stopping data recording...", flush=True)
                print("[record-bi] lerobot-record exit=0", flush=True)
                sys.exit(0)
''')


@pytest.fixture
def env_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "x.env"
    path.write_text("TASK_SLUG=x\n")
    return path


@pytest.fixture
def session(tmp_path: pathlib.Path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, str(script)], {}))
    yield s
    if s.active:
        s.proc.kill()


def test_command_is_the_terminal_line(env_file):
    argv, env = collect.command(collect.Options(mode="session", tag="s2", episodes=30, env_file=env_file), pathlib.Path("/r"))
    assert argv == ["bash", "/r/scripts/collect_bimanual.sh", "session", "s2", "30"]
    assert env["ENV_FILE"] == str(env_file) and env["SAVE_ON_EPISODE_END"] == "true" and env["L_BILATERAL_GRIPPER_KP"] == "1.25"
    argv, env = collect.command(collect.Options(mode="gate", bilateral=False, save_on_episode_end=False, env_file=env_file), pathlib.Path("/r"))
    assert argv == ["bash", "/r/scripts/collect_bimanual.sh", "gate"]
    assert env["BILATERAL_FEEDBACK"] == "false" and "L_BILATERAL_GRIPPER_KP" not in env


def test_refuses_while_a_teleop_runs(tmp_path, env_file):
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: "run_teleop")
    with pytest.raises(collect.CollectError, match="col.err.teleop"):
        s.start(collect.Options(env_file=env_file))
    assert s.phase == "idle"


def test_refuses_a_missing_settings_file_and_a_session_without_tag(tmp_path, env_file):
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None)
    with pytest.raises(collect.CollectError, match="col.err.env"):
        s.start(collect.Options(env_file=tmp_path / "nope.env"))
    with pytest.raises(collect.CollectError, match="col.err.tag"):
        s.start(collect.Options(mode="session", env_file=env_file))


def test_phase_follows_the_output():
    s = collect.CollectSession(teleop_probe=lambda: None)
    for line, phase in (("INFO Waiting: press 'a' to return", "ready"), ("INFO Recording episode 4", "recording"), ("Right arrow key pressed. Exiting loop...", "returning"),
                        ("INFO Arm returned. Restore the scene", "ready"), ("Escape key pressed. Stopping data recording...", "stopping"), ("INFO Waiting: press", "stopping")):
        s.feed(line)
        assert s.phase == phase
    assert s.episode == 4 and s.started == 1


def test_run_folder_is_read_from_the_output():
    s = collect.CollectSession(teleop_probe=lambda: None)
    s.feed("\x1b[32m[record-bi] 記録: /home/x/collections/2026-abc\x1b[0m")
    assert s.run_dir == "/home/x/collections/2026-abc"


def test_keys_reach_a_real_terminal_and_stop_ends_the_run(session, env_file):
    session.start(collect.Options(env_file=env_file))
    wait_for(lambda: session.phase == "ready")
    session.send("align")
    wait_for(lambda: any("Arm returned" in line for line in session.snapshot()["lines"]))
    session.send("next")
    wait_for(lambda: session.phase == "recording")
    assert session.episode == 3
    session.send("stop")
    assert session.wait() == 0
    assert session.snapshot()["phase"] == "finished" and not session.active


def test_second_start_while_running_is_refused_and_send_without_run_fails(session, env_file):
    with pytest.raises(collect.CollectError, match="col.err.notrunning"):
        session.send("next")
    session.start(collect.Options(env_file=env_file))
    with pytest.raises(collect.CollectError, match="col.err.busy"):
        session.start(collect.Options(env_file=env_file))


def wait_for(cond, timeout: float = 5.0) -> None:
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def test_remaining_episodes_count_kept_takes_only():
    s = collect.CollectSession(teleop_probe=lambda: None)
    s.feed("[collect-bi] mode=gate tag=bi-x-v1-s1 episodes=10 scene=x")
    assert s.snapshot()["left"] == 10
    for line in ("INFO Recording episode 0", "Right arrow key pressed. Exiting loop...",  # kept
                 "INFO Arm returned. Restore the scene", "INFO Recording episode 1", "Left arrow key pressed. Exiting loop and rerecord",  # discarded
                 "INFO Re-record episode", "INFO Arm returned. Restore the scene", "INFO Recording episode 1", "INFO Arm returned. Restore the scene"):  # ended by the time limit: kept
        s.feed(line)
    snap = s.snapshot()
    assert (snap["done"], snap["left"], snap["target"]) == (2, 8, 10)
    s.feed("INFO Recording episode 2")
    s.feed("Escape key pressed. Stopping data recording...")  # Esc keeps the running take
    s.feed("INFO Stop recording")
    assert s.snapshot()["done"] == 3


def test_episode_time_and_pose_set_reach_the_script_only_when_given(env_file):
    _, env = collect.command(collect.Options(env_file=env_file), pathlib.Path("/r"))
    assert "EPISODE_TIME_S" not in env and "POSE_SET" not in env  # the settings file decides
    _, env = collect.command(collect.Options(env_file=env_file, episode_time_s=600, pose_set=" my-set "), pathlib.Path("/r"))
    assert env["EPISODE_TIME_S"] == "600" and env["POSE_SET"] == "my-set"


def test_env_value_reads_a_setting_without_quotes_or_comments(tmp_path):
    path = tmp_path / "a.env"
    path.write_text('EPISODE_TIME_S=120\nPOSE_SET=kitakyushu-wood-l1-v1   # comment\nTASK="Pick it up"\n')
    assert collect.env_value(path, "EPISODE_TIME_S") == "120"
    assert collect.env_value(path, "POSE_SET") == "kitakyushu-wood-l1-v1"
    assert collect.env_value(path, "TASK") == "Pick it up"
    assert collect.env_value(path, "NOPE") == "" and collect.env_value(tmp_path / "missing", "X") == ""


def test_start_pose_hold_is_passed_only_when_ticked(env_file):
    _, env = collect.command(collect.Options(env_file=env_file), pathlib.Path("/r"))
    assert env["START_POSE_HOLD"] == "true" and env["START_COUNTDOWN_S"] == "3"  # the page's default is on
    _, env = collect.command(collect.Options(env_file=env_file, start_pose_hold=False), pathlib.Path("/r"))
    assert "START_POSE_HOLD" not in env and "START_COUNTDOWN_S" not in env  # off: the script's own default, as before
    _, env = collect.command(collect.Options(env_file=env_file, start_pose_hold=True, start_countdown_s=3), pathlib.Path("/r"))
    assert env["START_POSE_HOLD"] == "true" and env["START_COUNTDOWN_S"] == "3"


def test_automatic_return_is_on_by_default_and_saved_episodes_are_counted(env_file):
    _, env = collect.command(collect.Options(env_file=env_file), pathlib.Path("/r"))
    assert env["RETURN_ON_EPISODE_END"] == "true"
    _, env = collect.command(collect.Options(env_file=env_file, return_on_episode_end=False), pathlib.Path("/r"))
    assert env["RETURN_ON_EPISODE_END"] == "false"
    s = collect.CollectSession(teleop_probe=lambda: None)
    s.feed("  episode 0 を保存中（腕は動かない） …")
    assert s.phase == "saving"
    s.feed("  episode 0 を保存した")
    assert s.snapshot()["saved"] == 1 and s.phase == "returning"
    s.feed("INFO Waiting: both arms are held at the start pose.")
    assert s.phase == "ready"
