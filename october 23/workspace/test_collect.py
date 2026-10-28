"""Tests for collect.py: the command line, the refusals, the phase from the output, and keys typed into a real pseudo-terminal."""

import json
import os
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
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, str(script)], {}), set_file=tmp_path / "set.json")
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


def test_operator_reaches_the_script_and_an_empty_or_full_width_one_is_refused(tmp_path, env_file):
    _, env = collect.command(collect.Options(operator=" Suraj ", env_file=env_file))
    assert env["OPERATOR"] == "Suraj"  # wins over the settings file's OPERATOR (collect_bimanual.sh)
    _, env = collect.command(collect.Options(env_file=env_file))
    assert "OPERATOR" not in env
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, set_file=tmp_path / "set.json")
    with pytest.raises(collect.CollectError, match="col.err.operator$"):
        s.start(collect.Options(operator="  ", env_file=env_file))
    with pytest.raises(collect.CollectError, match="col.err.operator.chars"):
        s.start(collect.Options(operator="Ｓｕｒａｊ", env_file=env_file))  # full width, as typed with a Japanese keyboard
    assert s.phase == "idle" and not (tmp_path / "set.json").exists()


def test_task_files_are_listed_and_their_four_values_read(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "bimanual-collect.wood.env").write_text('TASK="Pick up the wood"\nSCENE=wood-v1  # scene\nPOSE_SET=p1\nEPISODE_TIME_S=600\n')
    (cfg / "bimanual-collect.sort.env").write_text("TASK=Sort\n")
    (cfg / "other.env").write_text("TASK=x\n")
    assert [f.name for f in collect.env_files(tmp_path)] == ["bimanual-collect.sort.env", "bimanual-collect.wood.env"]
    assert collect.env_summary(cfg / "bimanual-collect.wood.env") == {"TASK": "Pick up the wood", "SCENE": "wood-v1", "POSE_SET": "p1", "EPISODE_TIME_S": "600"}
    assert collect.env_summary(cfg / "bimanual-collect.sort.env")["SCENE"] == ""


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
    session.start(collect.Options(env_file=env_file, operator="Tester"))
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
    session.start(collect.Options(env_file=env_file, operator="Tester"))
    with pytest.raises(collect.CollectError, match="col.err.busy"):
        session.start(collect.Options(env_file=env_file, operator="Tester"))


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


def test_remaining_count_follows_the_japanese_banner():
    """The fork prints `録画中 episode N` and no longer `Recording episode N` (2026-10-26: 残り本数 stayed at 12 for a whole run)."""
    s = collect.CollectSession(teleop_probe=lambda: None, set_file=None)
    s.feed("[collect-bi] mode=session tag=bi-x-v1-s2 episodes=12 scene=x")
    for n in range(3):
        for line in ("Right arrow key pressed. Exiting loop...", f"  録画中 episode {n}（600 秒で自動終了）", "Right arrow key pressed. Exiting loop...",
                     f"  episode {n} を保存中（腕は動かない） …", f"  episode {n} を保存した", "INFO Waiting: both arms are held at the start pose."):
            s.feed(line)
    snap = s.snapshot()
    assert (snap["done"], snap["saved"], snap["left"], snap["episode"]) == (3, 3, 9, 2)
    s.feed("  録画中 episode 3（600 秒で自動終了）")
    s.feed("Left arrow key pressed. Exiting loop and rerecord the last episode...")  # discarded: not counted
    assert s.snapshot()["left"] == 9


def _dataset(root: pathlib.Path, episodes: int) -> pathlib.Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(f'{{"total_episodes": {episodes}}}')
    return root


def test_a_stopped_set_resumes_with_only_the_episodes_left_into_its_dataset(tmp_path, env_file):
    run = tmp_path / "run"
    run.mkdir()
    ds = _dataset(tmp_path / "ds_v1_20261026_141740", 8)
    (run / "manifest.json").write_text('{"artifacts": {"dataset_root": "%s", "repo_id_stamped": "Orboh/ds_v1_20261026_141740"}}' % ds)
    script = tmp_path / "fake.py"
    script.write_text(f"print('[record-bi] 記録: {run}')\n")
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, str(script)], {}), set_file=tmp_path / "set.json")
    s.start(collect.Options(mode="gate", env_file=env_file, operator="Tester"))
    assert s.wait() == 0
    todo = s.pending()
    assert (todo["mode"], todo["target"], todo["have"], todo["left"]) == ("gate", 10, 8, 2)
    opts = s.resume_options(collect.Options(env_file=env_file, pose_set="v5", operator="Tester"))
    assert (opts.mode, opts.episodes, opts.pose_set) == ("gate", 2, "v5")
    _, env = collect.command(opts, pathlib.Path("/r"))
    assert env["RESUME"] == "true" and env["DATASET_ROOT"] == str(ds) and env["REPO_ID"] == "Orboh/ds_v1_20261026_141740"
    s.start(opts)  # the resume keeps the set: once the dataset holds 10 there is nothing left
    assert s.wait() == 0
    (ds / "meta" / "info.json").write_text('{"total_episodes": 10}')
    assert s.pending() is None
    with pytest.raises(collect.CollectError, match="col.err.noresume"):
        s.resume_options(collect.Options(env_file=env_file))


def test_a_set_that_saved_nothing_starts_again_with_a_new_dataset_and_dry_leaves_the_set_alone(tmp_path, env_file):
    set_file = tmp_path / "set.json"
    set_file.write_text('{"mode": "session", "tag": "s2", "target": 12, "dataset_root": "%s", "repo_id": "Orboh/x"}' % _dataset(tmp_path / "empty", 0))
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, "-c", "pass"], {}), set_file=set_file)
    opts = s.resume_options(collect.Options(env_file=env_file))
    assert (opts.mode, opts.tag, opts.episodes, opts.resume_root) == ("session", "s2", 12, "")
    _, env = collect.command(opts, pathlib.Path("/r"))
    assert "RESUME" not in env
    s.start(collect.Options(mode="dry", env_file=env_file, operator="Tester"))
    s.wait()
    assert s.pending()["left"] == 12


def test_a_session_waits_until_the_gate_is_reviewed(tmp_path, env_file):
    ds = _dataset(tmp_path / "ds", 10)
    set_file = tmp_path / "set.json"
    set_file.write_text('{"mode": "gate", "tag": "", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % ds)
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, "-c", "pass"], {}), set_file=set_file)
    assert s.review_pending()["have"] == 10
    with pytest.raises(collect.CollectError, match="col.err.review"):
        s.start(collect.Options(mode="session", tag="s2", env_file=env_file, operator="Tester"))
    s.start(collect.Options(mode="dry", env_file=env_file, operator="Tester"))  # a dry run is still allowed (a new gate needs the confirmation, see below)
    assert s.wait() == 0
    s.mark_reviewed("Kota")
    assert s.review_pending() is None
    saved = json.loads(set_file.read_text())
    assert saved["reviewed_by"] == "Kota" and saved["reviewed_at"]
    s.start(collect.Options(mode="session", tag="s2", env_file=env_file, operator="Tester"))
    assert s.wait() == 0


def test_a_gate_that_stopped_short_blocks_a_session_until_it_is_finished(tmp_path, env_file):
    ds = _dataset(tmp_path / "ds", 7)
    set_file = tmp_path / "set.json"
    set_file.write_text('{"mode": "gate", "tag": "", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % ds)
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, set_file=set_file)
    assert s.review_pending() is None and s.pending()["left"] == 3  # resume it first; there is nothing to review yet
    with pytest.raises(collect.CollectError, match="col.err.gateopen"):
        s.start(collect.Options(mode="session", tag="s2", env_file=env_file, operator="Tester"))


def test_a_finished_run_says_where_it_saved_and_only_paths_that_exist(tmp_path, env_file):
    run = tmp_path / "20261028T042323Z-bi-test-dry"
    run.mkdir()
    ds = _dataset(tmp_path / "ds_test_20261028_132329", 2)
    (run / "manifest.json").write_text('{"artifacts": {"dataset_root": "%s", "repo_id_stamped": "Orboh/ds_test_20261028_132329"}}' % ds)
    script = tmp_path / "fake.py"
    script.write_text(f"print('[record-bi] 記録: {run}')\n")
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, str(script)], {}), set_file=None)
    s.start(collect.Options(mode="dry", env_file=env_file, operator="Tester"))
    assert s.wait() == 0
    assert s.snapshot()["saved_to"] == {"run_dir": str(run), "dataset_root": str(ds), "episodes": 2}
    (run / "manifest.json").write_text('{"artifacts": {"dataset_root": "%s"}}' % (tmp_path / "never_made"))  # 10/26: ended before a dataset was made
    assert collect.saved_location(str(run)) == {"run_dir": str(run), "dataset_root": None, "episodes": None}
    assert collect.saved_location(None)["run_dir"] is None and collect.saved_location(str(tmp_path / "gone"))["run_dir"] is None


def test_a_full_width_tag_is_refused_before_the_set_file_is_touched(tmp_path, env_file):
    assert collect.tag_problem("s2") is None and collect.tag_problem("") is None
    assert collect.tag_problem("s１") == "col.err.tag.chars|s１" and collect.tag_problem("s 2") is not None
    set_file = tmp_path / "set.json"
    set_file.write_text('{"mode": "gate", "tag": "", "target": 10, "dataset_root": null, "repo_id": null}')
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, "-c", "pass"], {}), set_file=set_file)
    with pytest.raises(collect.CollectError, match="col.err.tag.chars"):
        s.start(collect.Options(mode="session", tag="s１", env_file=env_file, operator="Tester"))
    assert json.loads(set_file.read_text())["mode"] == "gate"  # 10/26: this record was overwritten and Resume was lost


def test_a_new_set_over_one_that_stopped_short_needs_a_confirmation(tmp_path, env_file):
    ds = _dataset(tmp_path / "ds", 7)
    set_file = tmp_path / "set.json"
    set_file.write_text('{"mode": "session", "tag": "s1", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % ds)
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, "-c", "pass"], {}), set_file=set_file)
    with pytest.raises(collect.CollectError, match=r"col.err.replace\|session s1: 7/10"):
        s.start(collect.Options(mode="session", tag="s2", env_file=env_file, operator="Tester"))
    assert json.loads(set_file.read_text())["tag"] == "s1"
    s.start(collect.Options(mode="dry", env_file=env_file, operator="Tester"))  # a dry run does not touch the set
    assert s.wait() == 0
    s.start(collect.Options(mode="session", tag="s2", env_file=env_file, operator="Tester", replace_set=True))
    assert s.wait() == 0 and json.loads(set_file.read_text())["tag"] == "s2"


def test_the_disk_stamp_changes_when_the_set_or_its_dataset_changes_outside_the_page(tmp_path):
    ds = _dataset(tmp_path / "ds", 7)
    set_file = tmp_path / "set.json"
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, set_file=set_file)
    assert s.disk_stamp() == (None, None)
    set_file.write_text('{"mode": "gate", "tag": "", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % ds)
    before = s.disk_stamp()
    assert None not in before
    info = ds / "meta" / "info.json"
    info.write_text('{"total_episodes": 10}')
    os.utime(info, ns=(before[1] + 10**9, before[1] + 10**9))  # a merge written a second later
    assert s.disk_stamp() != before


def test_branches_reads_each_checkout_and_compares_it_with_the_default(tmp_path):
    import subprocess

    repo = tmp_path / "orboh"
    subprocess.run(["git", "init", "-q", "-b", "feat/wood", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    rows = collect.branches({"openarm-orboh": (repo, "feat/wood"), "lerobot": (tmp_path / "missing", "feat/soft")})
    assert rows[0] == {"name": "openarm-orboh", "current": "feat/wood", "default": "feat/wood", "ok": True}
    assert rows[1]["current"] is None and rows[1]["ok"] is False
    subprocess.run(["git", "-C", str(repo), "switch", "-q", "--detach"], check=True)
    assert collect.current_branch(repo).startswith("HEAD (")


def test_a_new_set_over_a_finished_gate_waiting_for_its_ok_needs_a_confirmation(tmp_path, env_file):
    ds = _dataset(tmp_path / "ds_v1_20261028_123834", 10)
    set_file = tmp_path / "set.json"
    set_file.write_text('{"mode": "gate", "tag": "test", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % ds)
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, "-c", "pass"], {}), set_file=set_file)
    with pytest.raises(collect.CollectError, match=r"col.err.replace.review\|10/10"):  # 10/28 14:54: this gate was replaced without a word
        s.start(collect.Options(mode="gate", env_file=env_file, operator="Tester"))
    assert json.loads(set_file.read_text())["dataset_root"] == str(ds)
    s.mark_reviewed("Kota")  # once it has its OK, a new gate is an ordinary start
    s.start(collect.Options(mode="gate", env_file=env_file, operator="Tester"))
    assert s.wait() == 0


def test_the_confirmation_lets_a_new_gate_replace_one_waiting_for_its_ok(tmp_path, env_file):
    ds = _dataset(tmp_path / "ds", 10)
    set_file = tmp_path / "set.json"
    set_file.write_text('{"mode": "gate", "tag": "", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % ds)
    s = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, "-c", "pass"], {}), set_file=set_file)
    s.start(collect.Options(mode="gate", env_file=env_file, operator="Tester", replace_set=True))
    assert s.wait() == 0 and json.loads(set_file.read_text())["dataset_root"] is None


def test_the_default_branches_come_from_bootstrap_sh_or_its_last_known_values(tmp_path):
    boot = tmp_path / "bootstrap.sh"
    boot.write_text('OPENARM_BRANCH="${OPENARM_BRANCH:-dev/next}"\nLEROBOT_BRANCH="${LEROBOT_BRANCH:-feat/other}"\n')
    found = collect.default_branches(boot)
    assert found["openarm-orboh"][1] == "dev/next" and found["lerobot"][1] == "feat/other"
    missing = collect.default_branches(tmp_path / "gone.sh")
    assert missing["openarm-orboh"][1] == "dev/openarm-vla" and missing["lerobot"][1] == "feat/bi-openarm-soft-move-to"
