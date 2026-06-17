from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_primary_artifact_path_check_reports_missing(tmp_path):
    module = load_script_module(
        "verify_submission_snapshot_paths_unit",
        ROOT / "scripts" / "verify_submission_snapshot.py",
    )
    present = tmp_path / module.PRIMARY_ARTIFACT_PATHS[0]
    present.parent.mkdir(parents=True)
    present.write_text("fixture", encoding="utf-8")

    missing = module.missing_primary_artifacts(tmp_path)

    assert module.PRIMARY_ARTIFACT_PATHS[0] not in missing
    assert set(missing) == set(module.PRIMARY_ARTIFACT_PATHS[1:])


def test_verification_command_construction_uses_current_python():
    module = load_script_module(
        "verify_submission_snapshot_commands_unit",
        ROOT / "scripts" / "verify_submission_snapshot.py",
    )

    manifest = module.manifest_check_command(ROOT)
    focused = module.pytest_command(full_tests=False)
    full = module.pytest_command(full_tests=True)

    assert manifest == [sys.executable, str(ROOT / "scripts" / "write_artifact_manifest.py"), "--check"]
    assert focused[:3] == [sys.executable, "-m", "pytest"]
    assert module.FOCUSED_TEST_TARGETS[0] in focused
    assert focused[-3:] == ["-q", "-p", "no:cacheprovider"]
    assert full == [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
    assert module.git_diff_check_command() == ["git", "diff", "--check"]


def test_main_runs_read_only_verification_plan_without_recursive_pytest(monkeypatch, tmp_path):
    module = load_script_module(
        "verify_submission_snapshot_main_unit",
        ROOT / "scripts" / "verify_submission_snapshot.py",
    )
    for relative in module.PRIMARY_ARTIFACT_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run_command(command: list[str], *, root: Path) -> int:
        assert root == tmp_path.resolve()
        commands.append(command)
        return 0

    monkeypatch.setattr(module, "run_command", fake_run_command)

    exit_code = module.main(["--root", str(tmp_path)])

    assert exit_code == 0
    assert commands == [
        module.manifest_check_command(tmp_path.resolve()),
        module.pytest_command(full_tests=False),
        module.git_diff_check_command(),
    ]
