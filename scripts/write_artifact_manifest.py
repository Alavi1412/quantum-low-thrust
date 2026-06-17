from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "results" / "artifact_manifest.json"

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "logs",
}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".spl",
    ".synctex.gz",
    ".toc",
    ".bsp",
    ".tls",
    ".tpc",
}


def _git_output(root: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_paths(root: Path) -> list[Path] | None:
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    paths: list[Path] = []
    for item in raw.split(b"\0"):
        if item:
            paths.append(root / item.decode("utf-8", errors="replace"))
    return paths


def _fallback_paths(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_include(path: Path, root: Path, output: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    if path.resolve() == output.resolve():
        return False
    parts = set(relative.parts)
    if parts.intersection(EXCLUDED_DIRS):
        return False
    name = relative.name
    return not any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def collect_files(root: Path, output: Path) -> list[Path]:
    candidates = _git_paths(root)
    if candidates is None:
        candidates = _fallback_paths(root)
    files = [path for path in candidates if path.is_file() and should_include(path, root, output)]
    return sorted(files, key=lambda path: _relative(path, root))


def hash_file(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": _relative(path, root),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def build_manifest(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    generated_utc = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    files = [hash_file(path, root) for path in collect_files(root, output)]
    git_head = _git_output(root, ["rev-parse", "HEAD"])
    git_status = _git_output(root, ["status", "--short", "--untracked-files=all"])
    return {
        "manifest_version": 3,
        "generated_utc": generated_utc,
        "note": (
            "Scoped-file artifact manifest for paper reproducibility. File hashes and byte "
            "counts are authoritative for the working tree at generated_utc."
        ),
        "git_head_at_generation": git_head,
        "git_head_semantics": (
            "Records HEAD when this manifest was generated. If the manifest is committed, "
            "this value necessarily points to the commit before the final manifest commit; "
            "use file hashes for artifact identity."
        ),
        "working_tree_status_at_generation": git_status or "",
        "working_tree_status_semantics": (
            "Status is an audit snapshot only and may include manuscript, table, PDF, or "
            "manifest-refresh changes intentionally present during generation."
        ),
        "file_hash_semantics": (
            "Each listed sha256 and byte count is computed from the working-tree file content "
            "at generation time. The manifest file itself is excluded."
        ),
        "exclusions": {
            "directories": sorted(EXCLUDED_DIRS),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
            "explicit": [_relative(output, root) if output.is_relative_to(root) else str(output)],
        },
        "file_count": len(files),
        "files": files,
    }


def write_manifest(root: Path, output: Path) -> dict[str, object]:
    manifest = build_manifest(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def check_manifest(root: Path, output: Path) -> list[str]:
    if not output.is_file():
        return [f"manifest not found: {output}"]
    manifest = json.loads(output.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    problems: list[str] = []
    for row in files:
        rel = Path(str(row.get("path", "")))
        path = root / rel
        if path.resolve() == output.resolve():
            problems.append("manifest includes itself")
            continue
        if not path.is_file():
            problems.append(f"missing file: {rel.as_posix()}")
            continue
        current = hash_file(path, root)
        if current["sha256"] != row.get("sha256"):
            problems.append(f"sha256 mismatch: {rel.as_posix()}")
        if current["bytes"] != row.get("bytes"):
            problems.append(f"byte-count mismatch: {rel.as_posix()}")
    if int(manifest.get("file_count", -1)) != len(files):
        problems.append("file_count does not match files length")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write or verify the reproducibility artifact manifest.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Verify recorded file hashes instead of writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        problems = check_manifest(root, output.resolve())
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(f"manifest hash check passed: {output}")
        return 0
    manifest = write_manifest(root, output.resolve())
    print(f"wrote {output} with {manifest['file_count']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
