from __future__ import annotations

import json
import math
import platform
import hashlib
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cr3bp import propagate_controls_batch, propagate_feedback_batch
from .objective import ObjectiveConfig, outage_masks


def package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _project_root() -> Path:
    return Path.cwd()


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _project_manifest(root: Path) -> dict:
    manifest_roots = ["src", "scripts", "configs", "tests"]
    extra_files = ["README.md", "requirements.txt", ".gitignore"]
    hasher = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for name in manifest_roots:
        directory = root / name
        if not directory.exists():
            continue
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            try:
                content = path.read_bytes()
            except OSError:
                continue
            hasher.update(relative.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(hashlib.sha256(content).hexdigest().encode("ascii"))
            hasher.update(b"\0")
            file_count += 1
            total_bytes += len(content)
    for name in extra_files:
        path = root / name
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(content).hexdigest().encode("ascii"))
        hasher.update(b"\0")
        file_count += 1
        total_bytes += len(content)
    return {
        "project_manifest_roots": manifest_roots,
        "project_manifest_file_count": file_count,
        "project_manifest_total_bytes": total_bytes,
        "project_manifest_hash": hasher.hexdigest() if file_count else None,
    }


def revision_metadata() -> dict:
    root = _project_root()
    git_head = _git_output(["rev-parse", "HEAD"])
    git_status_short = _git_output(["status", "--short"])
    tracked_files = _git_output(["ls-files"])
    untracked_files = _git_output(["ls-files", "--others", "--exclude-standard"])
    return {
        "git_head": git_head,
        "git_status_short": git_status_short,
        "tracked_file_count": len(tracked_files.splitlines()) if tracked_files is not None else None,
        "untracked_file_count": len(untracked_files.splitlines()) if untracked_files is not None else None,
        **_project_manifest(root),
    }


def sanitize_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(sanitize_json(data), indent=2, allow_nan=False), encoding="utf-8")


def write_metadata(path: Path, command: str, config: dict, extra: dict | None = None) -> None:
    data = {
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(["numpy", "scipy", "matplotlib", "pandas", "pyyaml", "pytest"]),
        "config": config,
    }
    data.update(revision_metadata())
    if extra:
        data.update(extra)
    write_json(path, data)


def _summarize_label_values(values: pd.Series, *, ignore_empty: bool = False) -> str | None:
    clean = values.dropna()
    if ignore_empty:
        clean = clean[clean.astype(str).str.strip() != ""]
    unique = [str(v) for v in clean.unique()]
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else "mixed"


def _schedule_active_fraction(value) -> float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except (TypeError, ValueError):
        pass
    bits = str(value).strip()
    if not bits or any(bit not in {"0", "1"} for bit in bits):
        return float("nan")
    return float(sum(bit == "1" for bit in bits) / len(bits))


def _with_schedule_summary_columns(raw: pd.DataFrame) -> pd.DataFrame:
    prepared = raw.copy()
    if "schedule" in prepared.columns:
        if "refinement_candidate_schedule" not in prepared.columns:
            prepared["refinement_candidate_schedule"] = prepared["schedule"]
        if "refined_schedule" not in prepared.columns:
            prepared["refined_schedule"] = prepared["schedule"]
    for schedule_col, active_col in [
        ("refinement_candidate_schedule", "refinement_candidate_active_fraction"),
        ("refined_schedule", "refined_active_fraction"),
    ]:
        if active_col not in prepared.columns and schedule_col in prepared.columns:
            prepared[active_col] = prepared[schedule_col].map(_schedule_active_fraction)
    return prepared


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _with_schedule_summary_columns(raw)
    rows = []
    for method, group in raw.groupby("method", sort=False):
        row = {"method": method, "runs": int(len(group))}
        for col in [
            "benchmark_label",
            "target_mode",
            "comparison_group",
            "diagnostic_notes",
            "schedule",
            "refinement_candidate_schedule",
            "refined_schedule",
            "schedule_repair",
            "refined_schedule_source",
        ]:
            if col in group.columns:
                row[col] = _summarize_label_values(group[col], ignore_empty=(col == "diagnostic_notes"))
        for col in [
            "objective",
            "nominal_error",
            "worst_error",
            "robust_degradation",
            "active_fraction",
            "active_target_penalty",
            "active_target_deviation",
            "target_active_fraction",
            "target_active_weight",
            "refinement_candidate_active_fraction",
            "refined_active_fraction",
            "solver_true_evaluations",
            "shared_qubo_training_evaluations",
            "total_true_evaluations_including_training",
            "true_evaluations",
            "runtime_seconds",
            "refined_nominal_error",
            "refined_worst_error",
            "refined_selected_worst_error",
            "refined_all_mask_worst_error",
            "refinement_nfev",
            "refined_nominal_fuel",
            "refined_recovery_fuel_mean",
            "refined_recovery_fuel_max",
        ]:
            if col not in group.columns:
                continue
            vals = group[col].dropna().to_numpy(dtype=float)
            if vals.size:
                row[f"{col}_median"] = float(np.median(vals))
                row[f"{col}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        row["refinement_success_rate"] = float(group["refinement_success"].fillna(False).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _available_table_columns(summary: pd.DataFrame, specs: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    cols = [col for col, _ in specs if col in summary.columns]
    names = [name for col, name in specs if col in summary.columns]
    return cols, names


def write_latex_tables(summary: pd.DataFrame, qubo_diag: pd.DataFrame, tables_dir: Path) -> None:
    result_specs = [
        ("target_mode", "Benchmark"),
        ("method", "Method"),
        ("schedule_repair", "Schedule repair"),
        ("refined_schedule_source", "Refined schedule source"),
        ("comparison_group", "Comparison group"),
        ("schedule", "Solver best schedule"),
        ("refinement_candidate_schedule", "Refine candidate schedule"),
        ("refined_schedule", "Refined schedule"),
        ("refined_nominal_error_median", "Refined nominal error"),
        ("refined_selected_worst_error_median", "Selected recovery worst error"),
        ("refined_all_mask_worst_error_median", "All-mask diagnostic worst error"),
        ("active_fraction_median", "Solver best active fraction"),
        ("active_target_penalty_median", "Active target penalty"),
        ("refinement_candidate_active_fraction_median", "Refine candidate active fraction"),
        ("refined_active_fraction_median", "Refined active fraction"),
        ("solver_true_evaluations_median", "Solver true evals"),
        ("shared_qubo_training_evaluations_median", "Shared QUBO train evals"),
        ("total_true_evaluations_including_training_median", "Total true evals"),
        ("refinement_success_rate", "Refine success"),
    ]
    result_cols, result_names = _available_table_columns(summary, result_specs)
    table = summary[result_cols].copy()
    table.columns = result_names
    tables_dir.joinpath("results_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )
    ablation = qubo_diag.copy()
    tables_dir.joinpath("ablation_table.tex").write_text(
        ablation.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )
    recovery_specs = [
        ("method", "Method"),
        ("schedule_repair", "Schedule repair"),
        ("refined_schedule", "Refined schedule"),
        ("refined_schedule_source", "Refined schedule source"),
        ("refined_nominal_fuel_median", "Nominal fuel"),
        ("refined_recovery_fuel_mean_median", "Mean recovery fuel"),
        ("refined_recovery_fuel_max_median", "Max recovery fuel"),
        ("refinement_nfev_median", "Refinement nfev"),
        ("refinement_success_rate", "Recovery success"),
    ]
    recovery_cols, recovery_names = _available_table_columns(summary, recovery_specs)
    recovery = summary[recovery_cols].copy()
    recovery.columns = recovery_names
    tables_dir.joinpath("recovery_table.tex").write_text(
        recovery.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def _normal_comparison_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if "comparison_group" not in summary.columns:
        return summary
    normal = summary[summary["comparison_group"] != "oracle_diagnostic"].copy()
    return normal if len(normal) else summary


def savefig(path_base: Path) -> None:
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=220)
    plt.savefig(path_base.with_suffix(".pdf"))
    plt.close()


def plot_method_comparison(summary: pd.DataFrame, figures_dir: Path) -> None:
    summary = _normal_comparison_rows(summary)
    plt.figure(figsize=(8.0, 4.8))
    x = np.arange(len(summary))
    plt.bar(x - 0.18, summary["nominal_error_median"], width=0.36, label="Nominal")
    plt.bar(x + 0.18, summary["worst_error_median"], width=0.36, label="Worst outage")
    plt.xticks(x, summary["method"], rotation=25, ha="right")
    plt.ylabel("Final-state error")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.25)
    savefig(figures_dir / "method_comparison")


def plot_qubo_fit(fit_csvs: list[Path], figures_dir: Path) -> None:
    plt.figure(figsize=(5.8, 5.2))
    for path in fit_csvs:
        df = pd.read_csv(path)
        val = df[df["split"] == "val"]
        plt.scatter(val["true"], val["predicted"], s=18, alpha=0.7, label=path.stem.replace("qubo_fit_", ""))
    lo, hi = plt.xlim()
    ylo, yhi = plt.ylim()
    mn, mx = min(lo, ylo), max(hi, yhi)
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("True objective")
    plt.ylabel("QUBO surrogate prediction")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    savefig(figures_dir / "qubo_fit")


def plot_refinement_success(summary: pd.DataFrame, figures_dir: Path) -> None:
    summary = _normal_comparison_rows(summary)
    plt.figure(figsize=(8.0, 4.2))
    plt.bar(summary["method"], summary["refinement_success_rate"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Convergence success rate")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    savefig(figures_dir / "refinement_success")


def plot_recovery_fuel(summary: pd.DataFrame, figures_dir: Path) -> None:
    summary = _normal_comparison_rows(summary)
    plt.figure(figsize=(8.0, 4.4))
    x = np.arange(len(summary))
    plt.bar(x - 0.18, summary["refined_nominal_fuel_median"], width=0.36, label="Nominal")
    plt.bar(x + 0.18, summary["refined_recovery_fuel_max_median"], width=0.36, label="Max recovery")
    plt.xticks(x, summary["method"], rotation=25, ha="right")
    plt.ylabel("Integrated acceleration")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.25)
    savefig(figures_dir / "recovery_fuel")


def plot_trajectory_example(
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    schedule: np.ndarray,
    controls: np.ndarray | None,
    figures_dir: Path,
) -> None:
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    if controls is not None:
        nominal_final, nominal_hist = propagate_controls_batch(state0, controls, cfg.mu, cfg.tf, cfg.substeps, True)
        del nominal_final
        outage_controls = controls * (schedule * masks[0])[:, None]
        outage_final, outage_hist = propagate_controls_batch(state0, outage_controls, cfg.mu, cfg.tf, cfg.substeps, True)
        del outage_final
    else:
        _, nominal_hist = propagate_feedback_batch(
            state0, schedule, target, cfg.mu, cfg.tf, cfg.amax, cfg.kr, cfg.kv, cfg.substeps, True
        )
        _, outage_hist = propagate_feedback_batch(
            state0, schedule * masks[0], target, cfg.mu, cfg.tf, cfg.amax, cfg.kr, cfg.kv, cfg.substeps, True
        )
    nom = nominal_hist[:, 0, :]
    out = outage_hist[:, 0, :]
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(nom[:, 0], nom[:, 1], nom[:, 2], label="Nominal")
    ax.plot(out[:, 0], out[:, 1], out[:, 2], "--", label="One-block outage")
    ax.scatter([state0[0]], [state0[1]], [state0[2]], c="tab:green", s=45, label="Initial")
    ax.scatter([target[0]], [target[1]], [target[2]], c="tab:red", s=45, label="Target phase")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(frameon=False)
    savefig(figures_dir / "trajectory_example")
