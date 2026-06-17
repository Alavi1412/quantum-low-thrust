"""Deterministic evidence synthesis for the paper claim path.

This postprocessor reads recorded CSV/JSON artifacts and writes compact
cross-backend evidence tables. It does not launch trajectory optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

_CSV_FIELD_LIMIT = sys.maxsize
while True:
    try:
        csv.field_size_limit(_CSV_FIELD_LIMIT)
        break
    except OverflowError:
        _CSV_FIELD_LIMIT //= 10

DEFAULT_RESULTS_DIR = ROOT / "data" / "results" / "evidence_synthesis"
DEFAULT_TABLES_DIR = ROOT / "tables" / "evidence_synthesis"

CONFIGURED_THRESHOLDS = ("0.09", "0.17")
STRINGENT_THRESHOLDS = ("0.065", "0.10")
NEAR_TIGHT_THRESHOLDS = ("0.05", "0.10")
TIGHT_THRESHOLDS = ("0.05", "0.09")

THRESHOLD_SENSITIVITY_CSV = (
    ROOT / "data" / "results" / "phase_shift_cardinality_30seed" / "threshold_sensitivity.csv"
)
THRESHOLD_SENSITIVITY_METADATA = (
    ROOT / "data" / "results" / "phase_shift_cardinality_30seed" / "threshold_sensitivity_metadata.json"
)
CONTINUATION_CSV = (
    ROOT / "data" / "results" / "continuation_extension_suite" / "continuation_margin_suite.csv"
)
CONTINUATION_METADATA = (
    ROOT / "data" / "results" / "continuation_extension_suite" / "continuation_margin_suite_metadata.json"
)
DIRECT_COLLOCATION_CSV = (
    ROOT / "data" / "results" / "direct_collocation_baseline" / "direct_collocation_baseline.csv"
)
DIRECT_COLLOCATION_METADATA = (
    ROOT / "data" / "results" / "direct_collocation_baseline" / "direct_collocation_baseline_metadata.json"
)
INDEPENDENT_HS_CSV = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_continuation_baseline"
    / "independent_hs_continuation_baseline.csv"
)
INDEPENDENT_HS_METADATA = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_continuation_baseline"
    / "independent_hs_continuation_baseline_metadata.json"
)
INDEPENDENT_HS_ALL_CONFIGURED_CSV = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_all_configured_headroom"
    / "independent_hs_all_configured_headroom.csv"
)
INDEPENDENT_HS_ALL_CONFIGURED_METADATA = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_all_configured_headroom"
    / "independent_hs_all_configured_headroom_metadata.json"
)
INDEPENDENT_HS_BRANCH_REPLAY_CSV = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_branch_control_replay"
    / "independent_hs_branch_control_replay.csv"
)
INDEPENDENT_HS_BRANCH_REPLAY_METADATA = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_branch_control_replay"
    / "independent_hs_branch_control_replay_metadata.json"
)
INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_CSV = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_bicircular_phase_stress"
    / "independent_hs_bicircular_phase_stress.csv"
)
INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_METADATA = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_bicircular_phase_stress"
    / "independent_hs_bicircular_phase_stress_metadata.json"
)
INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_CSV = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_horizons_solar_tidal_replay"
    / "independent_hs_horizons_solar_tidal_replay.csv"
)
INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_METADATA = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_horizons_solar_tidal_replay"
    / "independent_hs_horizons_solar_tidal_replay_metadata.json"
)
INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_CSV = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_horizons_point_mass_retuning"
    / "independent_hs_horizons_point_mass_retuning.csv"
)
INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_METADATA = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_horizons_point_mass_retuning"
    / "independent_hs_horizons_point_mass_retuning_metadata.json"
)
TAIL_COAST_CSV = (
    ROOT / "data" / "results" / "hard_catalog_tail_coast_recovery" / "tail_coast_recovery.csv"
)
TAIL_COAST_METADATA = (
    ROOT / "data" / "results" / "hard_catalog_tail_coast_recovery" / "tail_coast_recovery_metadata.json"
)

SAMPLED_METHODS = [
    "random",
    "cross_entropy",
    "genetic",
    "true_sa",
    "surrogate_qubo_sa",
    "qaoa_statevector",
]

SYNTHESIS_COLUMNS = [
    "row_id",
    "artifact_family",
    "representative_case_or_statistic",
    "target_family",
    "backend_initializer_role",
    "mask_scope",
    "phase_time",
    "transfer_time",
    "amax",
    "segments",
    "nominal_error",
    "selected_worst_error",
    "all_mask_worst_error",
    "configured_pass",
    "stringent_0p065_0p10_all_mask_pass",
    "near_tight_0p05_0p10_all_mask_pass",
    "tight_0p05_0p09_all_mask_pass",
    "pass_status_note",
    "practitioner_interpretation",
    "source_artifact",
    "source_row_id",
]

TABLE_COLUMNS = [
    "Axis",
    "Representative evidence",
    "Role",
    "Scope",
    "Nom / Sel / All",
    "Configured",
    "Tighter or near-tight status",
    "Practitioner lesson",
]


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"source CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first_matching(rows: list[dict[str, str]], path: Path, **criteria: object) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            return row
    criteria_text = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
    raise RuntimeError(f"no row in {_relative_or_absolute(path)} matching {criteria_text}")


def _decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise RuntimeError(f"expected decimal value, got {value!r}") from exc


def _bool_text(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"true", "1", "yes"}:
        return "True"
    if text.lower() in {"false", "0", "no"}:
        return "False"
    return text


def _threshold_pass(
    nominal_error: str | None,
    diagnostic_error: str | None,
    thresholds: tuple[str, str],
) -> str:
    nominal = _decimal(nominal_error)
    diagnostic = _decimal(diagnostic_error)
    if nominal is None or diagnostic is None:
        return "n/a"
    return str(nominal <= Decimal(thresholds[0]) and diagnostic <= Decimal(thresholds[1]))


def _format_float_text(value: str | None, digits: int = 4) -> str:
    decimal = _decimal(value)
    if decimal is None:
        return ""
    return f"{float(decimal):.{digits}f}"


def _errors_for_table(row: dict[str, str]) -> str:
    nominal = _format_float_text(row.get("nominal_error"))
    selected = _format_float_text(row.get("selected_worst_error"))
    all_mask = _format_float_text(row.get("all_mask_worst_error"))
    if not any((nominal, selected, all_mask)):
        return "counts only"
    return f"{nominal} / {selected} / {all_mask}"


def independent_hs_branch_replay_available() -> bool:
    return INDEPENDENT_HS_BRANCH_REPLAY_CSV.is_file() and INDEPENDENT_HS_BRANCH_REPLAY_METADATA.is_file()


def independent_hs_bicircular_phase_stress_available() -> bool:
    return (
        INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_CSV.is_file()
        and INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_METADATA.is_file()
    )


def independent_hs_horizons_solar_tidal_replay_available() -> bool:
    return (
        INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_CSV.is_file()
        and INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_METADATA.is_file()
    )


def independent_hs_horizons_point_mass_retuning_available() -> bool:
    return (
        INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_CSV.is_file()
        and INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_METADATA.is_file()
    )


def _case_metric_row(
    *,
    row_id: str,
    artifact_family: str,
    representative_case_or_statistic: str,
    target_family: str,
    backend_initializer_role: str,
    mask_scope: str,
    source_artifact: Path,
    source_row_id: str,
    source_row: dict[str, str],
    practitioner_interpretation: str,
    pass_status_note: str,
) -> dict[str, str]:
    nominal = source_row.get("nominal_error", "")
    selected = source_row.get("selected_worst_error", "")
    all_mask = source_row.get("all_mask_worst_error", selected)
    return {
        "row_id": row_id,
        "artifact_family": artifact_family,
        "representative_case_or_statistic": representative_case_or_statistic,
        "target_family": target_family,
        "backend_initializer_role": backend_initializer_role,
        "mask_scope": mask_scope,
        "phase_time": source_row.get("phase_time", ""),
        "transfer_time": source_row.get("transfer_time", ""),
        "amax": source_row.get("amax", ""),
        "segments": source_row.get("segments", ""),
        "nominal_error": nominal,
        "selected_worst_error": selected,
        "all_mask_worst_error": all_mask,
        "configured_pass": _bool_text(source_row.get("meets_thresholds", "")),
        "stringent_0p065_0p10_all_mask_pass": _threshold_pass(nominal, all_mask, STRINGENT_THRESHOLDS),
        "near_tight_0p05_0p10_all_mask_pass": _threshold_pass(nominal, all_mask, NEAR_TIGHT_THRESHOLDS),
        "tight_0p05_0p09_all_mask_pass": _threshold_pass(nominal, all_mask, TIGHT_THRESHOLDS),
        "pass_status_note": pass_status_note,
        "practitioner_interpretation": practitioner_interpretation,
        "source_artifact": _relative_or_absolute(source_artifact),
        "source_row_id": source_row_id,
    }


def _independent_hs_branch_replay_row() -> dict[str, str]:
    metadata = json.loads(INDEPENDENT_HS_BRANCH_REPLAY_METADATA.read_text(encoding="utf-8"))
    summary = list(metadata["summary"])
    if int(metadata["branch_row_count"]) != 16:
        raise RuntimeError("expected independent-HS branch replay to contain 16 branch rows")
    original = next(
        row for row in summary if row["case_id"] == "ihs_all_single_p04_amax02_warm_from_p03"
    )
    polish = next(
        row for row in summary if row["case_id"] == "ihs_all_single_p04_amax02_polish_from_p04"
    )
    max_delta = str(metadata["max_terminal_error_delta"])
    return {
        "row_id": "ihs_branch_control_replay_p04_amax02",
        "artifact_family": "independent-HS branch-control replay",
        "representative_case_or_statistic": "original and polish p=0.4, amax=0.2 branch sidecars",
        "target_family": "catalog halo phase-shift",
        "backend_initializer_role": "deterministic replay of persisted endpoint-plus-midpoint controls",
        "mask_scope": "16 branch replay rows; 8/8 one-segment masks for original and polish rows",
        "phase_time": "0.4",
        "transfer_time": "0.5",
        "amax": "0.2",
        "segments": "8",
        "nominal_error": "",
        "selected_worst_error": str(original["recorded_selected_worst_error"]),
        "all_mask_worst_error": str(polish["recorded_all_mask_worst_error"]),
        "configured_pass": str(bool(metadata["passes_tolerance"])),
        "stringent_0p065_0p10_all_mask_pass": "replay delta only",
        "near_tight_0p05_0p10_all_mask_pass": "replay delta only",
        "tight_0p05_0p09_all_mask_pass": "replay delta only",
        "pass_status_note": (
            f"max replay delta={max_delta}; original branches={original['branch_row_count']}; "
            f"polish branches={polish['branch_row_count']}"
        ),
        "practitioner_interpretation": (
            "Persisted independent-HS branch controls replay exactly under normalized CR3BP; "
            "this strengthens recovery-side reproducibility without adding high-fidelity or production-solver evidence."
        ),
        "source_artifact": _relative_or_absolute(INDEPENDENT_HS_BRANCH_REPLAY_CSV),
        "source_row_id": "independent-HS branch-control replay metadata summary",
    }


def _independent_hs_bicircular_phase_stress_row() -> dict[str, str]:
    metadata = json.loads(INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_METADATA.read_text(encoding="utf-8"))
    polish = metadata["polish_case_summary"]
    if int(polish["nominal_phase_count"]) != 8:
        raise RuntimeError("expected independent-HS bicircular phase stress to contain 8 polish nominal phases")
    if int(polish["branch_phase_count"]) != 64:
        raise RuntimeError("expected independent-HS bicircular phase stress to contain 64 polish branch-phase rows")
    nominal_max = str(polish["max_nominal_bicircular_terminal_error"])
    branch_max = str(polish["max_branch_bicircular_terminal_error"])
    nominal_pass = int(polish["nominal_bicircular_pass_count"])
    branch_pass = int(polish["branch_bicircular_pass_count"])
    phases = ", ".join(str(value) for value in metadata["phase_degrees"])
    return {
        "row_id": "ihs_bicircular_phase_stress_polish_p04_amax02",
        "artifact_family": "independent-HS simple bicircular phase-sweep stress",
        "representative_case_or_statistic": "ihs_all_single_p04_amax02_polish_from_p04",
        "target_family": "catalog halo phase-shift",
        "backend_initializer_role": "persisted endpoint-plus-midpoint controls replayed with simple circular solar tide",
        "mask_scope": f"8/8 configured one-segment masks over Sun phases {phases}",
        "phase_time": "0.4",
        "transfer_time": "0.5",
        "amax": "0.2",
        "segments": "8",
        "nominal_error": nominal_max,
        "selected_worst_error": branch_max,
        "all_mask_worst_error": branch_max,
        "configured_pass": str(bool(nominal_pass == 8 and branch_pass == 64)),
        "stringent_0p065_0p10_all_mask_pass": _threshold_pass(nominal_max, branch_max, STRINGENT_THRESHOLDS),
        "near_tight_0p05_0p10_all_mask_pass": _threshold_pass(nominal_max, branch_max, NEAR_TIGHT_THRESHOLDS),
        "tight_0p05_0p09_all_mask_pass": _threshold_pass(nominal_max, branch_max, TIGHT_THRESHOLDS),
        "pass_status_note": (
            f"simple bicircular stress replay: nominal {nominal_pass}/8; branch {branch_pass}/64; "
            "no SPICE/high-fidelity or production parity claim"
        ),
        "practitioner_interpretation": (
            "The converged independent-HS all-configured row has a positive deterministic "
            "beyond-CR3BP stress replay under the simple circular solar-tidal model; the result "
            "is useful benchmark-resource evidence, not flight validation."
        ),
        "source_artifact": _relative_or_absolute(INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_CSV),
        "source_row_id": "polish_case_summary in independent-HS bicircular phase-stress metadata",
    }


def _independent_hs_horizons_solar_tidal_replay_row() -> dict[str, str]:
    metadata = json.loads(INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_METADATA.read_text(encoding="utf-8"))
    nominal_error = str(metadata["polish_nominal_horizons_solar_tidal_terminal_error"])
    branch_worst = str(metadata["polish_branch_horizons_solar_tidal_worst_error"])
    branch_pass = int(metadata["polish_branch_horizons_solar_tidal_pass_count"])
    branch_count = int(metadata["polish_branch_horizons_solar_tidal_row_count"])
    cr3bp_delta = str(metadata["cr3bp_max_replay_delta"])
    sun_range = metadata["sun_distance_lu_range"]
    if branch_count != 8:
        raise RuntimeError("expected independent-HS cached-Horizons replay to contain 8 polish branch rows")
    return {
        "row_id": "ihs_horizons_solar_tidal_replay_polish_p04_amax02",
        "artifact_family": "independent-HS cached-Horizons-derived solar-tidal replay",
        "representative_case_or_statistic": "ihs_all_single_p04_amax02_polish_from_p04",
        "target_family": "catalog halo phase-shift",
        "backend_initializer_role": "persisted endpoint-plus-midpoint controls replayed with cached JPL Horizons geometry",
        "mask_scope": "8/8 configured one-segment masks at representative 2026-Jan-01 epoch",
        "phase_time": "0.4",
        "transfer_time": "0.5",
        "amax": "0.2",
        "segments": "8",
        "nominal_error": nominal_error,
        "selected_worst_error": branch_worst,
        "all_mask_worst_error": branch_worst,
        "configured_pass": str(bool(float(nominal_error) <= 0.09 and branch_pass == branch_count)),
        "stringent_0p065_0p10_all_mask_pass": _threshold_pass(nominal_error, branch_worst, STRINGENT_THRESHOLDS),
        "near_tight_0p05_0p10_all_mask_pass": _threshold_pass(nominal_error, branch_worst, NEAR_TIGHT_THRESHOLDS),
        "tight_0p05_0p09_all_mask_pass": _threshold_pass(nominal_error, branch_worst, TIGHT_THRESHOLDS),
        "pass_status_note": (
            f"cached-Horizons stress replay: branch {branch_pass}/{branch_count}; "
            f"CR3BP replay delta={cr3bp_delta}; Sun LU {sun_range[0]}--{sun_range[1]}"
        ),
        "practitioner_interpretation": (
            "This is stronger than the simple bicircular phase sweep because the Sun geometry comes "
            "from cached JPL Horizons vectors, but it remains a simplified stress replay, not "
            "SPICE/high-fidelity/flight validation or production solver parity."
        ),
        "source_artifact": _relative_or_absolute(INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_CSV),
        "source_row_id": "polish_case_summary in independent-HS cached-Horizons replay metadata",
    }


def _independent_hs_horizons_point_mass_retuning_row() -> dict[str, str]:
    metadata = json.loads(INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_METADATA.read_text(encoding="utf-8"))
    polish = metadata["polish_case_summary"]
    replay_nominal = str(polish["persisted_nominal_point_mass_replay_error"])
    replay_branch = str(polish["persisted_branch_point_mass_replay_worst_error"])
    retuned_nominal = str(polish["retuned_nominal_point_mass_error"])
    retuned_branch = str(polish["retuned_branch_point_mass_worst_error"])
    branch_pass = int(polish["retuned_branch_pass_count"])
    branch_count = int(polish["branch_row_count"])
    if branch_count != 8:
        raise RuntimeError("expected independent-HS point-mass retuning to contain 8 polish branch rows")
    if bool(polish["persisted_nominal_replay_passes_configured_threshold"]):
        raise RuntimeError("point-mass retuning row should capture a failed persisted nominal replay")
    return {
        "row_id": "ihs_horizons_point_mass_retuning_polish_p04_amax02",
        "artifact_family": "independent-HS cached-Horizons Earth/Moon/Sun point-mass retuning",
        "representative_case_or_statistic": "ihs_all_single_p04_amax02_polish_from_p04",
        "target_family": "catalog halo phase-shift",
        "backend_initializer_role": "independent endpoint-plus-midpoint retuning under cached-Horizons point-mass dynamics",
        "mask_scope": "8/8 configured one-segment masks at representative 2026-Jan-01 epoch",
        "phase_time": "0.4",
        "transfer_time": "0.5",
        "amax": "0.2",
        "segments": "8",
        "nominal_error": retuned_nominal,
        "selected_worst_error": retuned_branch,
        "all_mask_worst_error": retuned_branch,
        "configured_pass": str(bool(float(retuned_nominal) <= 0.09 and branch_pass == branch_count)),
        "stringent_0p065_0p10_all_mask_pass": _threshold_pass(retuned_nominal, retuned_branch, STRINGENT_THRESHOLDS),
        "near_tight_0p05_0p10_all_mask_pass": _threshold_pass(retuned_nominal, retuned_branch, NEAR_TIGHT_THRESHOLDS),
        "tight_0p05_0p09_all_mask_pass": _threshold_pass(retuned_nominal, retuned_branch, TIGHT_THRESHOLDS),
        "pass_status_note": (
            f"persisted replay failed: nominal {replay_nominal}, branch worst {replay_branch}; "
            f"retuned branch {branch_pass}/{branch_count}"
        ),
        "practitioner_interpretation": (
            "Persisted controls fail direct ephemeris point-mass replay, but independent retuning "
            "restores feasibility for nominal and all 8 branches at the representative epoch; this is "
            "not SPICE/full high-fidelity/flight validation or production solver parity."
        ),
        "source_artifact": _relative_or_absolute(INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_CSV),
        "source_row_id": "polish_case_summary in independent-HS point-mass retuning metadata",
    }


def _threshold_count_row(threshold_rows: list[dict[str, str]]) -> dict[str, str]:
    tight_rows = [
        row
        for row in threshold_rows
        if row.get("threshold_id") == "continuous_dominance_0p05_0p09"
    ]
    by_method = {row["method"]: row for row in tight_rows}
    missing = [method for method in [*SAMPLED_METHODS, "all_windows_continuous"] if method not in by_method]
    if missing:
        raise RuntimeError(f"tight threshold-sensitivity rows missing methods: {missing}")
    sampled_counts = "; ".join(f"{method}={by_method[method]['success_count']}" for method in SAMPLED_METHODS)
    all_windows_count = by_method["all_windows_continuous"]["success_count"]
    if any(by_method[method]["success_count"] != "0/30" for method in SAMPLED_METHODS):
        raise RuntimeError("expected all sampled methods to be 0/30 at the tight threshold")
    if all_windows_count != "30/30":
        raise RuntimeError("expected all-windows continuous to be 30/30 at the tight threshold")
    return {
        "row_id": "phase_shift_tight_threshold_counts",
        "artifact_family": "30-seed threshold sensitivity",
        "representative_case_or_statistic": "tight (0.050,0.090) pass counts",
        "target_family": "catalog halo phase-shift, duty-cycle-prior N=12",
        "backend_initializer_role": "sampled binary initializers versus all-windows continuous",
        "mask_scope": "30 deterministic seeds; selected one-segment recovery masks",
        "phase_time": "0.3",
        "transfer_time": "0.5",
        "amax": "0.2",
        "segments": "12",
        "nominal_error": "",
        "selected_worst_error": "",
        "all_mask_worst_error": "",
        "configured_pass": "see source threshold table",
        "stringent_0p065_0p10_all_mask_pass": "source counts vary by method",
        "near_tight_0p05_0p10_all_mask_pass": "not tabulated in source",
        "tight_0p05_0p09_all_mask_pass": f"sampled methods 0/30; all-windows {all_windows_count}",
        "pass_status_note": f"{sampled_counts}; all_windows_continuous={all_windows_count}",
        "practitioner_interpretation": (
            "Tight screening exposes margin limits of sampled high-duty schedules; "
            "the positive result is the benchmark diagnostic, not sampler superiority."
        ),
        "source_artifact": _relative_or_absolute(THRESHOLD_SENSITIVITY_CSV),
        "source_row_id": "threshold_id=continuous_dominance_0p05_0p09",
    }


def build_synthesis() -> pd.DataFrame:
    threshold_rows = _read_csv_rows(THRESHOLD_SENSITIVITY_CSV)
    continuation_rows = _read_csv_rows(CONTINUATION_CSV)
    direct_rows = _read_csv_rows(DIRECT_COLLOCATION_CSV)
    ihs_rows = _read_csv_rows(INDEPENDENT_HS_CSV)
    ihs_all_configured_rows = _read_csv_rows(INDEPENDENT_HS_ALL_CONFIGURED_CSV)
    tail_rows = _read_csv_rows(TAIL_COAST_CSV)

    all_single_p04 = _first_matching(
        continuation_rows,
        CONTINUATION_CSV,
        case_id="all_single_p04_warm_from_p03",
    )
    two_segment_n8_p03 = _first_matching(
        continuation_rows,
        CONTINUATION_CSV,
        case_id="two_segment_n8_p03_cold",
    )
    direct_p04 = _first_matching(direct_rows, DIRECT_COLLOCATION_CSV, phase_time="0.4")
    ihs_tighter_thrust = _first_matching(
        ihs_rows,
        INDEPENDENT_HS_CSV,
        case_id="ihs_phase_p04_amax02_warm_from_p03",
    )
    ihs_all_configured = _first_matching(
        ihs_all_configured_rows,
        INDEPENDENT_HS_ALL_CONFIGURED_CSV,
        case_id="ihs_all_single_p04_amax02_warm_from_p03",
    )
    ihs_hard_failure = _first_matching(
        ihs_rows,
        INDEPENDENT_HS_CSV,
        case_id="ihs_catalog_dro_tf4_amax1_selected1",
    )
    tail_combined = _first_matching(
        tail_rows,
        TAIL_COAST_CSV,
        suite_case_id="tail_coast_all_one_two_segment_t5_portfolio",
    )

    rows = [
        _threshold_count_row(threshold_rows),
        _case_metric_row(
            row_id="continuation_all_single_p04",
            artifact_family="continuation-extension multiple shooting",
            representative_case_or_statistic="all_single_p04_warm_from_p03",
            target_family="catalog halo phase-shift",
            backend_initializer_role="all one-segment continuous continuation from p=0.3",
            mask_scope="8/8 configured one-segment masks selected and evaluated",
            source_artifact=CONTINUATION_CSV,
            source_row_id="case_id=all_single_p04_warm_from_p03",
            source_row=all_single_p04,
            practitioner_interpretation=(
                "All-mask continuation has headroom at p=0.4 under 0.065/0.10 "
                "screening, but the nominal error is above 0.05."
            ),
            pass_status_note="selected masks equal all configured masks for this row",
        ),
        _case_metric_row(
            row_id="continuation_two_segment_n8_p03",
            artifact_family="continuation-extension multiple shooting",
            representative_case_or_statistic="two_segment_n8_p03_cold",
            target_family="catalog halo phase-shift",
            backend_initializer_role="cold all one/two-mask continuous continuation",
            mask_scope="15/15 configured one- and two-segment masks selected and evaluated",
            source_artifact=CONTINUATION_CSV,
            source_row_id="case_id=two_segment_n8_p03_cold",
            source_row=two_segment_n8_p03,
            practitioner_interpretation=(
                "A broader one/two-mask phase-shift row passes 0.065/0.10 "
                "all-mask diagnostics, but not the 0.05 nominal headroom check."
            ),
            pass_status_note="selected masks equal all configured masks for this row",
        ),
        _case_metric_row(
            row_id="direct_collocation_selected_p04",
            artifact_family="direct-collocation baseline",
            representative_case_or_statistic="selected-branch p=0.4",
            target_family="catalog halo phase-shift",
            backend_initializer_role="compact trapezoidal selected-branch collocation",
            mask_scope="1 selected outage branch; 12 one-segment all-mask diagnostics",
            source_artifact=DIRECT_COLLOCATION_CSV,
            source_row_id="phase_time=0.4",
            source_row=direct_p04,
            practitioner_interpretation=(
                "Compact collocation can be tight on this selected branch, but "
                "unselected masks remain diagnostics and should be reported."
            ),
            pass_status_note="tight status uses the all-mask diagnostic, not only the optimized selected branch",
        ),
        _case_metric_row(
            row_id="ihs_tighter_thrust_p04",
            artifact_family="independent-midpoint Hermite-Simpson",
            representative_case_or_statistic="ihs_phase_p04_amax02_warm_from_p03",
            target_family="catalog halo phase-shift",
            backend_initializer_role="selected-branch independent-midpoint collocation with tighter thrust",
            mask_scope="3 selected outage branches; 8 one-segment all-mask diagnostics",
            source_artifact=INDEPENDENT_HS_CSV,
            source_row_id="case_id=ihs_phase_p04_amax02_warm_from_p03",
            source_row=ihs_tighter_thrust,
            practitioner_interpretation=(
                "Independent midpoint controls meet tight all-mask diagnostics on "
                "this phase-shift row even at amax=0.2."
            ),
            pass_status_note="tight status uses the all-mask diagnostic, not only the optimized selected branches",
        ),
        _case_metric_row(
            row_id="ihs_all_configured_headroom_p04_amax02",
            artifact_family="independent-midpoint Hermite-Simpson all-configured headroom",
            representative_case_or_statistic="ihs_all_single_p04_amax02_warm_from_p03",
            target_family="catalog halo phase-shift",
            backend_initializer_role="all one-segment independent-midpoint collocation with tighter thrust",
            mask_scope="8/8 configured one-segment masks selected and evaluated",
            source_artifact=INDEPENDENT_HS_ALL_CONFIGURED_CSV,
            source_row_id="case_id=ihs_all_single_p04_amax02_warm_from_p03",
            source_row=ihs_all_configured,
            practitioner_interpretation=(
                "The independent-midpoint HS headroom package upgrades the earlier "
                "selected-branch row to all-configured one-segment evidence at "
                "p=0.4 and amax=0.2, while retaining the max-nfev optimizer caveat."
            ),
            pass_status_note="selected masks equal all configured one-segment masks for this row",
        ),
        *([_independent_hs_branch_replay_row()] if independent_hs_branch_replay_available() else []),
        *(
            [_independent_hs_bicircular_phase_stress_row()]
            if independent_hs_bicircular_phase_stress_available()
            else []
        ),
        *(
            [_independent_hs_horizons_solar_tidal_replay_row()]
            if independent_hs_horizons_solar_tidal_replay_available()
            else []
        ),
        *(
            [_independent_hs_horizons_point_mass_retuning_row()]
            if independent_hs_horizons_point_mass_retuning_available()
            else []
        ),
        _case_metric_row(
            row_id="tail_coast_hard_catalog_all_one_two",
            artifact_family="hard-catalog tail-coast recovery",
            representative_case_or_statistic="tail_coast_all_one_two_segment_t5_portfolio",
            target_family="catalog DRO phase hard case",
            backend_initializer_role="locked-nominal fixed-final-time tail-coast branch portfolio",
            mask_scope="27/27 configured one- and two-segment masks selected and evaluated",
            source_artifact=TAIL_COAST_CSV,
            source_row_id="suite_case_id=tail_coast_all_one_two_segment_t5_portfolio",
            source_row=tail_combined,
            practitioner_interpretation=(
                "The hard-catalog pass is real but scoped to the tail-coast backend; "
                "it passes 0.05/0.10 and narrowly misses 0.05/0.09 because the "
                "selected/all worst error is 0.0936063931709301."
            ),
            pass_status_note=(
                "passes configured and 0.05/0.10; misses 0.05/0.09 on selected/all worst error"
            ),
        ),
        _case_metric_row(
            row_id="ihs_hard_catalog_selected_failure",
            artifact_family="independent-midpoint Hermite-Simpson",
            representative_case_or_statistic="ihs_catalog_dro_tf4_amax1_selected1",
            target_family="catalog DRO phase hard case",
            backend_initializer_role="bounded selected-outage independent-midpoint collocation negative diagnostic",
            mask_scope="1 selected outage branch; 15 one/two-mask all-mask diagnostics",
            source_artifact=INDEPENDENT_HS_CSV,
            source_row_id="case_id=ihs_catalog_dro_tf4_amax1_selected1",
            source_row=ihs_hard_failure,
            practitioner_interpretation=(
                "The hard-catalog result is not a generic continuous-backend pass; "
                "this independent-HS selected-outage diagnostic fails badly."
            ),
            pass_status_note="negative control for hard-catalog backend generality",
        ),
    ]
    return pd.DataFrame(rows, columns=SYNTHESIS_COLUMNS)


def _compact_table_frame(synthesis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in synthesis.to_dict(orient="records"):
        rows.append(
            {
                "Axis": str(row["artifact_family"]),
                "Representative evidence": str(row["representative_case_or_statistic"]),
                "Role": str(row["backend_initializer_role"]),
                "Scope": str(row["mask_scope"]),
                "Nom / Sel / All": _errors_for_table(row),
                "Configured": str(row["configured_pass"]),
                "Tighter or near-tight status": _table_pass_status(row),
                "Practitioner lesson": str(row["practitioner_interpretation"]),
            }
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _table_pass_status(row: dict[str, str]) -> str:
    if row["row_id"] == "phase_shift_tight_threshold_counts":
        return str(row["tight_0p05_0p09_all_mask_pass"])
    if row["row_id"] == "tail_coast_hard_catalog_all_one_two":
        return "0.05/0.10: True; 0.05/0.09: False"
    return (
        f"0.065/0.10: {row['stringent_0p065_0p10_all_mask_pass']}; "
        f"0.05/0.09: {row['tight_0p05_0p09_all_mask_pass']}"
    )


def write_latex_tables(synthesis: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "evidence_synthesis_table.tex"
    lessons_path = tables_dir / "practitioner_lessons_table.tex"

    compact = _compact_table_frame(synthesis)
    compact.to_latex(table_path, index=False, escape=True)

    lessons = synthesis[
        [
            "artifact_family",
            "representative_case_or_statistic",
            "practitioner_interpretation",
            "source_artifact",
        ]
    ].rename(
        columns={
            "artifact_family": "Axis",
            "representative_case_or_statistic": "Evidence",
            "practitioner_interpretation": "Practitioner lesson",
            "source_artifact": "Source artifact",
        }
    )
    lessons.to_latex(lessons_path, index=False, escape=True)
    return {
        "evidence_synthesis_table_tex": table_path,
        "practitioner_lessons_table_tex": lessons_path,
    }


def write_artifacts(
    *,
    results_dir: Path,
    tables_dir: Path,
    command: str,
) -> dict[str, object]:
    synthesis = build_synthesis()
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "evidence_synthesis.csv"
    metadata_path = results_dir / "evidence_synthesis_metadata.json"
    table_paths = write_latex_tables(synthesis, tables_dir)
    synthesis.to_csv(csv_path, index=False)

    input_artifacts = [
        THRESHOLD_SENSITIVITY_CSV,
        THRESHOLD_SENSITIVITY_METADATA,
        CONTINUATION_CSV,
        CONTINUATION_METADATA,
        DIRECT_COLLOCATION_CSV,
        DIRECT_COLLOCATION_METADATA,
        INDEPENDENT_HS_CSV,
        INDEPENDENT_HS_METADATA,
        INDEPENDENT_HS_ALL_CONFIGURED_CSV,
        INDEPENDENT_HS_ALL_CONFIGURED_METADATA,
        TAIL_COAST_CSV,
        TAIL_COAST_METADATA,
    ]
    if independent_hs_branch_replay_available():
        input_artifacts.extend([INDEPENDENT_HS_BRANCH_REPLAY_CSV, INDEPENDENT_HS_BRANCH_REPLAY_METADATA])
    if independent_hs_bicircular_phase_stress_available():
        input_artifacts.extend(
            [INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_CSV, INDEPENDENT_HS_BICIRCULAR_PHASE_STRESS_METADATA]
        )
    if independent_hs_horizons_solar_tidal_replay_available():
        input_artifacts.extend(
            [
                INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_CSV,
                INDEPENDENT_HS_HORIZONS_SOLAR_TIDAL_REPLAY_METADATA,
            ]
        )
    if independent_hs_horizons_point_mass_retuning_available():
        input_artifacts.extend(
            [
                INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_CSV,
                INDEPENDENT_HS_HORIZONS_POINT_MASS_RETUNING_METADATA,
            ]
        )
    metadata = {
        "command": command,
        "row_count": int(len(synthesis)),
        "optimization_rerun": False,
        "source_mode": "Recorded CSV/JSON artifacts only; no trajectory optimization is launched.",
        "determinism_note": (
            "Runtime is intentionally omitted so rerunning this deterministic "
            "postprocessor leaves metadata unchanged when inputs and command are unchanged."
        ),
        "thresholds": {
            "configured": {
                "nominal": float(CONFIGURED_THRESHOLDS[0]),
                "selected_or_all_diagnostic": float(CONFIGURED_THRESHOLDS[1]),
            },
            "stringent": {
                "nominal": float(STRINGENT_THRESHOLDS[0]),
                "selected_or_all_diagnostic": float(STRINGENT_THRESHOLDS[1]),
            },
            "near_tight": {
                "nominal": float(NEAR_TIGHT_THRESHOLDS[0]),
                "selected_or_all_diagnostic": float(NEAR_TIGHT_THRESHOLDS[1]),
            },
            "tight": {
                "nominal": float(TIGHT_THRESHOLDS[0]),
                "selected_or_all_diagnostic": float(TIGHT_THRESHOLDS[1]),
            },
        },
        "pass_semantics": (
            "Configured pass is copied from the source row where available. "
            "Derived stringent, near-tight, and tight pass columns use nominal_error "
            "and all_mask_worst_error as a conservative diagnostic; for rows that "
            "select all configured masks, selected_worst_error equals all_mask_worst_error."
        ),
        "row_order": synthesis["row_id"].to_list(),
        "input_artifacts": [
            {
                "path": _relative_or_absolute(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_artifacts
        ],
        "artifacts": {
            "evidence_synthesis_csv": _relative_or_absolute(csv_path),
            "evidence_synthesis_metadata_json": _relative_or_absolute(metadata_path),
            **{
                key: _relative_or_absolute(path)
                for key, path in table_paths.items()
            },
        },
        "interpretation_limits": [
            "This synthesis is an evidence-indexing layer over recorded artifacts, not a new experiment.",
            "Terminal-error thresholds are normalized screening tolerances, not flight targeting tolerances.",
            "Selected-branch collocation rows retain all-mask diagnostics as diagnostics unless all masks are explicitly selected.",
            "The hard-catalog positive row is scoped to the tail-coast locked-nominal fixed-final-time backend.",
            (
                "The independent-HS branch-control replay row, when present, is normalized-CR3BP "
                "persisted-control replay only; it does not rerun optimization or add high-fidelity validation."
            ),
            (
                "The independent-HS bicircular phase-stress row, when present, is a positive simple "
                "circular solar-tidal stress replay for the converged all-configured row; it is not "
                "SPICE/high-fidelity validation or production solver parity."
            ),
            (
                "The independent-HS cached-Horizons solar-tidal replay row, when present, uses cached "
                "JPL Horizons geometry in a simplified stress replay; it is not SPICE/high-fidelity/"
                "flight validation or production solver parity."
            ),
            (
                "The independent-HS cached-Horizons point-mass retuning row, when present, reports a "
                "failed persisted-control replay and an independent retuning pass; it is not SPICE/"
                "full high-fidelity/flight validation or production solver parity."
            ),
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "synthesis": synthesis,
        "metadata": metadata,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        **table_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize representative evidence rows from recorded artifacts "
            "without rerunning trajectory optimization."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    tables_dir = args.tables_dir if args.tables_dir.is_absolute() else ROOT / args.tables_dir
    artifacts = write_artifacts(
        results_dir=results_dir,
        tables_dir=tables_dir,
        command=" ".join(sys.argv),
    )
    metadata = artifacts["metadata"]
    print(
        "Completed evidence synthesis "
        f"with {metadata['row_count']} rows and no trajectory optimization rerun.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
