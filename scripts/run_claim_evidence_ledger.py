"""Build a reviewer-facing claim evidence ledger from recorded artifacts.

This postprocessor separates selected-branch evidence, all-mask diagnostics,
and rows where every configured mask was selected/evaluated. It reads existing
CSV/JSON artifacts only and does not launch trajectory optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
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

DEFAULT_RESULTS_DIR = ROOT / "data" / "results" / "claim_evidence_ledger"
DEFAULT_TABLES_DIR = ROOT / "tables" / "claim_evidence_ledger"

MAIN_SUMMARY_CSV = ROOT / "data" / "results" / "phase_shift_cardinality_30seed" / "summary.csv"
MAIN_STATS_METADATA = (
    ROOT / "data" / "results" / "phase_shift_cardinality_30seed" / "main_method_statistics_metadata.json"
)
MAIN_THRESHOLD_CSV = (
    ROOT / "data" / "results" / "phase_shift_cardinality_30seed" / "threshold_sensitivity.csv"
)
QAOA_SUMMARY_CSV = ROOT / "data" / "results" / "qaoa_depth_ablation_30seed" / "summary.csv"
QAOA_PAIRED_CSV = ROOT / "data" / "results" / "qaoa_depth_ablation_30seed" / "paired_comparisons.csv"
CONTINUATION_CSV = (
    ROOT / "data" / "results" / "continuation_extension_suite" / "continuation_margin_suite.csv"
)
DIRECT_COLLOCATION_CSV = (
    ROOT / "data" / "results" / "direct_collocation_baseline" / "direct_collocation_baseline.csv"
)
INDEPENDENT_HS_CSV = (
    ROOT
    / "data"
    / "results"
    / "independent_hs_continuation_baseline"
    / "independent_hs_continuation_baseline.csv"
)
TAIL_COAST_CSV = (
    ROOT / "data" / "results" / "hard_catalog_tail_coast_recovery" / "tail_coast_recovery.csv"
)
TAIL_COAST_BRANCH_REPLAY_SOURCE_CSV = (
    ROOT
    / "data"
    / "results"
    / "hard_catalog_tail_coast_branch_control_replay"
    / "tail_coast_recovery.csv"
)
TAIL_COAST_BRANCH_REPLAY_CSV = (
    ROOT
    / "data"
    / "results"
    / "hard_catalog_tail_coast_branch_control_replay"
    / "tail_coast_branch_control_replay.csv"
)
TAIL_COAST_BRANCH_REPLAY_METADATA = (
    ROOT
    / "data"
    / "results"
    / "hard_catalog_tail_coast_branch_control_replay"
    / "tail_coast_branch_control_replay_metadata.json"
)
BICIRCULAR_SOLAR_TIDAL_STRESS_CSV = (
    ROOT
    / "data"
    / "results"
    / "bicircular_solar_tidal_stress"
    / "bicircular_solar_tidal_stress.csv"
)
BICIRCULAR_SOLAR_TIDAL_STRESS_METADATA = (
    ROOT
    / "data"
    / "results"
    / "bicircular_solar_tidal_stress"
    / "bicircular_solar_tidal_stress_metadata.json"
)
BICIRCULAR_TAIL_COAST_RECOVERY_CSV = (
    ROOT
    / "data"
    / "results"
    / "bicircular_tail_coast_recovery"
    / "bicircular_tail_coast_recovery.csv"
)
BICIRCULAR_TAIL_COAST_RECOVERY_SUMMARY_CSV = (
    ROOT
    / "data"
    / "results"
    / "bicircular_tail_coast_recovery"
    / "bicircular_tail_coast_recovery_summary.csv"
)
BICIRCULAR_TAIL_COAST_RECOVERY_METADATA = (
    ROOT
    / "data"
    / "results"
    / "bicircular_tail_coast_recovery"
    / "bicircular_tail_coast_recovery_metadata.json"
)
HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_CSV = (
    ROOT
    / "data"
    / "results"
    / "horizons_ephemeris_force_model_contrast"
    / "horizons_ephemeris_force_model_contrast.csv"
)
HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_METADATA = (
    ROOT
    / "data"
    / "results"
    / "horizons_ephemeris_force_model_contrast"
    / "horizons_ephemeris_force_model_contrast_metadata.json"
)
DELAYED_RECOVERY_CSV = (
    ROOT / "data" / "results" / "hard_catalog_delayed_recovery" / "delayed_locked_recovery.csv"
)


def tail_coast_branch_control_replay_artifacts_available() -> bool:
    return all(
        path.is_file()
        for path in (
            TAIL_COAST_BRANCH_REPLAY_SOURCE_CSV,
            TAIL_COAST_BRANCH_REPLAY_CSV,
            TAIL_COAST_BRANCH_REPLAY_METADATA,
        )
    )


def bicircular_solar_tidal_stress_artifacts_available() -> bool:
    return all(
        path.is_file()
        for path in (
            BICIRCULAR_SOLAR_TIDAL_STRESS_CSV,
            BICIRCULAR_SOLAR_TIDAL_STRESS_METADATA,
        )
    )


def bicircular_tail_coast_recovery_artifacts_available() -> bool:
    return all(
        path.is_file()
        for path in (
            BICIRCULAR_TAIL_COAST_RECOVERY_CSV,
            BICIRCULAR_TAIL_COAST_RECOVERY_SUMMARY_CSV,
            BICIRCULAR_TAIL_COAST_RECOVERY_METADATA,
        )
    )


def horizons_ephemeris_force_model_contrast_artifacts_available() -> bool:
    return all(
        path.is_file()
        for path in (
            HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_CSV,
            HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_METADATA,
        )
    )


TAIL_COAST_COMBINED_CASE = "tail_coast_all_one_two_segment_t5_portfolio"

THRESHOLD_PAIRS = [
    ("configured_0p09_0p17", Decimal("0.09"), Decimal("0.17")),
    ("screen_0p05_0p12", Decimal("0.05"), Decimal("0.12")),
    ("near_margin_0p025_0p095", Decimal("0.025"), Decimal("0.095")),
    ("robust_tight_0p025_0p09", Decimal("0.025"), Decimal("0.09")),
    ("very_tight_0p02_0p09", Decimal("0.02"), Decimal("0.09")),
]

LEDGER_COLUMNS = [
    "claim_id",
    "evidence_family",
    "target_family",
    "target_mode",
    "source_case",
    "backend_or_method",
    "mask_scope",
    "selected_branch_semantics",
    "all_mask_semantics",
    "all_configured_mask_evidence",
    "nominal_error",
    "selected_worst_error",
    "all_mask_worst_error",
    "thresholds",
    "passes_configured_thresholds",
    "primary_interpretation",
    "explicit_boundary",
    "source_artifact",
]

THRESHOLD_AUDIT_COLUMNS = [
    "threshold_id",
    "suite_case_id",
    "target_mode",
    "nominal_threshold",
    "selected_or_all_worst_threshold",
    "nominal_error",
    "selected_worst_error",
    "all_mask_worst_error",
    "nominal_pass",
    "selected_worst_pass",
    "all_mask_worst_pass",
    "passes_threshold_pair",
    "audit_semantics",
    "source_artifact",
]

BRANCH_AUDIT_COLUMNS = [
    "suite_case_id",
    "target_mode",
    "total_branches",
    "optimizer_ran_count",
    "eligible_optimizer_branch_count",
    "optimizer_success_count",
    "optimizer_success_count_among_optimizer_ran",
    "no_recovery_direct_evaluation_count",
    "no_recovery_threshold_feasible_count",
    "fallback_accepted_count",
    "max_terminal_error_overall",
    "max_terminal_error_optimizer_ran",
    "max_terminal_error_no_recovery",
    "accepted_weight_variant_counts",
    "accepted_initialization_kind_counts",
    "accepted_initialization_label_counts",
    "branch_control_replay_claim",
    "audit_semantics",
    "source_artifact",
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


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"source JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _first_matching(rows: list[dict[str, str]], path: Path, **criteria: object) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            return row
    criteria_text = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
    raise RuntimeError(f"no row in {_relative_or_absolute(path)} matching {criteria_text}")


def _decimal(value: object) -> Decimal:
    text = str(value).strip()
    if not text:
        raise RuntimeError("expected decimal value, got blank")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise RuntimeError(f"expected decimal value, got {value!r}") from exc


def _float_text(value: object, digits: int = 6) -> str:
    return f"{float(_decimal(value)):.{digits}g}"


def _bool_text(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"true", "1", "yes"}:
        return "True"
    if text.lower() in {"false", "0", "no"}:
        return "False"
    return text


def _counter_text(counter: Counter[str]) -> str:
    return "; ".join(f"{key}={counter[key]}" for key in sorted(counter))


def _configured_counts(threshold_rows: list[dict[str, str]]) -> str:
    rows = [row for row in threshold_rows if row.get("threshold_id") == "configured_0p09_0p17"]
    if not rows:
        raise RuntimeError("missing configured threshold rows for main-method package")
    return "; ".join(f"{row['method']}={row['success_count']}" for row in rows)


def _tight_counts(threshold_rows: list[dict[str, str]]) -> str:
    rows = [row for row in threshold_rows if row.get("threshold_id") == "continuous_dominance_0p05_0p09"]
    by_method = {row["method"]: row["success_count"] for row in rows}
    required = {
        "random",
        "cross_entropy",
        "genetic",
        "true_sa",
        "surrogate_qubo_sa",
        "qaoa_statevector",
        "all_windows_continuous",
    }
    missing = required.difference(by_method)
    if missing:
        raise RuntimeError(f"missing tight threshold rows for methods: {sorted(missing)}")
    sampled = [method for method in required if method != "all_windows_continuous"]
    if any(by_method[method] != "0/30" for method in sampled):
        raise RuntimeError("expected all sampled methods to be 0/30 at tight threshold")
    if by_method["all_windows_continuous"] != "30/30":
        raise RuntimeError("expected all-windows continuous to be 30/30 at tight threshold")
    return "tight (0.05,0.09): sampled methods 0/30; all_windows_continuous=30/30"


def _summary_row(rows: list[dict[str, str]], method: str, path: Path) -> dict[str, str]:
    return _first_matching(rows, path, method=method)


def _metric_pair(method: str, row: dict[str, str]) -> str:
    return (
        f"{method} median nominal={_float_text(row['refined_nominal_error_median'])}, "
        f"selected={_float_text(row['refined_selected_worst_error_median'])}"
    )


def _main_method_row() -> dict[str, str]:
    summary_rows = _read_csv_rows(MAIN_SUMMARY_CSV)
    threshold_rows = _read_csv_rows(MAIN_THRESHOLD_CSV)
    metadata = _read_json(MAIN_STATS_METADATA)
    all_windows = _summary_row(summary_rows, "all_windows_continuous", MAIN_SUMMARY_CSV)
    qaoa = _summary_row(summary_rows, "qaoa_statevector", MAIN_SUMMARY_CSV)
    qubo = _summary_row(summary_rows, "surrogate_qubo_sa", MAIN_SUMMARY_CSV)
    comparisons = metadata["comparisons"]["all_windows_continuous"]  # type: ignore[index]
    sampled_methods = sorted(comparisons)
    sign_tests = {
        method: comparisons[method]["selected_worst_error_sign_test_p_two_sided"]  # type: ignore[index]
        for method in sampled_methods
    }
    if any(Decimal(str(value)) > Decimal("0.05") for value in sign_tests.values()):
        raise RuntimeError("expected all sampled-vs-all-windows selected-worst sign tests to be significant")
    return {
        "claim_id": "phase_shift_main_method_30seed_selected_branch",
        "evidence_family": "30-seed phase-shift main-method initializer comparison",
        "target_family": "halo phase-shift",
        "target_mode": str(all_windows["target_mode"]),
        "source_case": str(all_windows["benchmark_label"]),
        "backend_or_method": (
            "random, cross_entropy, genetic, true_sa, surrogate_qubo_sa, "
            "qaoa_statevector, all_windows_continuous"
        ),
        "mask_scope": "30 seeds; selected one-segment branch-recovery benchmark",
        "selected_branch_semantics": (
            "selected-worst errors come from the configured selected recovery branch benchmark"
        ),
        "all_mask_semantics": "all-mask values in this family are diagnostics, not selected all-configured masks",
        "all_configured_mask_evidence": "False",
        "nominal_error": "; ".join(
            [
                _metric_pair("all_windows_continuous", all_windows).split(", selected=")[0],
                _metric_pair("surrogate_qubo_sa", qubo).split(", selected=")[0],
                _metric_pair("qaoa_statevector", qaoa).split(", selected=")[0],
            ]
        ),
        "selected_worst_error": "; ".join(
            [
                f"all_windows_continuous median selected={_float_text(all_windows['refined_selected_worst_error_median'])}",
                f"surrogate_qubo_sa median selected={_float_text(qubo['refined_selected_worst_error_median'])}",
                f"qaoa_statevector median selected={_float_text(qaoa['refined_selected_worst_error_median'])}",
            ]
        ),
        "all_mask_worst_error": "diagnostic medians only; not all-configured-mask evidence",
        "thresholds": "configured (0.09,0.17); " + _tight_counts(threshold_rows),
        "passes_configured_thresholds": _configured_counts(threshold_rows),
        "primary_interpretation": (
            "No sampled-method or QAOA superiority; all-windows continuous has lower paired "
            "selected-worst error than every sampled method in this benchmark."
        ),
        "explicit_boundary": (
            "Selected-branch statistical evidence only; no quantum advantage and no all-configured-mask claim."
        ),
        "source_artifact": (
            f"{_relative_or_absolute(MAIN_SUMMARY_CSV)}; "
            f"{_relative_or_absolute(MAIN_STATS_METADATA)}; "
            f"{_relative_or_absolute(MAIN_THRESHOLD_CSV)}"
        ),
    }


def _qaoa_ablation_row() -> dict[str, str]:
    summary_rows = _read_csv_rows(QAOA_SUMMARY_CSV)
    paired_rows = _read_csv_rows(QAOA_PAIRED_CSV)
    qubo = _summary_row(summary_rows, "surrogate_qubo_sa", QAOA_SUMMARY_CSV)
    p2 = _summary_row(summary_rows, "qaoa_optimized_p2", QAOA_SUMMARY_CSV)
    paired = _first_matching(
        paired_rows,
        QAOA_PAIRED_CSV,
        baseline_method="surrogate_qubo_sa",
        method="qaoa_optimized_p2",
    )
    return {
        "claim_id": "phase_shift_qaoa_qubo_30seed_selected_branch",
        "evidence_family": "30-seed QAOA/QUBO ablation",
        "target_family": "halo phase-shift",
        "target_mode": str(p2["target_mode"]),
        "source_case": str(p2["benchmark_label"]),
        "backend_or_method": "surrogate-QUBO simulated annealing versus optimized statevector QAOA p=2",
        "mask_scope": "30 seeds; QUBO/QAOA sampler-family ablation on selected one-segment benchmark",
        "selected_branch_semantics": "paired selected-worst-error comparison against surrogate-QUBO SA",
        "all_mask_semantics": "not an all-configured-mask evaluation",
        "all_configured_mask_evidence": "False",
        "nominal_error": (
            f"surrogate_qubo_sa median={_float_text(qubo['refined_nominal_error_median'])}; "
            f"qaoa_optimized_p2 median={_float_text(p2['refined_nominal_error_median'])}"
        ),
        "selected_worst_error": (
            f"surrogate_qubo_sa median={_float_text(qubo['refined_selected_worst_error_median'])}; "
            f"qaoa_optimized_p2 median={_float_text(p2['refined_selected_worst_error_median'])}"
        ),
        "all_mask_worst_error": "diagnostic medians only; not all-configured-mask evidence",
        "thresholds": "configured (0.09,0.17)",
        "passes_configured_thresholds": (
            f"surrogate_qubo_sa={paired['baseline_successes']}/30; "
            f"qaoa_optimized_p2={paired['method_successes']}/30"
        ),
        "primary_interpretation": (
            "Optimized p=2 QAOA is competitive with surrogate-QUBO SA in this ablation, "
            "but paired tests do not support statistical superiority "
            f"(mean selected-worst delta={_float_text(paired['selected_worst_error_diff_mean'])}, "
            f"sign-test p={_float_text(paired['selected_worst_error_sign_test_p_two_sided'])})."
        ),
        "explicit_boundary": "Simulated QAOA ablation only; no hardware, quantum-advantage, or superiority claim.",
        "source_artifact": f"{_relative_or_absolute(QAOA_SUMMARY_CSV)}; {_relative_or_absolute(QAOA_PAIRED_CSV)}",
    }


def _case_metric_row(
    *,
    claim_id: str,
    evidence_family: str,
    target_family: str,
    source_case: str,
    backend_or_method: str,
    mask_scope: str,
    selected_branch_semantics: str,
    all_mask_semantics: str,
    all_configured_mask_evidence: bool,
    primary_interpretation: str,
    explicit_boundary: str,
    source_artifact: Path,
    source_row_key: str,
    source_row: dict[str, str],
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "evidence_family": evidence_family,
        "target_family": target_family,
        "target_mode": str(source_row.get("target_mode", "")),
        "source_case": source_case,
        "backend_or_method": backend_or_method,
        "mask_scope": mask_scope,
        "selected_branch_semantics": selected_branch_semantics,
        "all_mask_semantics": all_mask_semantics,
        "all_configured_mask_evidence": str(bool(all_configured_mask_evidence)),
        "nominal_error": str(source_row.get("nominal_error", "")),
        "selected_worst_error": str(source_row.get("selected_worst_error", "")),
        "all_mask_worst_error": str(source_row.get("all_mask_worst_error", "")),
        "thresholds": (
            f"nominal<={source_row.get('nominal_threshold', '')}; "
            f"selected/all<={source_row.get('selected_worst_threshold', '')}"
        ),
        "passes_configured_thresholds": _bool_text(source_row.get("meets_thresholds", "")),
        "primary_interpretation": primary_interpretation,
        "explicit_boundary": explicit_boundary,
        "source_artifact": f"{_relative_or_absolute(source_artifact)} ({source_row_key})",
    }


def _recorded_case_rows() -> list[dict[str, str]]:
    continuation_rows = _read_csv_rows(CONTINUATION_CSV)
    direct_rows = _read_csv_rows(DIRECT_COLLOCATION_CSV)
    ihs_rows = _read_csv_rows(INDEPENDENT_HS_CSV)
    tail_rows = _read_csv_rows(TAIL_COAST_CSV)
    delayed_rows = _read_csv_rows(DELAYED_RECOVERY_CSV)

    all_single = _first_matching(
        continuation_rows,
        CONTINUATION_CSV,
        case_id="all_single_p04_warm_from_p03",
    )
    two_segment = _first_matching(
        continuation_rows,
        CONTINUATION_CSV,
        case_id="two_segment_n8_p03_cold",
    )
    direct_p04 = _first_matching(direct_rows, DIRECT_COLLOCATION_CSV, phase_time="0.4")
    ihs_p04 = _first_matching(
        ihs_rows,
        INDEPENDENT_HS_CSV,
        case_id="ihs_phase_p04_amax02_warm_from_p03",
    )
    tail_combined = _first_matching(
        tail_rows,
        TAIL_COAST_CSV,
        suite_case_id=TAIL_COAST_COMBINED_CASE,
    )
    delayed_h6 = _first_matching(
        delayed_rows,
        DELAYED_RECOVERY_CSV,
        suite_case_id="delayed_hard_all_single_h6_portfolio",
    )

    return [
        _case_metric_row(
            claim_id="phase_shift_continuation_all_single_p04_all_configured",
            evidence_family="continuation extension",
            target_family="halo phase-shift",
            source_case="all_single_p04_warm_from_p03",
            backend_or_method="bounded multiple-shooting continuation",
            mask_scope="8/8 configured one-segment masks selected and evaluated",
            selected_branch_semantics="selected branches equal every configured one-segment mask",
            all_mask_semantics="all-mask diagnostic is the selected/evaluated all-configured one-segment set",
            all_configured_mask_evidence=True,
            primary_interpretation=(
                "All configured one-segment phase-shift masks pass in this continuation row."
            ),
            explicit_boundary=(
                "One-segment N=8 phase-shift scope only; not a QUBO/QAOA or high-fidelity claim."
            ),
            source_artifact=CONTINUATION_CSV,
            source_row_key="case_id=all_single_p04_warm_from_p03",
            source_row=all_single,
        ),
        _case_metric_row(
            claim_id="phase_shift_continuation_two_segment_n8_p03_all_configured",
            evidence_family="continuation extension",
            target_family="halo phase-shift",
            source_case="two_segment_n8_p03_cold",
            backend_or_method="bounded multiple-shooting continuation",
            mask_scope="15/15 configured one- and two-segment masks selected and evaluated",
            selected_branch_semantics="selected branches equal every configured one/two-segment mask",
            all_mask_semantics="all-mask diagnostic is the selected/evaluated all-configured one/two set",
            all_configured_mask_evidence=True,
            primary_interpretation=(
                "All configured N=8 one- and two-segment phase-shift masks pass in this continuation row."
            ),
            explicit_boundary=(
                "N=8 one/two-mask phase-shift scope only; not broader outage-family robustness."
            ),
            source_artifact=CONTINUATION_CSV,
            source_row_key="case_id=two_segment_n8_p03_cold",
            source_row=two_segment,
        ),
        _case_metric_row(
            claim_id="phase_shift_direct_collocation_p04_selected_branch_diagnostic",
            evidence_family="direct-collocation baseline",
            target_family="halo phase-shift",
            source_case="phase_time=0.4",
            backend_or_method="compact direct-collocation continuous backend",
            mask_scope="one selected branch optimized; all configured one-segment masks evaluated diagnostically",
            selected_branch_semantics=str(direct_p04.get("selected_branch_semantics", "")),
            all_mask_semantics=str(direct_p04.get("all_mask_diagnostic_semantics", "")),
            all_configured_mask_evidence=False,
            primary_interpretation=(
                "The selected p=0.4 branch is feasible and all-mask diagnostic is below configured thresholds."
            ),
            explicit_boundary=(
                "Selected-branch continuous-backend diagnostic; all masks were not selected for optimization."
            ),
            source_artifact=DIRECT_COLLOCATION_CSV,
            source_row_key="phase_time=0.4",
            source_row=direct_p04,
        ),
        _case_metric_row(
            claim_id="phase_shift_independent_hs_p04_amax02_selected_branch_diagnostic",
            evidence_family="independent-midpoint Hermite-Simpson baseline",
            target_family="halo phase-shift",
            source_case="ihs_phase_p04_amax02_warm_from_p03",
            backend_or_method="independent midpoint controls",
            mask_scope="3 selected branches optimized; 8 one-segment masks evaluated diagnostically",
            selected_branch_semantics=str(ihs_p04.get("selected_branch_semantics", "")),
            all_mask_semantics=str(ihs_p04.get("all_mask_diagnostic_semantics", "")),
            all_configured_mask_evidence=False,
            primary_interpretation=(
                "Independent midpoint controls give a strong phase-shift continuous-backend diagnostic."
            ),
            explicit_boundary=(
                "Selected-branch/all-mask distinction is preserved; not all-configured-mask evidence."
            ),
            source_artifact=INDEPENDENT_HS_CSV,
            source_row_key="case_id=ihs_phase_p04_amax02_warm_from_p03",
            source_row=ihs_p04,
        ),
        _case_metric_row(
            claim_id="catalog_dro_tail_coast_all_one_two_segment_t5_all_configured",
            evidence_family="hard-catalog tail-coast fixed-final-time recovery",
            target_family="catalog-DRO",
            source_case=TAIL_COAST_COMBINED_CASE,
            backend_or_method="locked-nominal fixed-final-time tail-coast branch portfolio",
            mask_scope="27/27 configured one- and two-segment masks selected and evaluated",
            selected_branch_semantics=str(tail_combined.get("selected_branch_semantics", "")),
            all_mask_semantics=str(tail_combined.get("all_mask_diagnostic_semantics", "")),
            all_configured_mask_evidence=True,
            primary_interpretation=(
                "The combined hard-catalog tail-coast row passes configured fixed-final-time thresholds "
                "for all configured one- and two-segment masks."
            ),
            explicit_boundary=(
                "Fixed-final-time tail-coast locked-nominal backend only; no high-fidelity, fuel-optimal, "
                "quantum, QUBO, or QAOA claim."
            ),
            source_artifact=TAIL_COAST_CSV,
            source_row_key=f"suite_case_id={TAIL_COAST_COMBINED_CASE}",
            source_row=tail_combined,
        ),
        _case_metric_row(
            claim_id="catalog_dro_delayed_h6_all_single_delayed_arrival",
            evidence_family="hard-catalog delayed-arrival recovery",
            target_family="catalog-DRO",
            source_case="delayed_hard_all_single_h6_portfolio",
            backend_or_method="locked-nominal delayed-arrival branch portfolio",
            mask_scope="14/14 configured one-segment masks selected and evaluated against the delayed target",
            selected_branch_semantics=str(delayed_h6.get("selected_branch_semantics", "")),
            all_mask_semantics=str(delayed_h6.get("all_mask_diagnostic_semantics", "")),
            all_configured_mask_evidence=True,
            primary_interpretation=(
                "The h6 portfolio recovers all one-segment masks against the delayed-arrival target."
            ),
            explicit_boundary=(
                "Delayed-arrival evidence only; not fixed-final-time robustness, not fuel optimality, "
                "and not quantum/QUBO/QAOA evidence."
            ),
            source_artifact=DELAYED_RECOVERY_CSV,
            source_row_key="suite_case_id=delayed_hard_all_single_h6_portfolio",
            source_row=delayed_h6,
        ),
    ]


def _tail_coast_branch_control_replay_ledger_row() -> dict[str, str]:
    source_rows = _read_csv_rows(TAIL_COAST_BRANCH_REPLAY_SOURCE_CSV)
    replay_rows = _read_csv_rows(TAIL_COAST_BRANCH_REPLAY_CSV)
    metadata = _read_json(TAIL_COAST_BRANCH_REPLAY_METADATA)
    source = _first_matching(
        source_rows,
        TAIL_COAST_BRANCH_REPLAY_SOURCE_CSV,
        suite_case_id=TAIL_COAST_COMBINED_CASE,
    )
    branch_rows = [row for row in replay_rows if row.get("record_type") == "branch"]
    nominal_rows = [row for row in replay_rows if row.get("record_type") == "nominal"]
    if len(nominal_rows) != 1:
        raise RuntimeError("expected exactly one nominal replay row for tail-coast branch-control replay")
    branch_row_count = int(metadata["branch_row_count"])
    if branch_row_count != len(branch_rows):
        raise RuntimeError("branch-control replay metadata branch_row_count does not match replay CSV")
    if branch_row_count <= 0:
        raise RuntimeError("branch-control replay requires at least one branch row")
    max_branch_delta = str(metadata["max_branch_terminal_error_delta"])
    max_delta = str(metadata["max_terminal_error_delta"])
    tolerance = str(metadata["tolerance"])
    passes = bool(metadata["passes_tolerance"])
    return {
        "claim_id": "catalog_dro_tail_coast_branch_control_replay_accepted_controls",
        "evidence_family": "hard-catalog tail-coast accepted branch-control replay",
        "target_family": "catalog-DRO",
        "target_mode": str(source["target_mode"]),
        "source_case": TAIL_COAST_COMBINED_CASE,
        "backend_or_method": "normalized CR3BP replay of persisted accepted full-control schedules",
        "mask_scope": f"{branch_row_count}/27 persisted accepted branch-control sidecars replayed",
        "selected_branch_semantics": (
            "replays only accepted nominal and selected branch full-control schedules persisted by the focused tail-coast run"
        ),
        "all_mask_semantics": (
            "the focused source row selected all configured one- and two-segment masks; replay does not rerun branch selection"
        ),
        "all_configured_mask_evidence": "True",
        "nominal_error": (
            f"recorded={nominal_rows[0]['recorded_terminal_error']}; "
            f"replayed={nominal_rows[0]['replay_terminal_error']}; "
            f"delta={nominal_rows[0]['terminal_error_delta']}"
        ),
        "selected_worst_error": (
            f"source selected_worst={source['selected_worst_error']}; "
            f"max branch replay delta={max_branch_delta}"
        ),
        "all_mask_worst_error": (
            f"source all_mask_worst={source['all_mask_worst_error']}; "
            "all configured masks were accepted-control replay rows, not new optimized branches"
        ),
        "thresholds": f"replay tolerance<={tolerance}; source nominal<={source['nominal_threshold']}; source selected/all<={source['selected_worst_threshold']}",
        "passes_configured_thresholds": str(passes),
        "primary_interpretation": (
            f"Persisted accepted controls replay deterministically under the normalized CR3BP model "
            f"with max terminal-error delta {max_delta} across nominal plus {branch_row_count} branch rows."
        ),
        "explicit_boundary": (
            "Accepted-control replay only; no optimization rerun, high-fidelity validation, production solver parity, "
            "fuel optimality, quantum, QUBO, or QAOA claim."
        ),
        "source_artifact": (
            f"{_relative_or_absolute(TAIL_COAST_BRANCH_REPLAY_CSV)}; "
            f"{_relative_or_absolute(TAIL_COAST_BRANCH_REPLAY_METADATA)}; "
            f"{_relative_or_absolute(TAIL_COAST_BRANCH_REPLAY_SOURCE_CSV)}"
        ),
    }


def _bicircular_solar_tidal_stress_ledger_row() -> dict[str, str]:
    metadata = _read_json(BICIRCULAR_SOLAR_TIDAL_STRESS_METADATA)
    rows = _read_csv_rows(BICIRCULAR_SOLAR_TIDAL_STRESS_CSV)
    if int(metadata["row_count"]) != len(rows):
        raise RuntimeError("bicircular stress metadata row_count does not match CSV")
    summary = metadata["solar_tidal_summary"]  # type: ignore[index]
    parameters = metadata["solar_tidal_parameters"]  # type: ignore[index]
    branch_pass_count = int(summary["branch_solar_tidal_pass_count"])  # type: ignore[index]
    branch_row_count = int(summary["branch_solar_tidal_row_count"])  # type: ignore[index]
    max_error = str(summary["max_solar_tidal_terminal_error"])  # type: ignore[index]
    max_delta = str(summary["max_solar_tidal_delta_from_cr3bp"])  # type: ignore[index]
    phase_degrees = ", ".join(str(value) for value in metadata["phase_degrees"])  # type: ignore[index]
    return {
        "claim_id": "catalog_dro_tail_coast_bicircular_solar_tidal_stress_probe",
        "evidence_family": "bicircular solar-tidal stress replay",
        "target_family": "catalog-DRO",
        "target_mode": "catalog_dro_phase",
        "source_case": TAIL_COAST_COMBINED_CASE,
        "backend_or_method": "persisted accepted controls replayed with a circular solar third-body tidal term",
        "mask_scope": (
            f"{branch_row_count} branch-phase stress rows from 27 accepted branch sidecars "
            f"and Sun phases {phase_degrees}"
        ),
        "selected_branch_semantics": (
            "replays persisted nominal and accepted branch full-control schedules; controls are not retuned"
        ),
        "all_mask_semantics": (
            "the source accepted-control package covers all configured one- and two-segment masks, "
            "but the solar-tidal stress rows are phase-sweep replays rather than new branch optimizations"
        ),
        "all_configured_mask_evidence": "False",
        "nominal_error": (
            "CR3BP replay max delta="
            f"{metadata['baseline_reproduction']['max_cr3bp_delta_from_source']}; "  # type: ignore[index]
            "solar-tidal nominal rows fail the configured nominal threshold in the generated CSV"
        ),
        "selected_worst_error": (
            f"solar-tidal branch pass count={branch_pass_count}/{branch_row_count}; "
            f"max solar-tidal terminal error={max_error}; max delta from CR3BP={max_delta}"
        ),
        "all_mask_worst_error": (
            "negative external-validity stress outcome; not an all-configured-mask robustness pass under solar tide"
        ),
        "thresholds": (
            "source nominal<=0.09; source branch<=0.17; "
            f"Sun distance={parameters['sun_distance_lu']} LU, mu ratio={parameters['sun_mu_ratio']}, "
            f"rotating phase rate={parameters['rotating_frame_phase_rate']}"
        ),
        "passes_configured_thresholds": str(bool(summary["all_branch_solar_tidal_rows_pass"])),  # type: ignore[index]
        "primary_interpretation": (
            "The persisted controls reproduce exactly in CR3BP but do not remain threshold-feasible "
            "under the configured bicircular solar-tidal stress sweep."
        ),
        "explicit_boundary": (
            "Beyond-CR3BP stress probe only; not SPICE ephemeris validation, high-fidelity flight validation, "
            "production solver parity, fuel optimality, quantum, QUBO, or QAOA evidence."
        ),
        "source_artifact": (
            f"{_relative_or_absolute(BICIRCULAR_SOLAR_TIDAL_STRESS_CSV)}; "
            f"{_relative_or_absolute(BICIRCULAR_SOLAR_TIDAL_STRESS_METADATA)}"
        ),
    }


def _bicircular_tail_coast_retuned_recovery_ledger_row() -> dict[str, str]:
    metadata = _read_json(BICIRCULAR_TAIL_COAST_RECOVERY_METADATA)
    rows = _read_csv_rows(BICIRCULAR_TAIL_COAST_RECOVERY_CSV)
    summary_rows = _read_csv_rows(BICIRCULAR_TAIL_COAST_RECOVERY_SUMMARY_CSV)
    if int(metadata["row_count"]) != len(rows):
        raise RuntimeError("bicircular retuned recovery metadata row_count does not match CSV")
    if int(metadata["summary_row_count"]) != len(summary_rows):
        raise RuntimeError("bicircular retuned recovery metadata summary_row_count does not match summary CSV")
    if bool(metadata["high_fidelity_validation"]):  # type: ignore[index]
        raise RuntimeError("bicircular retuned recovery metadata must not claim high_fidelity_validation")
    for flag_name in (
        "spice_ephemeris_validation",
        "production_solver_parity_claim",
        "fuel_optimality_claim",
        "quantum_advantage_claim",
    ):
        if bool(metadata[flag_name]):  # type: ignore[index]
            raise RuntimeError(f"bicircular retuned recovery metadata must not claim {flag_name}")
    if len(summary_rows) != 1:
        raise RuntimeError("bicircular retuned recovery summary CSV must contain exactly one row")
    summary = summary_rows[0]
    settings = metadata["settings"]  # type: ignore[index]
    branch_rows = int(summary["branch_row_count"])  # type: ignore[index]
    expected = int(summary["expected_branch_count"])  # type: ignore[index]
    complete = str(summary["package_complete"]).strip().lower() == "true"  # type: ignore[index]
    branch_pass = int(summary["branch_pass_count"])  # type: ignore[index]
    strict_branch_pass = int(summary["strict_branch_pass_count"])  # type: ignore[index]
    phase_degrees = ", ".join(str(value) for value in settings["phase_degrees"])  # type: ignore[index]
    all_configured = bool(complete and branch_rows == expected)
    nominal_rows = [row for row in rows if row.get("record_type") == "nominal"]
    if len(nominal_rows) != 1:
        raise RuntimeError("bicircular retuned recovery CSV must contain exactly one nominal row")
    nominal_row = nominal_rows[0]
    initial_nominal = nominal_row["initial_bicircular_terminal_error"]
    retuned_nominal = summary["nominal_error"]  # type: ignore[index]
    if not all_configured:
        raise RuntimeError("bicircular retuned recovery package is not complete; ledger row would be partial")
    return {
        "claim_id": "bicircular_tail_coast_retuned_recovery",
        "evidence_family": "bicircular tail-coast retuned recovery, negative",
        "target_family": "catalog-DRO",
        "target_mode": str(settings.get("target_mode", "catalog_dro_phase")),  # type: ignore[union-attr]
        "source_case": TAIL_COAST_COMBINED_CASE,
        "backend_or_method": "simple bicircular solar-tidal retuning from persisted CR3BP accepted controls",
        "mask_scope": (
            f"fixed Sun phase {phase_degrees} deg; all {branch_rows}/{expected} configured "
            f"one- and two-segment masks retuned; package_complete={complete}"
        ),
        "selected_branch_semantics": (
            "nominal and every configured branch row are retuned against the original fixed target "
            "and original fixed final time under the simple bicircular solar-tidal model"
        ),
        "all_mask_semantics": (
            "all configured one- and two-segment masks are covered, but the full package still fails "
            "the recorded configured and strict threshold checks"
        ),
        "all_configured_mask_evidence": str(all_configured),
        "nominal_error": (
            f"initial bicircular={initial_nominal}; retuned nominal={retuned_nominal}; "
            f"configured pass={_bool_text(summary['nominal_pass'])}"  # type: ignore[index]
        ),
        "selected_worst_error": (
            f"configured branch pass count={branch_pass}/{branch_rows}; "
            f"strict branch pass count={strict_branch_pass}/{branch_rows}; "
            f"max retuned branch error={summary['max_branch_error']}"  # type: ignore[index]
        ),
        "all_mask_worst_error": str(summary["all_mask_worst_error"]),  # type: ignore[index]
        "thresholds": (
            f"configured nominal<={metadata['thresholds']['configured_nominal_success']}; "  # type: ignore[index]
            f"configured branch<={metadata['thresholds']['configured_robust_success']}; "  # type: ignore[index]
            f"strict=({metadata['thresholds']['strict_nominal_success']}, "  # type: ignore[index]
            f"{metadata['thresholds']['strict_robust_success']}); "  # type: ignore[index]
            f"meets configured={_bool_text(summary['meets_thresholds'])}; "  # type: ignore[index]
            f"strict meets={_bool_text(summary['strict_meets_thresholds'])}"  # type: ignore[index]
        ),
        "passes_configured_thresholds": _bool_text(summary["meets_thresholds"]),  # type: ignore[index]
        "primary_interpretation": (
            f"Completed fixed-phase simple bicircular retuning lowers the nominal error from {initial_nominal} "
            f"to {retuned_nominal} but still fails: nominal configured pass={_bool_text(summary['nominal_pass'])}, "  # type: ignore[index]
            f"branch pass count={branch_pass}/{branch_rows}, max branch error={summary['max_branch_error']}, "  # type: ignore[index]
            f"strict branch pass count={strict_branch_pass}/{branch_rows}."
        ),
        "explicit_boundary": (
            f"Simple bicircular solar-tidal retuning at fixed phase {phase_degrees} deg only; original fixed "
            "target and final time; not SPICE/high-fidelity/flight validation, production solver parity, "
            "fuel optimality, quantum, QUBO, or QAOA evidence."
        ),
        "source_artifact": (
            f"{_relative_or_absolute(BICIRCULAR_TAIL_COAST_RECOVERY_CSV)}; "
            f"{_relative_or_absolute(BICIRCULAR_TAIL_COAST_RECOVERY_SUMMARY_CSV)}; "
            f"{_relative_or_absolute(BICIRCULAR_TAIL_COAST_RECOVERY_METADATA)}"
        ),
    }


def _horizons_ephemeris_force_model_contrast_ledger_row() -> dict[str, str]:
    metadata = _read_json(HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_METADATA)
    rows = _read_csv_rows(HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_CSV)
    if int(metadata["row_count"]) != len(rows):
        raise RuntimeError("Horizons contrast metadata row_count does not match CSV")
    if bool(metadata["high_fidelity_validation"]):  # type: ignore[index]
        raise RuntimeError("Horizons contrast metadata must not claim high_fidelity_validation")
    geometry = metadata["geometry_summary"]  # type: ignore[index]
    tidal = metadata["solar_tidal_acceleration_summary"]  # type: ignore[index]
    cache = metadata["cache"]  # type: ignore[index]
    representative = metadata["representative_states"]  # type: ignore[index]
    return {
        "claim_id": "catalog_dro_tail_coast_horizons_ephemeris_force_model_contrast",
        "evidence_family": "Horizons ephemeris force-model contrast",
        "target_family": "catalog-DRO",
        "target_mode": "catalog_dro_phase",
        "source_case": TAIL_COAST_COMBINED_CASE,
        "backend_or_method": (
            "cached JPL Horizons Earth/Moon/Sun geometry compared with simple bicircular assumptions"
        ),
        "mask_scope": (
            f"{metadata['row_count']} hard-catalog transfer nodes; accepted nominal controls and "
            f"representative branch mask {representative['branch_mask_index']} sampled for acceleration contrast"
        ),
        "selected_branch_semantics": (
            "accepted-control trajectory nodes are sampled under CR3BP only; no branch retuning or Horizons-dynamics replay"
        ),
        "all_mask_semantics": (
            "not an all-mask robustness test; the source accepted-control package remains the mask-scope artifact"
        ),
        "all_configured_mask_evidence": "False",
        "nominal_error": (
            "not a terminal-error replay; max nominal tidal-acceleration delta="
            f"{tidal['nominal_tidal_accel_delta_norm_max']}"
        ),
        "selected_worst_error": (
            f"representative branch mask={representative['branch_mask_index']}; "
            "max branch tidal-acceleration delta="
            f"{tidal['representative_branch_tidal_accel_delta_norm_max']}"
        ),
        "all_mask_worst_error": (
            "not computed; force-model contrast only and no all-configured-mask feasibility pass"
        ),
        "thresholds": (
            "no trajectory threshold tested; EM distance ratio range="
            f"{geometry['em_distance_ratio_min']}--{geometry['em_distance_ratio_max']}; "
            "EM angular-rate ratio range="
            f"{geometry['em_angular_rate_ratio_min']}--{geometry['em_angular_rate_ratio_max']}; "
            "Sun distance LU range="
            f"{geometry['sun_distance_lu_min']}--{geometry['sun_distance_lu_max']}; "
            f"fixed reference distance={geometry.get('reference_distance_km', 'not recorded')} km/LU"
        ),
        "passes_configured_thresholds": "False",
        "primary_interpretation": (
            "Cached Horizons geometry quantifies Earth-Moon distance/rate variation and fixed-reference "
            "Sun distance/phase differences missing from the simple circular model over the hard-catalog transfer nodes."
        ),
        "explicit_boundary": (
            "Ephemeris force-model contrast only; not SPICE validation, high-fidelity flight validation, "
            "accepted-control high-fidelity replay, production solver parity, fuel optimality, quantum, QUBO, or QAOA evidence."
        ),
        "source_artifact": (
            f"{_relative_or_absolute(HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_CSV)}; "
            f"{_relative_or_absolute(HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_METADATA)}; "
            f"{cache['path']}"
        ),
    }


def build_claim_evidence_ledger(
    include_branch_control_replay: bool | None = None,
    include_bicircular_solar_tidal_stress: bool | None = None,
    include_bicircular_tail_coast_recovery: bool | None = None,
    include_horizons_ephemeris_force_model_contrast: bool | None = None,
) -> pd.DataFrame:
    if include_branch_control_replay is None:
        include_branch_control_replay = tail_coast_branch_control_replay_artifacts_available()
    if include_bicircular_solar_tidal_stress is None:
        include_bicircular_solar_tidal_stress = bicircular_solar_tidal_stress_artifacts_available()
    if include_bicircular_tail_coast_recovery is None:
        include_bicircular_tail_coast_recovery = bicircular_tail_coast_recovery_artifacts_available()
    if include_horizons_ephemeris_force_model_contrast is None:
        include_horizons_ephemeris_force_model_contrast = (
            horizons_ephemeris_force_model_contrast_artifacts_available()
        )
    recorded_rows = _recorded_case_rows()
    rows = [
        _main_method_row(),
        _qaoa_ablation_row(),
        *recorded_rows[:5],
        *recorded_rows[5:],
    ]
    if include_branch_control_replay:
        rows.insert(7, _tail_coast_branch_control_replay_ledger_row())
    if include_bicircular_solar_tidal_stress:
        insert_at = 8 if include_branch_control_replay else 7
        rows.insert(insert_at, _bicircular_solar_tidal_stress_ledger_row())
    if include_bicircular_tail_coast_recovery:
        insert_at = 7 + int(include_branch_control_replay) + int(include_bicircular_solar_tidal_stress)
        rows.insert(insert_at, _bicircular_tail_coast_retuned_recovery_ledger_row())
    if include_horizons_ephemeris_force_model_contrast:
        insert_at = (
            7
            + int(include_branch_control_replay)
            + int(include_bicircular_solar_tidal_stress)
            + int(include_bicircular_tail_coast_recovery)
        )
        rows.insert(insert_at, _horizons_ephemeris_force_model_contrast_ledger_row())
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def _tail_coast_combined_row() -> dict[str, str]:
    return _first_matching(
        _read_csv_rows(TAIL_COAST_CSV),
        TAIL_COAST_CSV,
        suite_case_id=TAIL_COAST_COMBINED_CASE,
    )


def build_tail_coast_threshold_audit() -> pd.DataFrame:
    row = _tail_coast_combined_row()
    nominal = _decimal(row["nominal_error"])
    selected = _decimal(row["selected_worst_error"])
    all_mask = _decimal(row["all_mask_worst_error"])
    rows: list[dict[str, object]] = []
    for threshold_id, nominal_threshold, robust_threshold in THRESHOLD_PAIRS:
        nominal_pass = nominal <= nominal_threshold
        selected_pass = selected <= robust_threshold
        all_pass = all_mask <= robust_threshold
        rows.append(
            {
                "threshold_id": threshold_id,
                "suite_case_id": TAIL_COAST_COMBINED_CASE,
                "target_mode": row["target_mode"],
                "nominal_threshold": str(nominal_threshold),
                "selected_or_all_worst_threshold": str(robust_threshold),
                "nominal_error": row["nominal_error"],
                "selected_worst_error": row["selected_worst_error"],
                "all_mask_worst_error": row["all_mask_worst_error"],
                "nominal_pass": str(bool(nominal_pass)),
                "selected_worst_pass": str(bool(selected_pass)),
                "all_mask_worst_pass": str(bool(all_pass)),
                "passes_threshold_pair": str(bool(nominal_pass and selected_pass and all_pass)),
                "audit_semantics": (
                    "Recorded-error threshold audit only; no optimization rerun and no new trajectory solve."
                ),
                "source_artifact": _relative_or_absolute(TAIL_COAST_CSV),
            }
        )
    return pd.DataFrame(rows, columns=THRESHOLD_AUDIT_COLUMNS)


def build_tail_coast_branch_audit() -> pd.DataFrame:
    row = _tail_coast_combined_row()
    branches = json.loads(row["branch_results"])
    if not isinstance(branches, list):
        raise RuntimeError("branch_results must decode to a list")
    total = len(branches)
    optimizer_ran = [branch for branch in branches if bool(branch.get("optimizer_ran"))]
    no_recovery = [
        branch
        for branch in branches
        if str(branch.get("accepted_branch_initialization_kind")) == "no_recovery_variables"
    ]
    optimizer_success_count = sum(bool(branch.get("optimizer_success")) for branch in optimizer_ran)
    no_recovery_feasible = sum(bool(branch.get("no_recovery_variable_threshold_feasible")) for branch in no_recovery)
    fallback_accepted = sum(bool(branch.get("accepted_branch_initialization_is_fallback")) for branch in branches)
    terminal_errors = [_decimal(branch["terminal_error"]) for branch in branches]
    optimizer_terminal_errors = [_decimal(branch["terminal_error"]) for branch in optimizer_ran]
    no_recovery_terminal_errors = [_decimal(branch["terminal_error"]) for branch in no_recovery]
    weight_counts = Counter(str(branch.get("accepted_branch_weight_variant_label", "")) for branch in branches)
    kind_counts = Counter(str(branch.get("accepted_branch_initialization_kind", "")) for branch in branches)
    label_counts = Counter(str(branch.get("accepted_branch_initialization_label", "")) for branch in branches)

    audit_row = {
        "suite_case_id": TAIL_COAST_COMBINED_CASE,
        "target_mode": row["target_mode"],
        "total_branches": total,
        "optimizer_ran_count": len(optimizer_ran),
        "eligible_optimizer_branch_count": len(optimizer_ran),
        "optimizer_success_count": optimizer_success_count,
        "optimizer_success_count_among_optimizer_ran": optimizer_success_count,
        "no_recovery_direct_evaluation_count": len(no_recovery),
        "no_recovery_threshold_feasible_count": no_recovery_feasible,
        "fallback_accepted_count": fallback_accepted,
        "max_terminal_error_overall": str(max(terminal_errors)),
        "max_terminal_error_optimizer_ran": str(max(optimizer_terminal_errors)),
        "max_terminal_error_no_recovery": str(max(no_recovery_terminal_errors)),
        "accepted_weight_variant_counts": _counter_text(weight_counts),
        "accepted_initialization_kind_counts": _counter_text(kind_counts),
        "accepted_initialization_label_counts": _counter_text(label_counts),
        "branch_control_replay_claim": "False",
        "audit_semantics": (
            "Summary of recorded branch_results JSON from the historical four-row tail-coast package only; "
            "the focused branch-control replay package is reported as a separate ledger row."
        ),
        "source_artifact": _relative_or_absolute(TAIL_COAST_CSV),
    }
    return pd.DataFrame([audit_row], columns=BRANCH_AUDIT_COLUMNS)


def write_latex_tables(
    *,
    ledger: pd.DataFrame,
    threshold_audit: pd.DataFrame,
    branch_audit: pd.DataFrame,
    tables_dir: Path,
) -> dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = tables_dir / "claim_evidence_ledger_table.tex"
    threshold_path = tables_dir / "tail_coast_threshold_audit_table.tex"
    branch_path = tables_dir / "tail_coast_branch_audit_table.tex"

    ledger_table = ledger[
        [
            "claim_id",
            "evidence_family",
            "target_family",
            "mask_scope",
            "all_configured_mask_evidence",
            "primary_interpretation",
            "explicit_boundary",
        ]
    ].rename(
        columns={
            "claim_id": "Claim row",
            "evidence_family": "Evidence family",
            "target_family": "Target",
            "mask_scope": "Mask scope",
            "all_configured_mask_evidence": "All configured?",
            "primary_interpretation": "Interpretation",
            "explicit_boundary": "Boundary",
        }
    )
    ledger_table.to_latex(ledger_path, index=False, escape=True)

    threshold_table = threshold_audit[
        [
            "threshold_id",
            "nominal_threshold",
            "selected_or_all_worst_threshold",
            "nominal_pass",
            "selected_worst_pass",
            "all_mask_worst_pass",
            "passes_threshold_pair",
        ]
    ].rename(
        columns={
            "threshold_id": "Threshold pair",
            "nominal_threshold": "Nominal threshold",
            "selected_or_all_worst_threshold": "Selected/all threshold",
            "nominal_pass": "Nominal pass",
            "selected_worst_pass": "Selected pass",
            "all_mask_worst_pass": "All-mask pass",
            "passes_threshold_pair": "Pair pass",
        }
    )
    threshold_table.to_latex(threshold_path, index=False, escape=True)

    branch_table = branch_audit[
        [
            "suite_case_id",
            "total_branches",
            "optimizer_ran_count",
            "optimizer_success_count_among_optimizer_ran",
            "no_recovery_direct_evaluation_count",
            "fallback_accepted_count",
            "max_terminal_error_overall",
            "accepted_weight_variant_counts",
            "accepted_initialization_kind_counts",
        ]
    ].rename(
        columns={
            "suite_case_id": "Case",
            "total_branches": "Branches",
            "optimizer_ran_count": "Optimizer ran",
            "optimizer_success_count_among_optimizer_ran": "Optimizer success",
            "no_recovery_direct_evaluation_count": "Direct no-recovery",
            "fallback_accepted_count": "Fallback accepted",
            "max_terminal_error_overall": "Max terminal error",
            "accepted_weight_variant_counts": "Accepted weights",
            "accepted_initialization_kind_counts": "Accepted init kinds",
        }
    )
    branch_table.to_latex(branch_path, index=False, escape=True)
    return {
        "claim_evidence_ledger_table_tex": ledger_path,
        "tail_coast_threshold_audit_table_tex": threshold_path,
        "tail_coast_branch_audit_table_tex": branch_path,
    }


def _input_artifacts() -> list[Path]:
    paths = [
        MAIN_SUMMARY_CSV,
        MAIN_STATS_METADATA,
        MAIN_THRESHOLD_CSV,
        QAOA_SUMMARY_CSV,
        QAOA_PAIRED_CSV,
        CONTINUATION_CSV,
        DIRECT_COLLOCATION_CSV,
        INDEPENDENT_HS_CSV,
        TAIL_COAST_CSV,
        DELAYED_RECOVERY_CSV,
    ]
    if tail_coast_branch_control_replay_artifacts_available():
        paths.extend(
            [
                TAIL_COAST_BRANCH_REPLAY_SOURCE_CSV,
                TAIL_COAST_BRANCH_REPLAY_CSV,
                TAIL_COAST_BRANCH_REPLAY_METADATA,
            ]
        )
    if bicircular_solar_tidal_stress_artifacts_available():
        paths.extend(
            [
                BICIRCULAR_SOLAR_TIDAL_STRESS_CSV,
                BICIRCULAR_SOLAR_TIDAL_STRESS_METADATA,
            ]
        )
    if bicircular_tail_coast_recovery_artifacts_available():
        paths.extend(
            [
                BICIRCULAR_TAIL_COAST_RECOVERY_CSV,
                BICIRCULAR_TAIL_COAST_RECOVERY_SUMMARY_CSV,
                BICIRCULAR_TAIL_COAST_RECOVERY_METADATA,
            ]
        )
    if horizons_ephemeris_force_model_contrast_artifacts_available():
        paths.extend(
            [
                HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_CSV,
                HORIZONS_EPHEMERIS_FORCE_MODEL_CONTRAST_METADATA,
            ]
        )
    return paths


def write_artifacts(
    *,
    results_dir: Path,
    tables_dir: Path,
    command: str,
) -> dict[str, object]:
    branch_control_replay_available = tail_coast_branch_control_replay_artifacts_available()
    bicircular_stress_available = bicircular_solar_tidal_stress_artifacts_available()
    bicircular_retuned_available = bicircular_tail_coast_recovery_artifacts_available()
    horizons_contrast_available = horizons_ephemeris_force_model_contrast_artifacts_available()
    ledger = build_claim_evidence_ledger(
        include_branch_control_replay=branch_control_replay_available,
        include_bicircular_solar_tidal_stress=bicircular_stress_available,
        include_bicircular_tail_coast_recovery=bicircular_retuned_available,
        include_horizons_ephemeris_force_model_contrast=horizons_contrast_available,
    )
    threshold_audit = build_tail_coast_threshold_audit()
    branch_audit = build_tail_coast_branch_audit()

    results_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = results_dir / "claim_evidence_ledger.csv"
    metadata_path = results_dir / "claim_evidence_ledger_metadata.json"
    threshold_path = results_dir / "tail_coast_threshold_audit.csv"
    branch_path = results_dir / "tail_coast_branch_audit.csv"

    ledger.to_csv(ledger_path, index=False)
    threshold_audit.to_csv(threshold_path, index=False)
    branch_audit.to_csv(branch_path, index=False)
    table_paths = write_latex_tables(
        ledger=ledger,
        threshold_audit=threshold_audit,
        branch_audit=branch_audit,
        tables_dir=tables_dir,
    )

    metadata = {
        "command": command,
        "row_count": int(len(ledger)),
        "tail_coast_threshold_audit_row_count": int(len(threshold_audit)),
        "tail_coast_branch_audit_row_count": int(len(branch_audit)),
        "optimization_rerun": False,
        "uses_recorded_artifacts_only": True,
        "high_fidelity_claim": False,
        "branch_control_replay": branch_control_replay_available,
        "bicircular_solar_tidal_stress_probe": bicircular_stress_available,
        "bicircular_tail_coast_retuned_recovery": bicircular_retuned_available,
        "horizons_ephemeris_force_model_contrast": horizons_contrast_available,
        "fuel_optimality_claim": False,
        "quantum_advantage_claim": False,
        "source_mode": (
            "Recorded CSV/JSON artifacts only. The ledger, threshold audit, and branch audit "
            "are deterministic postprocessing outputs with no trajectory optimization rerun."
            + (
                " A branch-control replay ledger row is included because the focused replay package is present."
                if branch_control_replay_available
                else " No branch-control replay claim row is included because the focused replay package is absent."
            )
            + (
                " A bicircular solar-tidal stress row is included because its real stress-probe CSV and metadata exist."
                if bicircular_stress_available
                else " No bicircular solar-tidal stress row is included because that stress package is absent."
            )
            + (
                " A completed negative bicircular retuned recovery row is included because its real "
                "retuning CSV, summary CSV, and metadata exist."
                if bicircular_retuned_available
                else " No bicircular retuned recovery row is included because that package is absent."
            )
            + (
                " A Horizons ephemeris force-model contrast row is included because its real CSV and metadata exist."
                if horizons_contrast_available
                else " No Horizons ephemeris force-model contrast row is included because that package is absent."
            )
        ),
        "determinism_note": (
            "Runtime and wall-clock timestamps are intentionally omitted so identical inputs and "
            "command text produce byte-stable outputs."
        ),
        "claim_semantics": {
            "selected_branch_evidence": (
                "Rows where selected outage branches were optimized/evaluated and unselected masks, "
                "if present, remain diagnostics."
            ),
            "all_mask_diagnostic": (
                "All configured masks may be evaluated diagnostically without being selected for optimization."
            ),
            "all_configured_mask_evidence": (
                "True only when the row selected/evaluated every configured mask within the stated mask family."
            ),
        },
        "branch_control_replay_artifacts_available": branch_control_replay_available,
        "bicircular_solar_tidal_stress_artifacts_available": bicircular_stress_available,
        "bicircular_tail_coast_recovery_artifacts_available": bicircular_retuned_available,
        "horizons_ephemeris_force_model_contrast_artifacts_available": horizons_contrast_available,
        "tail_coast_threshold_pairs": [
            {
                "threshold_id": threshold_id,
                "nominal_threshold": float(nominal),
                "selected_or_all_worst_threshold": float(robust),
            }
            for threshold_id, nominal, robust in THRESHOLD_PAIRS
        ],
        "row_order": ledger["claim_id"].to_list(),
        "input_artifacts": [
            {
                "path": _relative_or_absolute(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in _input_artifacts()
        ],
        "artifacts": {
            "claim_evidence_ledger_csv": _relative_or_absolute(ledger_path),
            "claim_evidence_ledger_metadata_json": _relative_or_absolute(metadata_path),
            "tail_coast_threshold_audit_csv": _relative_or_absolute(threshold_path),
            "tail_coast_branch_audit_csv": _relative_or_absolute(branch_path),
            **{key: _relative_or_absolute(path) for key, path in table_paths.items()},
        },
        "interpretation_limits": [
            "This ledger clarifies evidence semantics; it does not add high-fidelity validation.",
            (
                "The historical tail-coast branch audit summarizes recorded branch_results JSON only; "
                "the focused replay row is included only when its real replay CSV and metadata exist."
            ),
            (
                "The bicircular solar-tidal row, when present, is a negative beyond-CR3BP stress replay "
                "and not high-fidelity validation or production solver parity."
            ),
            (
                "The bicircular retuned recovery row, when present, is a simple bicircular retuning stress "
                "experiment that reports a completed negative threshold result; it is not SPICE/high-fidelity "
                "validation, production solver parity, fuel optimality, or quantum evidence."
            ),
            (
                "The Horizons ephemeris contrast row, when present, is a cached force-model contrast only "
                "and not SPICE validation, high-fidelity replay, or a threshold-feasibility result."
            ),
            "The tail-coast positive row is fixed-final-time locked-nominal continuous-backend evidence only.",
            "The QAOA/QUBO rows are simulated initializer evidence and do not support quantum advantage or superiority claims.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "ledger": ledger,
        "threshold_audit": threshold_audit,
        "branch_audit": branch_audit,
        "metadata": metadata,
        "ledger_path": ledger_path,
        "metadata_path": metadata_path,
        "threshold_audit_path": threshold_path,
        "branch_audit_path": branch_path,
        **table_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build reviewer-facing claim evidence ledger artifacts from recorded CSV/JSON inputs only."
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
        "Completed claim evidence ledger "
        f"with {metadata['row_count']} ledger rows, "
        f"{metadata['tail_coast_threshold_audit_row_count']} threshold-audit rows, "
        f"and optimization_rerun={metadata['optimization_rerun']}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
