from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.delayed_recovery import normalize_branch_weight_variants
from qlt.direct_collocation import config_hash, file_identity, settings_fingerprint
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.locked_recovery import BranchRecoveryWeights, normalize_selected_outage_policy, selected_outage_count_for_policy
from qlt.objective import outage_masks
from qlt.reporting import sanitize_json, write_metadata
from qlt.tail_coast_recovery import (
    normalize_tail_coast_branch_initialization_fallbacks,
    run_tail_coast_recovery_baseline,
    tail_coast_branch_initialization_fallbacks_for_json,
)


TAIL_COAST_COLUMNS = [
    "suite_case_id",
    "purpose",
    "target_mode",
    "target_generation",
    "transfer_time",
    "amax",
    "segments",
    "substeps_per_segment",
    "outage_lengths",
    "outage_count",
    "selected_outage_policy",
    "selected_outages",
    "selected_outage_count",
    "tail_coast_segments",
    "optimized_nominal_segments",
    "nominal_max_nfev",
    "tail_nominal_max_nfev",
    "branch_max_nfev",
    "node_initialization",
    "node_initialization_blend",
    "terminal_weight",
    "control_weight",
    "smooth_weight",
    "continuity_weight",
    "xtol",
    "ftol",
    "gtol",
    "settings_fingerprint",
    "config_hash",
    "source_states_id",
    "nominal_control_path",
    "nominal_control_sha256",
    "branch_control_manifest_path",
    "branch_control_manifest_sha256",
    "branch_control_sidecar_count",
    "branch_control_replay_ready",
    "mode",
    "method_type",
    "nominal_seed_error",
    "nominal_tail_coast_error",
    "nominal_error",
    "nominal_baseline_error",
    "nominal_lock_error_delta",
    "nominal_tail_zero_max_abs",
    "nominal_tail_control_norm_max",
    "selected_recovery_worst_error",
    "selected_worst_error",
    "all_outage_worst_error",
    "all_mask_worst_error",
    "nominal_dt",
    "branch_total_duration",
    "branch_control_count",
    "original_target_state",
    "nominal_threshold",
    "selected_recovery_threshold",
    "selected_worst_threshold",
    "meets_nominal_threshold",
    "meets_selected_recovery_threshold",
    "meets_selected_worst_threshold",
    "meets_thresholds",
    "backend_success",
    "optimizer_success",
    "optimizer_success_semantics",
    "nominal_optimizer_success",
    "nominal_seed_optimizer_success",
    "nominal_tail_optimizer_success",
    "nominal_backend_success",
    "branch_optimizer_success",
    "branch_optimizer_all_success",
    "branch_optimizer_ran",
    "branch_portfolio_enabled",
    "branch_portfolio_variant_count",
    "branch_portfolio_variant_labels",
    "branch_portfolio_all_success",
    "branch_portfolio_all_success_semantics",
    "portfolio_acceptance_rule",
    "branch_portfolio_converged_threshold_feasible_candidate_counts",
    "branch_portfolio_candidate_counts",
    "branch_portfolio_candidate_optimizer_success_counts",
    "branch_portfolio_candidate_all_optimizer_success",
    "branch_portfolio_candidate_all_optimizer_success_by_branch",
    "branch_weight_variants",
    "branch_fallback_initialization_enabled",
    "branch_fallback_initialization_configured_count",
    "branch_fallback_initialization_labels",
    "branch_initialization_fallbacks",
    "branch_fallback_initialization_evaluated_counts",
    "branch_fallback_initialization_candidate_counts",
    "branch_fallback_initialization_any_evaluated",
    "branch_fallback_initialization_any_accepted",
    "branch_fallback_initialization_evaluated_branch_count",
    "branch_fallback_initialization_accepted_branch_count",
    "nominal_fuel",
    "recovery_fuel_mean",
    "recovery_fuel_max",
    "control_max_norm",
    "control_bound_violation",
    "nominal_seed_nfev",
    "nominal_tail_nfev",
    "nominal_nfev",
    "total_branch_nfev",
    "nfev",
    "nominal_seed_runtime_seconds",
    "nominal_tail_runtime_seconds",
    "nominal_runtime_seconds",
    "total_branch_runtime_seconds",
    "runtime_seconds",
    "cost",
    "optimality",
    "nominal_cost",
    "nominal_optimality",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "nominal_masked_outage_errors",
    "branch_nfev",
    "branch_runtime_seconds",
    "branch_optimizer_success_by_branch",
    "branch_optimizer_ran_by_branch",
    "branch_accepted_weight_variant_labels",
    "branch_accepted_weight_variant_indices",
    "branch_accepted_weights",
    "branch_accepted_initialization_labels",
    "branch_accepted_initialization_indices",
    "branch_accepted_initialization_is_fallback",
    "branch_accepted_initialization_kinds",
    "branch_accepted_variant_nfev",
    "branch_accepted_variant_runtime_seconds",
    "branch_recovery_starts",
    "branch_recovery_segments",
    "branch_control_counts",
    "branch_controls_remove_zero_nominal",
    "branch_results",
    "nominal_accepted_candidate",
    "backend_semantics",
    "selection_semantics",
    "selected_branch_semantics",
    "all_mask_diagnostic_semantics",
    "control_bound_semantics",
    "nominal_lock_semantics",
    "fixed_final_time_semantics",
    "worst_error_semantics",
    "message",
    "nominal_message",
    "nominal_seed_message",
]


def _json_list(value) -> str:
    return json.dumps(value, sort_keys=False)


def _json_bytes(data: object) -> bytes:
    text = json.dumps(sanitize_json(data), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _write_json_with_sha256(path: Path, data: object) -> str:
    payload = _json_bytes(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(path: Path) -> str:
    resolved = path.resolve()
    for base in (Path.cwd(), ROOT):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return str(path)


def _resolve_artifact_path(value: object) -> Path:
    text = str(value).strip()
    if not text:
        raise RuntimeError("expected artifact path, got blank")
    path = Path(text)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _safe_file_stem(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def _control_norm_diagnostics(controls: np.ndarray, amax: float) -> dict[str, object]:
    array = np.asarray(controls, dtype=float)
    if array.size == 0:
        norms = np.zeros(0, dtype=float)
    else:
        array = array.reshape((-1, 3))
        norms = np.linalg.norm(array, axis=1)
    max_norm = float(np.max(norms)) if norms.size else 0.0
    return {
        "control_norms": norms.astype(float).tolist(),
        "control_norm_min": float(np.min(norms)) if norms.size else 0.0,
        "control_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
        "control_norm_max": max_norm,
        "control_bound_violation": float(max(0.0, max_norm - float(amax))),
    }


def _float_or_default(value: object, default: float) -> float:
    if value is None:
        return float(default)
    try:
        if bool(pd.isna(value)):
            return float(default)
    except (TypeError, ValueError):
        pass
    return float(value)


def _branch_control_replay_limitations() -> list[str]:
    return [
        "Branch-control replay repropagates persisted accepted controls in the normalized CR3BP model only.",
        "Replay does not rerun least-squares optimization, branch portfolio selection, fallback search, or schedule search.",
        "Replay is not high-fidelity validation, production solver parity, fuel optimality evidence, or quantum advantage evidence.",
        "The evidence scope is the configured fixed-final-time target, outage masks, thresholds, and CR3BP integration settings.",
    ]


def _json_array(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _count_zero_recovery_branches(value) -> int:
    count = 0
    for item in _json_array(value):
        try:
            count += int(float(item)) == 0
        except (TypeError, ValueError):
            continue
    return int(count)


def _bool_list(value) -> list[bool]:
    out = []
    for item in _json_array(value):
        if isinstance(item, bool):
            out.append(item)
        else:
            out.append(str(item).strip().lower() in {"true", "1", "yes"})
    return out


def _readable_selected_outage_policy(value) -> str:
    text = str(value).strip()
    if text in {"", "0", "0.0"} or text.lower() == "nan":
        return "none"
    return text.replace("_", " ")


def _eligible_optimizer_converged_count(row) -> str:
    recovery_segments = _json_array(row.get("branch_recovery_segments"))
    success_by_branch = _bool_list(row.get("branch_optimizer_success_by_branch"))
    ran_by_branch = _bool_list(row.get("branch_optimizer_ran_by_branch"))

    eligible_indices: list[int] = []
    for index, segments in enumerate(recovery_segments):
        try:
            if int(float(segments)) > 0:
                eligible_indices.append(index)
        except (TypeError, ValueError):
            continue

    converged = 0
    for index in eligible_indices:
        success = index < len(success_by_branch) and bool(success_by_branch[index])
        ran = index < len(ran_by_branch) and bool(ran_by_branch[index])
        if success and ran:
            converged += 1
    return f"{converged}/{len(eligible_indices)}"


def _outage_count(segments: int, lengths: list[int]) -> int:
    return int(sum(max(0, int(segments) - int(length) + 1) for length in lengths))


def _implementation_identities() -> dict[str, str]:
    return {
        "tail_coast_recovery_module": file_identity(ROOT / "src" / "qlt" / "tail_coast_recovery.py"),
        "tail_coast_recovery_runner": file_identity(Path(__file__)),
        "locked_recovery_module": file_identity(ROOT / "src" / "qlt" / "locked_recovery.py"),
        "delayed_recovery_module": file_identity(ROOT / "src" / "qlt" / "delayed_recovery.py"),
        "cr3bp_module": file_identity(ROOT / "src" / "qlt" / "cr3bp.py"),
        "multiple_shooting_module": file_identity(ROOT / "src" / "qlt" / "multiple_shooting.py"),
        "objective_module": file_identity(ROOT / "src" / "qlt" / "objective.py"),
        "refinement_module": file_identity(ROOT / "src" / "qlt" / "refinement.py"),
    }


def _tail_config(config: dict) -> dict:
    return dict(config.get("tail_coast_recovery", {}) or {})


def _nominal_settings(config: dict) -> dict:
    return dict(_tail_config(config).get("nominal", {}) or {})


def _tail_nominal_settings(config: dict) -> dict:
    return dict(_tail_config(config).get("tail_nominal", {}) or {})


def _branch_settings(config: dict) -> dict:
    return dict(_tail_config(config).get("branch", {}) or {})


def _nominal_residual_weights(config: dict, case: dict | None = None) -> dict:
    raw = dict(_nominal_settings(config).get("residual_weights", {}) or {})
    if case is not None:
        raw.update(dict(case.get("nominal_residual_weights", {}) or {}))
    return {str(key): float(value) for key, value in raw.items()}


def _tail_nominal_weights(config: dict, case: dict | None = None) -> BranchRecoveryWeights:
    raw = copy.deepcopy(_tail_nominal_settings(config))
    if case is not None:
        merged = copy.deepcopy(raw)
        merged.update(copy.deepcopy(case.get("tail_nominal", {}) or {}))
        raw = merged
    return BranchRecoveryWeights.from_config(raw)


def _branch_config(config: dict, case: dict | None = None) -> dict:
    raw = copy.deepcopy(_branch_settings(config))
    if case is not None:
        merged = copy.deepcopy(raw)
        merged.update(copy.deepcopy(case.get("branch", {}) or {}))
        raw = merged
    return raw


def _branch_weights(config: dict, case: dict | None = None) -> BranchRecoveryWeights:
    return BranchRecoveryWeights.from_config(_branch_config(config, case))


def _branch_weight_variants(config: dict, case: dict | None = None) -> list[dict]:
    raw = _branch_config(config, case)
    variants = normalize_branch_weight_variants(
        branch_weights=_branch_weights(config, case),
        branch_weight_variants=raw.get("weight_variants"),
    )
    return [
        {"label": str(variant["label"]), "index": int(variant["index"]), "weights": variant["weights"].as_dict()}
        for variant in variants
    ]


def _branch_initialization_fallback_config(config: dict, case: dict | None = None) -> list[dict]:
    raw = _branch_config(config, case)
    value = raw.get("fallback_initializations", raw.get("initialization_fallbacks"))
    return copy.deepcopy(list(value or []))


def _branch_initialization_fallbacks(config: dict, case: dict | None = None) -> list[dict]:
    variants = normalize_tail_coast_branch_initialization_fallbacks(
        _branch_initialization_fallback_config(config, case)
    )
    return tail_coast_branch_initialization_fallbacks_for_json(variants)


def _tolerances(config: dict, case: dict | None = None) -> dict[str, float]:
    branch = _branch_settings(config)
    out = {
        "xtol": float(branch.get("xtol", 1e-5)),
        "ftol": float(branch.get("ftol", 1e-5)),
        "gtol": float(branch.get("gtol", 1e-5)),
    }
    if case is not None:
        merged_branch = dict(case.get("branch", {}) or {})
        for key in out:
            if key in case:
                out[key] = float(case[key])
            elif key in merged_branch:
                out[key] = float(merged_branch[key])
    return out


def _case_config(base_config: dict, case: dict) -> dict:
    config = copy.deepcopy(base_config)
    benchmark = config.setdefault("benchmark", {})
    benchmark["transfer_time"] = float(case["transfer_time"])
    benchmark["amax"] = float(case["amax"])
    benchmark["segments"] = int(case["segments"])
    config.setdefault("outages", {})["block_lengths"] = [int(value) for value in case["outage_lengths"]]
    return config


def _suite_cases(config: dict) -> list[dict]:
    tail_cfg = _tail_config(config)
    raw_cases = list(tail_cfg.get("cases", (config.get("suite", {}) or {}).get("cases", [])) or [])
    benchmark = config.get("benchmark", {}) or {}
    default_lengths = [int(value) for value in config.get("outages", {}).get("block_lengths", [1])]
    nominal = _nominal_settings(config)
    tail_nominal = _tail_nominal_settings(config)
    branch = _branch_settings(config)
    default_tail = int(tail_cfg.get("tail_coast_segments", 0))
    cases: list[dict] = []
    for index, raw in enumerate(raw_cases):
        if not bool(raw.get("enabled", True)):
            continue
        case = dict(raw)
        lengths = [int(value) for value in case.get("outage_lengths", case.get("block_lengths", default_lengths))]
        segments = int(case.get("segments", benchmark.get("segments")))
        tail = int(case.get("tail_coast_segments", default_tail))
        if tail < 0 or tail > segments:
            raise ValueError(f"tail_coast_segments must be in [0, segments] for {case.get('case_id', index)}")
        masks = outage_masks(segments, tuple(lengths))
        policy = normalize_selected_outage_policy(case.get("selected_outages", 0))
        selected_count = selected_outage_count_for_policy(policy, masks)
        outage_count = _outage_count(segments, lengths)
        cases.append(
            {
                "suite_case_id": str(case.get("case_id") or f"tail_coast_case_{index:03d}"),
                "purpose": str(case.get("purpose", "fixed-final-time tail-coast recovery case")),
                "transfer_time": float(case.get("transfer_time", benchmark.get("transfer_time"))),
                "amax": float(case.get("amax", benchmark.get("amax"))),
                "segments": segments,
                "outage_lengths": lengths,
                "outage_count": outage_count,
                "selected_outage_policy": policy.label,
                "selected_outages_raw": case.get("selected_outages", 0),
                "selected_outage_count": selected_count,
                "tail_coast_segments": tail,
                "optimized_nominal_segments": segments - tail,
                "nominal_max_nfev": int(case.get("nominal_max_nfev", nominal.get("max_nfev", 140))),
                "tail_nominal_max_nfev": int(case.get("tail_nominal_max_nfev", tail_nominal.get("max_nfev", nominal.get("max_nfev", 140)))),
                "branch_max_nfev": int(case.get("branch_max_nfev", branch.get("max_nfev", 120))),
                "node_initialization": str(case.get("node_initialization", nominal.get("node_initialization", "linear"))),
                "node_initialization_blend": float(case.get("node_initialization_blend", nominal.get("node_initialization_blend", 0.5))),
                "case_order": index,
                "case_raw": case,
            }
        )
    seen = set()
    for case in cases:
        if case["suite_case_id"] in seen:
            raise ValueError(f"duplicate tail-coast recovery case_id: {case['suite_case_id']}")
        seen.add(case["suite_case_id"])
    return cases


def _case_payload(case: dict) -> dict:
    return {
        "suite_case_id": str(case["suite_case_id"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "outage_lengths": [int(value) for value in case["outage_lengths"]],
        "outage_count": int(case["outage_count"]),
        "selected_outage_policy": str(case["selected_outage_policy"]),
        "selected_outages_raw": str(case["selected_outages_raw"]),
        "selected_outage_count": int(case["selected_outage_count"]),
        "tail_coast_segments": int(case["tail_coast_segments"]),
        "optimized_nominal_segments": int(case["optimized_nominal_segments"]),
        "nominal_max_nfev": int(case["nominal_max_nfev"]),
        "tail_nominal_max_nfev": int(case["tail_nominal_max_nfev"]),
        "branch_max_nfev": int(case["branch_max_nfev"]),
        "node_initialization": str(case["node_initialization"]),
        "node_initialization_blend": float(case["node_initialization_blend"]),
    }


def _effective_settings(config: dict, args, case: dict) -> dict:
    case_config = _case_config(config, case)
    return {
        "suite": "tail_coast_recovery",
        "case": _case_payload(case),
        "nominal_residual_weights": _nominal_residual_weights(config, case["case_raw"]),
        "tail_nominal_weights": _tail_nominal_weights(config, case["case_raw"]).as_dict(),
        "branch_weights": _branch_weights(config, case["case_raw"]).as_dict(),
        "branch_weight_variants": _branch_weight_variants(config, case["case_raw"]),
        "branch_initialization_fallbacks": _branch_initialization_fallbacks(config, case["case_raw"]),
        "tolerances": _tolerances(config, case["case_raw"]),
        "thresholds": copy.deepcopy(case_config["objective"]["thresholds"]),
        "benchmark": {
            "target_mode": str(case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
            "substeps_per_segment": int(case_config["benchmark"]["substeps_per_segment"]),
        },
        "config_hash": config_hash(case_config),
        "source_states_id": file_identity(args.source_states),
        "implementation_identities": _implementation_identities(),
    }


def _empty_sidecar_columns() -> dict[str, object]:
    return {
        "nominal_control_path": None,
        "nominal_control_sha256": None,
        "branch_control_manifest_path": None,
        "branch_control_manifest_sha256": None,
        "branch_control_sidecar_count": 0,
        "branch_control_replay_ready": False,
    }


def _branch_control_records(result: dict) -> list[dict]:
    records = result.get("accepted_branch_control_results")
    if records is None:
        records = result.get("accepted_branch_controls")
    if records is None:
        records = []
    return [dict(record) for record in list(records)]


class BranchControlSidecarStore:
    def __init__(
        self,
        *,
        results_dir: Path,
        case: dict,
        case_config: dict,
        states,
        cfg,
        masks: np.ndarray,
        expected: dict,
    ) -> None:
        self.results_dir = results_dir
        self.case = case
        self.case_config = case_config
        self.states = states
        self.cfg = cfg
        self.masks = np.asarray(masks)
        self.expected = expected
        self.case_id = str(case["suite_case_id"])
        self.safe_case_id = _safe_file_stem(self.case_id)
        self.controls_dir = results_dir / "controls"
        self.nominal_path = self.controls_dir / f"{self.safe_case_id}_nominal_controls.json"
        self.manifest_path = self.controls_dir / f"{self.safe_case_id}_branch_control_manifest.json"
        self.progress_csv_path = self.controls_dir / f"{self.safe_case_id}_branch_control_progress.csv"
        self.nominal_sha: str | None = None
        self.branch_entries: list[dict[str, object]] = []
        self.common = self._common_metadata()

    def _common_metadata(self) -> dict[str, object]:
        return {
            "suite_case_id": self.case_id,
            "settings_fingerprint": self.expected["settings_fingerprint"],
            "config_hash": self.expected["config_hash"],
            "source_states_id": self.expected["source_states_id"],
            "target_mode": str(self.case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
            "target_state": np.asarray(self.states.target, dtype=float).tolist(),
            "thresholds": dict(self.case_config["objective"]["thresholds"]),
            "transfer_time": float(self.cfg.tf),
            "segments": int(self.cfg.n_segments),
            "substeps_per_segment": int(self.cfg.substeps),
            "amax": float(self.cfg.amax),
            "tail_coast_segments": int(self.case["tail_coast_segments"]),
            "outage_lengths": [int(value) for value in self.case["outage_lengths"]],
            "replay_semantics": (
                "Normalized CR3BP replay of persisted accepted controls only; no optimization rerun, "
                "no high-fidelity validation, no fuel-optimality claim, and no quantum-advantage claim."
            ),
            "limitations": _branch_control_replay_limitations(),
        }

    def _compatible_manifest(self, manifest: dict) -> bool:
        for key in ("suite_case_id", "settings_fingerprint", "config_hash", "source_states_id"):
            if str(manifest.get(key, "")) != str(self.common[key]):
                return False
        for key in ("segments", "substeps_per_segment", "tail_coast_segments"):
            if int(manifest.get(key, -1)) != int(self.common[key]):
                return False
        for key in ("transfer_time", "amax"):
            if abs(float(manifest.get(key, float("nan"))) - float(self.common[key])) > 0.0:
                return False
        return True

    def _read_json_verified(self, path_text: object, expected_sha256: object | None = None) -> tuple[dict, Path, str]:
        path = _resolve_artifact_path(path_text)
        if not path.is_file():
            raise RuntimeError(f"sidecar not found: {path}")
        actual = _sha256(path)
        if expected_sha256 not in (None, "") and str(expected_sha256).strip() != actual:
            raise RuntimeError(f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")
        return json.loads(path.read_text(encoding="utf-8")), path, actual

    def _progress_rows(self, *, completed: bool) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if self.nominal_sha:
            rows.append(
                {
                    "suite_case_id": self.case_id,
                    "record_type": "nominal",
                    "branch_order": "",
                    "mask_index": "",
                    "path": _artifact_path(self.nominal_path),
                    "sha256": self.nominal_sha,
                    "status": "complete",
                }
            )
        for entry in sorted(self.branch_entries, key=lambda item: int(item["branch_order"])):
            rows.append(
                {
                    "suite_case_id": self.case_id,
                    "record_type": "branch",
                    "branch_order": int(entry["branch_order"]),
                    "mask_index": int(entry["mask_index"]),
                    "path": str(entry["path"]),
                    "sha256": str(entry["sha256"]),
                    "status": "complete",
                }
            )
        if not completed and rows:
            rows[-1]["status"] = "last_completed_checkpoint"
        return rows

    def _write_progress_csv(self, *, completed: bool) -> str | None:
        rows = self._progress_rows(completed=completed)
        if not rows:
            return None
        self.progress_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows,
            columns=["suite_case_id", "record_type", "branch_order", "mask_index", "path", "sha256", "status"],
        ).to_csv(self.progress_csv_path, index=False)
        return _sha256(self.progress_csv_path)

    def _expected_branch_count(self, result: dict | None = None) -> int:
        if result is not None:
            if result.get("selected_outage_count") is not None:
                return int(result["selected_outage_count"])
            if result.get("selected_outage_indices") is not None:
                return int(len(list(result["selected_outage_indices"])))
            records = _branch_control_records(result)
            if records:
                return int(len(records))
        return int(self.case["selected_outage_count"])

    def write_nominal(self, result: dict, *, completed: bool = False) -> str:
        nominal_controls = np.asarray(result.get("nominal_controls", result.get("controls")), dtype=float)
        nominal_controls = nominal_controls.reshape((int(self.cfg.n_segments), 3))
        self.nominal_sha = _write_json_with_sha256(
            self.nominal_path,
            {
                "schema_version": 1,
                "sidecar_type": "tail_coast_nominal_controls",
                **self.common,
                "nominal_accepted_candidate": str(result.get("nominal_accepted_candidate", "")),
                "nominal_error": float(result.get("nominal_error", result.get("nominal_tail_coast_error", float("nan")))),
                "nominal_tail_coast_error": float(
                    result.get("nominal_tail_coast_error", result.get("nominal_error", float("nan")))
                ),
                "nominal_seed_error": float(result.get("nominal_seed_error", float("nan"))),
                "nominal_fuel": float(result.get("nominal_fuel", float("nan"))),
                "nominal_dt": float(result.get("nominal_dt", float(self.cfg.tf) / float(self.cfg.n_segments))),
                "controls": nominal_controls.tolist(),
                "control_norm_diagnostics": _control_norm_diagnostics(nominal_controls, float(self.cfg.amax)),
            },
        )
        self.write_manifest(completed=completed)
        return self.nominal_sha

    def write_branch(self, record: dict, branch_order: int, *, completed: bool = False) -> str:
        mask_index = int(record["mask_index"])
        if mask_index < 0 or mask_index >= int(self.masks.shape[0]):
            raise ValueError(f"branch sidecar mask_index {mask_index} outside configured mask array")
        outage_mask = np.asarray(self.masks[mask_index], dtype=int)
        branch_controls = np.asarray(record["branch_controls"], dtype=float).reshape((int(self.cfg.n_segments), 3))
        recovery_controls = record.get("recovery_controls")
        if recovery_controls is not None:
            recovery_controls = np.asarray(recovery_controls, dtype=float).reshape((-1, 3))
        branch_path = self.controls_dir / f"{self.safe_case_id}_branch_{int(branch_order):03d}_mask_{mask_index:03d}_controls.json"
        branch_norms = _control_norm_diagnostics(branch_controls, float(self.cfg.amax))
        robust_threshold = float(self.case_config["objective"]["thresholds"]["robust_success"])
        branch_data = {
            "schema_version": 1,
            "sidecar_type": "tail_coast_branch_controls",
            **self.common,
            "branch_order": int(branch_order),
            "mask_index": mask_index,
            "outage_mask": outage_mask.astype(int).tolist(),
            "recovery_start": int(record["recovery_start"]),
            "recovery_segments": int(record["recovery_segments"]),
            "branch_control_count": int(record.get("branch_control_count", branch_controls.shape[0])),
            "nominal_dt": float(record.get("nominal_dt", float(self.cfg.tf) / float(self.cfg.n_segments))),
            "branch_total_duration": float(record.get("branch_total_duration", self.cfg.tf)),
            "accepted_candidate": str(record.get("accepted_candidate", "")),
            "optimizer_ran": bool(record.get("optimizer_ran", False)),
            "optimizer_success": bool(record.get("optimizer_success", False)),
            "no_recovery_variable_threshold_feasible": bool(
                record.get("no_recovery_variable_threshold_feasible", False)
            ),
            "message": str(record.get("message", "")),
            "nfev": int(record.get("nfev", 0)),
            "runtime_seconds": float(record.get("runtime_seconds", 0.0)),
            "accepted_variant_nfev": int(record.get("accepted_variant_nfev", record.get("nfev", 0))),
            "accepted_variant_runtime_seconds": float(
                record.get("accepted_variant_runtime_seconds", record.get("runtime_seconds", 0.0))
            ),
            "terminal_error": float(record["terminal_error"]),
            "branch_fuel": float(record["branch_fuel"]),
            "accepted_branch_weight_variant_label": str(record.get("accepted_branch_weight_variant_label", "")),
            "accepted_branch_weight_variant_index": int(record.get("accepted_branch_weight_variant_index", 0)),
            "accepted_branch_weights": dict(record.get("accepted_branch_weights", record.get("branch_weights", {}))),
            "accepted_branch_initialization_label": str(record.get("accepted_branch_initialization_label", "")),
            "accepted_branch_initialization_index": int(record.get("accepted_branch_initialization_index", 0)),
            "accepted_branch_initialization_is_fallback": bool(
                record.get("accepted_branch_initialization_is_fallback", False)
            ),
            "accepted_branch_initialization_kind": str(record.get("accepted_branch_initialization_kind", "")),
            "accepted_branch_initialization_vector": record.get("accepted_branch_initialization_vector"),
            "branch_weight_variants": list(record.get("branch_weight_variants", [])),
            "branch_initialization_fallbacks": list(record.get("branch_initialization_fallbacks", [])),
            "branch_portfolio_enabled": bool(record.get("branch_portfolio_enabled", False)),
            "branch_portfolio_variant_count": int(record.get("branch_portfolio_variant_count", 1)),
            "portfolio_acceptance_rule": str(record.get("portfolio_acceptance_rule", "")),
            "portfolio_robust_threshold": _float_or_default(record.get("portfolio_robust_threshold"), robust_threshold),
            "portfolio_converged_threshold_feasible_candidate_count": int(
                record.get("portfolio_converged_threshold_feasible_candidate_count", 0)
            ),
            "portfolio_nominal_converged_threshold_feasible_candidate_count": int(
                record.get("portfolio_nominal_converged_threshold_feasible_candidate_count", 0)
            ),
            "branch_portfolio_candidate_count": int(record.get("branch_portfolio_candidate_count", 0)),
            "branch_portfolio_candidate_optimizer_success_count": int(
                record.get("branch_portfolio_candidate_optimizer_success_count", 0)
            ),
            "branch_portfolio_candidate_all_optimizer_success": bool(
                record.get("branch_portfolio_candidate_all_optimizer_success", False)
            ),
            "branch_fallback_initialization_enabled": bool(record.get("branch_fallback_initialization_enabled", False)),
            "branch_fallback_initialization_configured_count": int(
                record.get("branch_fallback_initialization_configured_count", 0)
            ),
            "branch_fallback_initialization_evaluated_count": int(
                record.get("branch_fallback_initialization_evaluated_count", 0)
            ),
            "branch_fallback_initialization_candidate_count": int(
                record.get("branch_fallback_initialization_candidate_count", 0)
            ),
            "branch_nominal_initialization_candidate_count": int(
                record.get("branch_nominal_initialization_candidate_count", 1)
            ),
            "branch_initialization_variant_count": int(record.get("branch_initialization_variant_count", 1)),
            "branch_portfolio_candidate_results": list(record.get("branch_portfolio_candidate_results", [])),
            "branch_controls_remove_zero_nominal": bool(record.get("branch_controls_remove_zero_nominal", False)),
            "branch_missed_tail_indices": list(record.get("branch_missed_tail_indices", [])),
            "branch_note": str(record.get("branch_note", "")),
            "control_max_norm": float(record.get("control_max_norm", branch_norms["control_norm_max"])),
            "control_bound_violation": float(record.get("control_bound_violation", branch_norms["control_bound_violation"])),
            "branch_controls": branch_controls.tolist(),
            "branch_control_norm_diagnostics": branch_norms,
        }
        if recovery_controls is not None:
            branch_data["recovery_controls"] = recovery_controls.tolist()
            branch_data["recovery_control_norm_diagnostics"] = _control_norm_diagnostics(
                recovery_controls,
                float(self.cfg.amax),
            )
        for key in (
            "branch_weights",
            "target_error_semantics",
            "cost",
            "accepted_cost",
            "optimality",
            "accepted_optimality",
            "optimizer_cost",
            "optimizer_optimality",
        ):
            if key in record:
                branch_data[key] = record[key]
        branch_sha = _write_json_with_sha256(branch_path, branch_data)
        branch_entry = {
            "branch_order": int(branch_order),
            "mask_index": mask_index,
            "outage_mask": outage_mask.astype(int).tolist(),
            "recovery_start": int(record["recovery_start"]),
            "recovery_segments": int(record["recovery_segments"]),
            "terminal_error": float(record["terminal_error"]),
            "branch_fuel": float(record["branch_fuel"]),
            "optimizer_ran": bool(record.get("optimizer_ran", False)),
            "optimizer_success": bool(record.get("optimizer_success", False)),
            "path": _artifact_path(branch_path),
            "sha256": branch_sha,
        }
        self.branch_entries = [
            entry
            for entry in self.branch_entries
            if not (
                int(entry.get("branch_order", -1)) == int(branch_order)
                or int(entry.get("mask_index", -1)) == mask_index
            )
        ]
        self.branch_entries.append(branch_entry)
        self.branch_entries = sorted(self.branch_entries, key=lambda item: int(item["branch_order"]))
        self.write_manifest(completed=completed)
        return branch_sha

    def write_manifest(self, result: dict | None = None, *, completed: bool = False) -> str | None:
        if not self.nominal_sha and not self.branch_entries:
            return None
        progress_sha = self._write_progress_csv(completed=completed)
        expected_branch_count = self._expected_branch_count(result)
        complete = bool(completed and len(self.branch_entries) == expected_branch_count)
        manifest_data = {
            "schema_version": 1,
            "sidecar_type": "tail_coast_branch_control_manifest",
            **self.common,
            "nominal_control_path": _artifact_path(self.nominal_path) if self.nominal_sha else None,
            "nominal_control_sha256": self.nominal_sha,
            "progress_csv_path": _artifact_path(self.progress_csv_path) if progress_sha else None,
            "progress_csv_sha256": progress_sha,
            "progress_state": "complete" if complete else "in_progress",
            "expected_branch_count": expected_branch_count,
            "branch_count": int(len(self.branch_entries)),
            "branch_control_sidecar_count": int(len(self.branch_entries)),
            "branch_control_replay_ready": complete,
            "selected_outage_count": expected_branch_count,
            "selected_outage_indices": list(result.get("selected_outage_indices", [])) if result is not None else [],
            "nominal_error": (
                float(result.get("nominal_error", result.get("nominal_tail_coast_error", float("nan"))))
                if result is not None
                else None
            ),
            "selected_worst_error": float(result.get("selected_worst_error", float("nan"))) if result is not None else None,
            "all_mask_worst_error": float(result.get("all_mask_worst_error", float("nan"))) if result is not None else None,
            "meets_thresholds": bool(result.get("meets_thresholds", False)) if result is not None else None,
            "resume_semantics": (
                "On --resume, compatible completed branch sidecars are loaded by mask_index and the corresponding "
                "branch optimizations are skipped; remaining branches are optimized and checkpointed incrementally."
            ),
            "branch_control_sidecars": self.branch_entries,
        }
        return _write_json_with_sha256(self.manifest_path, manifest_data)

    def load_completed_branch_records(self) -> dict[int, dict]:
        if not self.manifest_path.is_file():
            return {}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not self._compatible_manifest(manifest):
            return {}
        if manifest.get("nominal_control_sha256"):
            self.nominal_sha = str(manifest["nominal_control_sha256"])
        loaded: dict[int, dict] = {}
        entries = sorted(list(manifest.get("branch_control_sidecars", [])), key=lambda item: int(item["branch_order"]))
        for entry in entries:
            branch, _, branch_sha = self._read_json_verified(entry["path"], entry.get("sha256"))
            if not self._compatible_manifest(branch):
                continue
            branch_order = int(branch["branch_order"])
            mask_index = int(branch["mask_index"])
            branch["branch_controls"] = np.asarray(branch["branch_controls"], dtype=float)
            if branch.get("recovery_controls") is not None:
                branch["recovery_controls"] = np.asarray(branch["recovery_controls"], dtype=float)
            loaded[mask_index] = branch
            self.branch_entries.append(
                {
                    "branch_order": branch_order,
                    "mask_index": mask_index,
                    "outage_mask": [int(value) for value in branch["outage_mask"]],
                    "recovery_start": int(branch["recovery_start"]),
                    "recovery_segments": int(branch["recovery_segments"]),
                    "terminal_error": float(branch["terminal_error"]),
                    "branch_fuel": float(branch["branch_fuel"]),
                    "optimizer_ran": bool(branch.get("optimizer_ran", False)),
                    "optimizer_success": bool(branch.get("optimizer_success", False)),
                    "path": str(entry["path"]),
                    "sha256": branch_sha,
                }
            )
        self.branch_entries = sorted(self.branch_entries, key=lambda item: int(item["branch_order"]))
        return loaded

    def finalize(self, result: dict) -> dict[str, object]:
        branch_records = _branch_control_records(result)
        if not branch_records:
            return _empty_sidecar_columns()
        self.write_nominal(result, completed=False)
        for branch_order, record in enumerate(branch_records):
            self.write_branch(record, int(branch_order), completed=False)
        manifest_sha = self.write_manifest(result, completed=True)
        expected_branch_count = self._expected_branch_count(result)
        return {
            "nominal_control_path": _artifact_path(self.nominal_path),
            "nominal_control_sha256": self.nominal_sha,
            "branch_control_manifest_path": _artifact_path(self.manifest_path),
            "branch_control_manifest_sha256": manifest_sha,
            "branch_control_sidecar_count": int(len(self.branch_entries)),
            "branch_control_replay_ready": bool(len(self.branch_entries) == expected_branch_count),
        }


def _write_control_sidecars(
    *,
    results_dir: Path,
    case: dict,
    case_config: dict,
    states,
    cfg,
    masks: np.ndarray,
    result: dict,
    expected: dict,
) -> dict[str, object]:
    store = BranchControlSidecarStore(
        results_dir=results_dir,
        case=case,
        case_config=case_config,
        states=states,
        cfg=cfg,
        masks=masks,
        expected=expected,
    )
    return store.finalize(result)


def _expected_index(config: dict, args, cases: list[dict]) -> dict[str, dict]:
    expected = {}
    for case in cases:
        settings = _effective_settings(config, args, case)
        expected[str(case["suite_case_id"])] = {
            "case": case,
            "settings_fingerprint": settings_fingerprint(settings),
            "config_hash": settings["config_hash"],
            "source_states_id": settings["source_states_id"],
        }
    return expected


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=TAIL_COAST_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in TAIL_COAST_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[TAIL_COAST_COLUMNS]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _missing_or_different(value, expected: str) -> bool:
    return _is_missing(value) or str(value) != str(expected)


def _compatible_existing_rows(df: pd.DataFrame, expected: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return pd.DataFrame(columns=TAIL_COAST_COLUMNS), []
    kept = []
    rejected = []
    seen = set()
    for row in df.to_dict(orient="records"):
        case_id = str(row.get("suite_case_id", ""))
        expected_row = expected.get(case_id)
        if expected_row is None:
            rejected.append({"suite_case_id": case_id, "reason": "not in current requested case set"})
            continue
        if _missing_or_different(row.get("settings_fingerprint"), expected_row["settings_fingerprint"]):
            rejected.append({"suite_case_id": case_id, "reason": "settings_fingerprint missing or mismatched"})
            continue
        mismatched = [key for key in ("config_hash", "source_states_id") if _missing_or_different(row.get(key), expected_row[key])]
        if mismatched:
            rejected.append({"suite_case_id": case_id, "reason": "provenance field mismatch", "mismatched_fields": mismatched})
            continue
        fingerprint = str(row["settings_fingerprint"])
        if fingerprint in seen:
            rejected.append({"suite_case_id": case_id, "reason": "duplicate compatible settings_fingerprint"})
            continue
        kept.append({column: row.get(column) for column in TAIL_COAST_COLUMNS})
        seen.add(fingerprint)
    return pd.DataFrame(kept, columns=TAIL_COAST_COLUMNS), rejected


def _artifact_refresh_rows_by_case(
    df: pd.DataFrame,
    expected: dict[str, dict],
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Keep existing evidence rows for table/plot refresh after table-only edits.

    This path is intentionally opt-in at the CLI. It still rejects unknown cases,
    duplicate case IDs, and config/source provenance mismatches; it only permits a
    settings_fingerprint mismatch caused by code/reporting identity changes.
    """
    if df.empty:
        return pd.DataFrame(columns=TAIL_COAST_COLUMNS), [], []
    kept = []
    rejected = []
    accepted_fingerprint_mismatches = []
    seen = set()
    for row in df.to_dict(orient="records"):
        case_id = str(row.get("suite_case_id", ""))
        expected_row = expected.get(case_id)
        if expected_row is None:
            rejected.append({"suite_case_id": case_id, "reason": "not in current requested case set"})
            continue
        mismatched = [key for key in ("config_hash", "source_states_id") if _missing_or_different(row.get(key), expected_row[key])]
        if mismatched:
            rejected.append({"suite_case_id": case_id, "reason": "provenance field mismatch", "mismatched_fields": mismatched})
            continue
        if case_id in seen:
            rejected.append({"suite_case_id": case_id, "reason": "duplicate compatible suite_case_id"})
            continue
        if _missing_or_different(row.get("settings_fingerprint"), expected_row["settings_fingerprint"]):
            accepted_fingerprint_mismatches.append(
                {
                    "suite_case_id": case_id,
                    "recorded_settings_fingerprint": str(row.get("settings_fingerprint", "")),
                    "current_settings_fingerprint": expected_row["settings_fingerprint"],
                    "reason": "accepted for artifact refresh only after config_hash and source_states_id matched",
                }
            )
        kept.append({column: row.get(column) for column in TAIL_COAST_COLUMNS})
        seen.add(case_id)
    return pd.DataFrame(kept, columns=TAIL_COAST_COLUMNS), rejected, accepted_fingerprint_mismatches


def _row_from_result(
    case: dict,
    case_config: dict,
    states,
    cfg,
    result: dict,
    expected: dict,
    config: dict,
    sidecar_columns: dict[str, object] | None = None,
) -> dict:
    weights = _branch_weights(config, case["case_raw"])
    variants = _branch_weight_variants(config, case["case_raw"])
    fallback_initializations = _branch_initialization_fallbacks(config, case["case_raw"])
    tolerances = _tolerances(config, case["case_raw"])
    row = {
        **_case_payload(case),
        "purpose": str(case["purpose"]),
        "target_mode": str(case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
        "target_generation": str((getattr(states, "target_metadata", {}) or {}).get("target_state_generation", "catalog target generation")),
        "substeps_per_segment": int(cfg.substeps),
        "outage_lengths": _json_list(case["outage_lengths"]),
        "selected_outages": str(case["selected_outages_raw"]),
        "terminal_weight": weights.terminal,
        "control_weight": weights.control,
        "smooth_weight": weights.smooth,
        "continuity_weight": weights.continuity,
        **tolerances,
        "settings_fingerprint": expected["settings_fingerprint"],
        "config_hash": expected["config_hash"],
        "source_states_id": expected["source_states_id"],
        **(sidecar_columns or _empty_sidecar_columns()),
        "original_target_state": _json_list(np.asarray(result["original_target_state"], dtype=float).tolist()),
        "branch_weight_variants": _json_list(variants),
        "branch_fallback_initialization_enabled": bool(fallback_initializations),
        "branch_fallback_initialization_configured_count": int(len(fallback_initializations)),
        "branch_fallback_initialization_labels": _json_list([str(item["label"]) for item in fallback_initializations]),
        "branch_initialization_fallbacks": _json_list(fallback_initializations),
    }
    direct_fields = [column for column in TAIL_COAST_COLUMNS if column not in row]
    for field in direct_fields:
        value = result.get(field)
        if isinstance(value, (list, dict)):
            value = _json_list(value)
        row[field] = value
    return {column: row.get(column) for column in TAIL_COAST_COLUMNS}


def run_case(config: dict, args, case: dict, expected: dict, results_dir: Path) -> dict:
    case_config = _case_config(config, case)
    states = load_configured_states(Path.cwd(), case_config, args.source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    tolerances = _tolerances(config, case["case_raw"])
    sidecar_store = BranchControlSidecarStore(
        results_dir=results_dir,
        case=case,
        case_config=case_config,
        states=states,
        cfg=cfg,
        masks=masks,
        expected=expected,
    )
    branch_result_overrides = sidecar_store.load_completed_branch_records() if bool(getattr(args, "resume", False)) else {}
    result = run_tail_coast_recovery_baseline(
        state0=states.initial,
        target=states.target,
        cfg=cfg,
        masks=masks,
        thresholds=case_config["objective"]["thresholds"],
        tail_coast_segments=int(case["tail_coast_segments"]),
        selected_outages=case["selected_outages_raw"],
        nominal_max_nfev=int(case["nominal_max_nfev"]),
        tail_nominal_max_nfev=int(case["tail_nominal_max_nfev"]),
        branch_max_nfev=int(case["branch_max_nfev"]),
        nominal_residual_weights=_nominal_residual_weights(config, case["case_raw"]),
        tail_nominal_weights=_tail_nominal_weights(config, case["case_raw"]),
        branch_weights=_branch_weights(config, case["case_raw"]),
        branch_weight_variants=_branch_weight_variants(config, case["case_raw"]),
        branch_initialization_fallbacks=_branch_initialization_fallback_config(config, case["case_raw"]),
        node_initialization=str(case["node_initialization"]),
        node_initialization_blend=float(case["node_initialization_blend"]),
        accepted_nominal_callback=sidecar_store.write_nominal,
        accepted_branch_callback=sidecar_store.write_branch,
        branch_result_overrides=branch_result_overrides,
        **tolerances,
    )
    sidecar_columns = sidecar_store.finalize(result)
    return _row_from_result(case, case_config, states, cfg, result, expected, config, sidecar_columns)


def write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    path = tables_dir / "tail_coast_recovery_table.tex"
    if df.empty:
        path.write_text("% No tail-coast recovery rows.\n", encoding="utf-8")
        return
    table_df = df.copy()
    table_df["no_recovery_branch_count"] = table_df["branch_recovery_segments"].map(_count_zero_recovery_branches)
    table_df["eligible_optimizer_converged_branches"] = table_df.apply(_eligible_optimizer_converged_count, axis=1)
    table_df["selected_outage_policy_readable"] = table_df["selected_outage_policy"].map(_readable_selected_outage_policy)
    table = table_df[
        [
            "suite_case_id",
            "selected_outage_policy_readable",
            "outage_lengths",
            "selected_outage_count",
            "tail_coast_segments",
            "branch_portfolio_variant_count",
            "branch_fallback_initialization_evaluated_branch_count",
            "branch_fallback_initialization_accepted_branch_count",
            "eligible_optimizer_converged_branches",
            "no_recovery_branch_count",
            "nominal_tail_coast_error",
            "selected_worst_error",
            "all_mask_worst_error",
            "control_max_norm",
            "control_bound_violation",
            "nfev",
            "meets_thresholds",
        ]
    ].copy()
    table.columns = [
        "Case",
        "Policy",
        "Outage lengths",
        "Selected masks",
        "Tail coast segments",
        "Portfolio variants",
        "Fallback eval branches",
        "Accepted fallback branches",
        "Eligible optimizer-converged branches",
        "Direct no-recovery branches",
        "Tail-coast nominal error",
        "Selected fixed-time worst error",
        "All-mask fixed-time diagnostic worst",
        "Max ||u||",
        "Bound violation",
        "nfev",
        "Meets thresholds",
    ]
    path.write_text(table.to_latex(index=False, float_format="%.4f", escape=True), encoding="utf-8")


def write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if df.empty:
        ax.text(0.5, 0.5, "No tail-coast recovery rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        colors = np.where(df["meets_thresholds"].astype(bool).to_numpy(), "tab:green", "tab:red")
        ax.scatter(df["nominal_tail_coast_error"], df["selected_worst_error"], c=colors, s=90, edgecolors="black", linewidths=0.6)
        for index, (_, row) in enumerate(df.iterrows()):
            ax.annotate(
                str(row["suite_case_id"]).replace("tail_coast_", ""),
                (float(row["nominal_tail_coast_error"]), float(row["selected_worst_error"])),
                textcoords="offset points",
                xytext=(6, 5 if index % 2 == 0 else -10),
                fontsize=8,
            )
        ax.axvline(float(df["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0)
        ax.axhline(float(df["selected_worst_threshold"].iloc[0]), color="0.45", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Tail-coast nominal original-target error")
        ax.set_ylabel("Selected fixed-final-time worst error")
        ax.set_title("Fixed-final-time tail-coast recovery")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"tail_coast_recovery{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def regenerate(
    df: pd.DataFrame,
    *,
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    config: dict,
    command: str,
    cases: list[dict],
    resume_rejected_rows: list[dict],
    skipped_cases: list[dict],
    write_csv: bool = True,
    artifact_refresh_accepted_fingerprint_mismatch_rows: list[dict] | None = None,
    artifact_refresh_allows_fingerprint_mismatch: bool = False,
    artifact_refresh_command: str | None = None,
    evidence_replay_command: str | None = None,
) -> None:
    df = pd.DataFrame(df, columns=TAIL_COAST_COLUMNS)
    if write_csv:
        df.to_csv(results_dir / "tail_coast_recovery.csv", index=False)
    write_table(df, tables_dir)
    write_plot(df, figures_dir)
    sidecar_counts = (
        pd.to_numeric(df["branch_control_sidecar_count"], errors="coerce").fillna(0)
        if "branch_control_sidecar_count" in df
        else pd.Series(dtype=float)
    )
    extra = {
        "row_count": int(len(df)),
        "feasible_row_count": int(df["meets_thresholds"].astype(bool).sum()) if not df.empty else 0,
        "branch_control_replay_ready_row_count": (
            int(df["branch_control_replay_ready"].fillna(False).astype(bool).sum())
            if "branch_control_replay_ready" in df and not df.empty
            else 0
        ),
        "branch_control_sidecar_count": int(sidecar_counts.sum()) if len(sidecar_counts) else 0,
        "expected_case_count": int(len(cases)),
        "completed_case_count": int(len(df)),
        "raw_csv_written": bool(write_csv),
        "raw_csv_write_semantics": (
            "--regenerate-artifacts-only leaves the existing tail_coast_recovery.csv bytes untouched; "
            "only optimization/resume runs rewrite the raw evidence CSV."
        ),
        "artifact_refresh_command": artifact_refresh_command or command,
        "evidence_replay_command": evidence_replay_command or command,
        "resume_rejected_rows": resume_rejected_rows,
        "skipped_cases": skipped_cases,
        "artifact_refresh_allows_fingerprint_mismatch": bool(artifact_refresh_allows_fingerprint_mismatch),
        "artifact_refresh_accepted_fingerprint_mismatch_rows": artifact_refresh_accepted_fingerprint_mismatch_rows or [],
        "implementation_identities": _implementation_identities(),
        "threshold_rule": "meets_thresholds requires tail-coast nominal error <= nominal_success and selected fixed-final-time worst error <= robust_success",
        "semantics": {
            "backend": "Fixed-final-time tail-coast locked-nominal continuous recovery evidence; not quantum evidence, not delayed arrival, not fuel optimality, and not robustness beyond the configured outage masks.",
            "nominal": "The nominal solve is seeded from an all-windows multiple-shooting trajectory, then refined with configured tail_nominal weights while the final tail_coast_segments controls are fixed exactly to zero.",
            "branch_recovery": (
                "Each selected missed-thrust mask is evaluated independently at the original target and original transfer time; "
                "branches with post-outage controls run an optimizer, while no-recovery-variable branches report direct "
                "threshold feasibility without claiming optimizer convergence."
            ),
            "selection": "all_single selects all one-segment masks; all/all_configured selects all configured masks; integer policies choose hardest masks by masked fixed-final-time terminal error.",
            "resume": "resume reuses only rows whose suite_case_id, settings_fingerprint, config_hash, and source_states_id match current settings.",
            "branch_weight_portfolio": (
                "Every configured branch residual variant is evaluated for every selected outage mask and all variant "
                "nfev/runtime are charged; optimizer convergence is separate from threshold feasibility. "
                "branch_portfolio_all_success is an accepted-branch result flag retained for CSV compatibility, "
                "not a claim that every evaluated portfolio candidate converged."
            ),
            "branch_initialization_fallbacks": (
                "Each selected branch first evaluates the branch weight portfolio from the nominal post-outage controls. "
                "Configured constant-vector fallback starts are evaluated across all branch weight variants only when "
                "the nominal-start portfolio has no optimizer-converged threshold-feasible candidate; all evaluated "
                "fallback nfev/runtime are charged. The row-level portfolio_acceptance_rule summarizes whether any "
                "selected branch evaluated or accepted fallbacks; branch_results retains each branch-specific rule."
            ),
            "branch_control_sidecars": (
                "Rows with accepted branch-control records write deterministic JSON sidecars for the nominal controls, "
                "each accepted full branch-control schedule, and a SHA-256 manifest. Sidecars support normalized CR3BP "
                "accepted-control replay only and are not high-fidelity validation or fuel-optimality evidence."
            ),
        },
        "limitations": [
            "This is continuous-backend evidence and does not establish quantum advantage.",
            "This is fixed-final-time evidence and intentionally does not use delayed targets or added recovery horizon segments.",
            "The configured evidence scope is case-specific; outage_lengths and selected_outage_policy record which missed-thrust masks each row covers.",
        ],
        "expected_cases": [_case_payload(case) for case in cases],
        "branch_weight_variants_by_case": {str(case["suite_case_id"]): _branch_weight_variants(config, case["case_raw"]) for case in cases},
        "branch_initialization_fallbacks_by_case": {
            str(case["suite_case_id"]): _branch_initialization_fallbacks(config, case["case_raw"])
            for case in cases
        },
        "rows": df.to_dict(orient="records"),
    }
    write_metadata(results_dir / "tail_coast_recovery_metadata.json", command, config, extra)


def _order_rows_by_cases(df: pd.DataFrame, cases: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(df, columns=TAIL_COAST_COLUMNS)
    if df.empty:
        return df
    order = {str(case["suite_case_id"]): int(case["case_order"]) for case in cases}
    df["_case_order"] = df["suite_case_id"].map(order).fillna(len(order))
    return df.sort_values("_case_order").drop(columns=["_case_order"]).reset_index(drop=True)


def _command_part(value: object) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(character.isspace() for character in text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _evidence_replay_command(args) -> str:
    parts: list[object] = ["py", "-3.11", r"scripts\run_tail_coast_recovery.py", "--config", args.config, "--resume"]
    if Path(args.source_states) != Path("data/source_states.json"):
        parts.extend(["--source-states", args.source_states])
    if args.max_cases is not None:
        parts.extend(["--max-cases", int(args.max_cases)])
    return " ".join(_command_part(part) for part in parts)


def _missing_compatible_cases(df: pd.DataFrame, cases: list[dict], expected: dict[str, dict]) -> list[dict]:
    completed = set(str(value) for value in df["suite_case_id"].dropna().tolist()) if "suite_case_id" in df else set()
    missing = []
    for case in cases:
        case_id = str(case["suite_case_id"])
        if case_id in completed:
            continue
        missing.append(
            {
                "suite_case_id": case_id,
                "reason": "no compatible existing row for regenerate-artifacts-only",
                "expected_settings_fingerprint": expected[case_id]["settings_fingerprint"],
            }
        )
    return missing


def _runtime_budget(args, config: dict) -> float | None:
    if args.runtime_budget_seconds is not None:
        return float(args.runtime_budget_seconds)
    value = _tail_config(config).get("runtime_budget_seconds")
    if value is None:
        value = (config.get("suite", {}) or {}).get("runtime_budget_seconds")
    return None if value is None else float(value)


def run(args) -> pd.DataFrame:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path.cwd()
    results_dir, figures_dir, tables_dir = output_directories(root, config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "tail_coast_recovery.csv"
    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    command = " ".join(sys.argv)
    evidence_replay_command = _evidence_replay_command(args)
    if args.regenerate_artifacts_only:
        expected = _expected_index(config, args, cases)
        accepted_fingerprint_mismatches: list[dict] = []
        if args.allow_artifact_refresh_fingerprint_mismatch:
            df, resume_rejected_rows, accepted_fingerprint_mismatches = _artifact_refresh_rows_by_case(
                _load_existing(csv_path),
                expected,
            )
        else:
            df, resume_rejected_rows = _compatible_existing_rows(_load_existing(csv_path), expected)
        df = _order_rows_by_cases(df, cases)
        skipped_cases = _missing_compatible_cases(df, cases, expected)
        regenerate(
            df,
            results_dir=results_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            config=config,
            command=command,
            cases=cases,
            resume_rejected_rows=resume_rejected_rows,
            skipped_cases=skipped_cases,
            artifact_refresh_accepted_fingerprint_mismatch_rows=accepted_fingerprint_mismatches,
            artifact_refresh_allows_fingerprint_mismatch=bool(args.allow_artifact_refresh_fingerprint_mismatch),
            write_csv=False,
            artifact_refresh_command=command,
            evidence_replay_command=evidence_replay_command,
        )
        return df
    expected = _expected_index(config, args, cases)
    if args.resume:
        df, resume_rejected_rows = _compatible_existing_rows(_load_existing(csv_path), expected)
    else:
        df = pd.DataFrame(columns=TAIL_COAST_COLUMNS)
        resume_rejected_rows = []
    completed = set(str(value) for value in df["settings_fingerprint"].dropna().tolist())
    skipped_cases: list[dict] = []
    budget = _runtime_budget(args, config)
    started = time.perf_counter()
    for case in cases:
        expected_row = expected[str(case["suite_case_id"])]
        if expected_row["settings_fingerprint"] in completed:
            continue
        elapsed = time.perf_counter() - started
        if budget is not None and elapsed >= budget:
            skipped_cases.append({"suite_case_id": str(case["suite_case_id"]), "reason": "runtime budget reached before launching case"})
            continue
        row = run_case(config, args, case, expected_row, results_dir)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        regenerate(
            df,
            results_dir=results_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            config=config,
            command=command,
            cases=cases,
            resume_rejected_rows=resume_rejected_rows,
            skipped_cases=skipped_cases,
            artifact_refresh_command=command,
            evidence_replay_command=evidence_replay_command,
        )
        print(
            f"case {case['suite_case_id']} tail={row['tail_coast_segments']} policy={row['selected_outage_policy']} "
            f"selected={row['selected_outage_count']}/{row['outage_count']}: nominal={row['nominal_error']:.6f}, "
            f"selected_fixed={row['selected_worst_error']:.6f}, all_fixed={row['all_mask_worst_error']:.6f}, "
            f"met={row['meets_thresholds']}, portfolio_variants={row['branch_portfolio_variant_count']}, "
            f"fallback_evals={row['branch_fallback_initialization_evaluated_counts']}, "
            f"branch_opt_ran={row['branch_optimizer_ran']}, branch_opt_all_success={row['branch_optimizer_all_success']}, "
            f"nfev={row['nfev']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )
    df = _order_rows_by_cases(df, cases)
    regenerate(
        df,
        results_dir=results_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        config=config,
        command=command,
        cases=cases,
        resume_rejected_rows=resume_rejected_rows,
        skipped_cases=skipped_cases,
        artifact_refresh_command=command,
        evidence_replay_command=evidence_replay_command,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-final-time tail-coast locked-nominal independent branch-recovery baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/hard_catalog_tail_coast_recovery.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--runtime-budget-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--regenerate-artifacts-only",
        action="store_true",
        help="Refresh table, figure, and metadata from the existing CSV without launching optimization or rewriting the CSV.",
    )
    parser.add_argument(
        "--allow-artifact-refresh-fingerprint-mismatch",
        action="store_true",
        help=(
            "With --regenerate-artifacts-only, reuse existing rows whose case_id, "
            "config_hash, and source_states_id match even if settings_fingerprint "
            "changed after reporting-only code edits. Does not run optimization."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
