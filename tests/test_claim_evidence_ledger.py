from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_claim_evidence_ledger_rows_and_semantics():
    module = load_script_module(
        "run_claim_evidence_ledger_unit",
        ROOT / "scripts" / "run_claim_evidence_ledger.py",
    )

    ledger = module.build_claim_evidence_ledger()

    assert len(ledger) == 8
    assert ledger["claim_id"].tolist() == [
        "phase_shift_main_method_30seed_selected_branch",
        "phase_shift_qaoa_qubo_30seed_selected_branch",
        "phase_shift_continuation_all_single_p04_all_configured",
        "phase_shift_continuation_two_segment_n8_p03_all_configured",
        "phase_shift_direct_collocation_p04_selected_branch_diagnostic",
        "phase_shift_independent_hs_p04_amax02_selected_branch_diagnostic",
        "catalog_dro_tail_coast_all_one_two_segment_t5_all_configured",
        "catalog_dro_delayed_h6_all_single_delayed_arrival",
    ]

    def row(claim_id: str) -> pd.Series:
        return ledger.loc[ledger["claim_id"] == claim_id].iloc[0]

    main = row("phase_shift_main_method_30seed_selected_branch")
    assert main["all_configured_mask_evidence"] == "False"
    assert main["target_mode"] == "catalog_halo_phase_shift"
    assert "No sampled-method or QAOA superiority" in main["primary_interpretation"]
    assert "all-windows continuous" in main["primary_interpretation"]
    assert "all_configured" not in main["mask_scope"]

    qaoa = row("phase_shift_qaoa_qubo_30seed_selected_branch")
    assert qaoa["all_configured_mask_evidence"] == "False"
    assert "not support statistical superiority" in qaoa["primary_interpretation"]
    assert "quantum-advantage" in qaoa["explicit_boundary"]

    all_single = row("phase_shift_continuation_all_single_p04_all_configured")
    assert all_single["all_configured_mask_evidence"] == "True"
    assert "8/8 configured one-segment masks" in all_single["mask_scope"]
    assert all_single["nominal_error"] == "0.0530980832118395"
    assert all_single["selected_worst_error"] == "0.0139134347944667"
    assert all_single["all_mask_worst_error"] == "0.0139134347944667"

    two_segment = row("phase_shift_continuation_two_segment_n8_p03_all_configured")
    assert two_segment["all_configured_mask_evidence"] == "True"
    assert "15/15 configured one- and two-segment masks" in two_segment["mask_scope"]
    assert two_segment["nominal_error"] == "0.0612012101866208"
    assert two_segment["selected_worst_error"] == "0.0434908055815499"

    direct = row("phase_shift_direct_collocation_p04_selected_branch_diagnostic")
    assert direct["all_configured_mask_evidence"] == "False"
    assert direct["nominal_error"] == "0.03567443236113212"
    assert direct["selected_worst_error"] == "0.019117085167272115"
    assert direct["all_mask_worst_error"] == "0.06024618745953626"
    assert "all masks were not selected" in direct["explicit_boundary"]

    ihs = row("phase_shift_independent_hs_p04_amax02_selected_branch_diagnostic")
    assert ihs["all_configured_mask_evidence"] == "False"
    assert ihs["backend_or_method"] == "independent midpoint controls"
    assert ihs["nominal_error"] == "0.0197147568098046"
    assert ihs["all_mask_worst_error"] == "0.0531572965780589"

    tail = row("catalog_dro_tail_coast_all_one_two_segment_t5_all_configured")
    assert tail["all_configured_mask_evidence"] == "True"
    assert tail["target_mode"] == "catalog_dro_phase"
    assert "27/27 configured one- and two-segment masks" in tail["mask_scope"]
    assert tail["nominal_error"] == "0.02299233817855882"
    assert tail["selected_worst_error"] == "0.0936063931709301"
    assert tail["all_mask_worst_error"] == "0.0936063931709301"
    assert "no high-fidelity" in tail["explicit_boundary"]

    delayed = row("catalog_dro_delayed_h6_all_single_delayed_arrival")
    assert delayed["all_configured_mask_evidence"] == "True"
    assert delayed["target_mode"] == "catalog_dro_phase"
    assert "14/14 configured one-segment masks" in delayed["mask_scope"]
    assert "Delayed-arrival evidence only" in delayed["explicit_boundary"]


def test_tail_coast_threshold_audit_statuses():
    module = load_script_module(
        "run_claim_evidence_ledger_threshold_unit",
        ROOT / "scripts" / "run_claim_evidence_ledger.py",
    )

    audit = module.build_tail_coast_threshold_audit()

    assert len(audit) == 5
    statuses = {
        row["threshold_id"]: row["passes_threshold_pair"]
        for row in audit.to_dict(orient="records")
    }
    assert statuses == {
        "configured_0p09_0p17": "True",
        "screen_0p05_0p12": "True",
        "near_margin_0p025_0p095": "True",
        "robust_tight_0p025_0p09": "False",
        "very_tight_0p02_0p09": "False",
    }

    robust_tight = audit.loc[audit["threshold_id"] == "robust_tight_0p025_0p09"].iloc[0]
    assert robust_tight["nominal_pass"] == "True"
    assert robust_tight["selected_worst_pass"] == "False"
    assert robust_tight["all_mask_worst_pass"] == "False"

    very_tight = audit.loc[audit["threshold_id"] == "very_tight_0p02_0p09"].iloc[0]
    assert very_tight["nominal_pass"] == "False"
    assert very_tight["selected_worst_pass"] == "False"
    assert very_tight["all_mask_worst_pass"] == "False"
    assert "Recorded-error threshold audit only" in very_tight["audit_semantics"]


def test_tail_coast_branch_audit_counts():
    module = load_script_module(
        "run_claim_evidence_ledger_branch_unit",
        ROOT / "scripts" / "run_claim_evidence_ledger.py",
    )

    audit = module.build_tail_coast_branch_audit()

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["suite_case_id"] == "tail_coast_all_one_two_segment_t5_portfolio"
    assert row["target_mode"] == "catalog_dro_phase"
    assert int(row["total_branches"]) == 27
    assert int(row["optimizer_ran_count"]) == 25
    assert int(row["eligible_optimizer_branch_count"]) == 25
    assert int(row["optimizer_success_count"]) == 25
    assert int(row["optimizer_success_count_among_optimizer_ran"]) == 25
    assert int(row["no_recovery_direct_evaluation_count"]) == 2
    assert int(row["no_recovery_threshold_feasible_count"]) == 2
    assert int(row["fallback_accepted_count"]) == 4
    assert row["max_terminal_error_overall"] == "0.0936063931709301"
    assert row["max_terminal_error_optimizer_ran"] == "0.0936063931709301"
    assert row["max_terminal_error_no_recovery"] == "0.02299233817855882"
    assert "terminal_only=21" in row["accepted_weight_variant_counts"]
    assert "regularized_001=6" in row["accepted_weight_variant_counts"]
    assert "no_recovery_variables=2" in row["accepted_initialization_kind_counts"]
    assert row["branch_control_replay_claim"] == "False"
    assert "not branch-control replay" in row["audit_semantics"]


def test_claim_evidence_ledger_writes_deterministic_artifacts_without_optimization(tmp_path):
    module = load_script_module(
        "run_claim_evidence_ledger_artifact_unit",
        ROOT / "scripts" / "run_claim_evidence_ledger.py",
    )

    kwargs = {
        "results_dir": tmp_path / "results",
        "tables_dir": tmp_path / "tables",
        "command": "unit-test",
    }
    first = module.write_artifacts(**kwargs)
    first_bytes = {
        "ledger": first["ledger_path"].read_bytes(),
        "metadata": first["metadata_path"].read_bytes(),
        "threshold": first["threshold_audit_path"].read_bytes(),
        "branch": first["branch_audit_path"].read_bytes(),
        "ledger_table": first["claim_evidence_ledger_table_tex"].read_bytes(),
        "threshold_table": first["tail_coast_threshold_audit_table_tex"].read_bytes(),
        "branch_table": first["tail_coast_branch_audit_table_tex"].read_bytes(),
    }
    second = module.write_artifacts(**kwargs)

    assert second["ledger_path"].read_bytes() == first_bytes["ledger"]
    assert second["metadata_path"].read_bytes() == first_bytes["metadata"]
    assert second["threshold_audit_path"].read_bytes() == first_bytes["threshold"]
    assert second["branch_audit_path"].read_bytes() == first_bytes["branch"]
    assert second["claim_evidence_ledger_table_tex"].read_bytes() == first_bytes["ledger_table"]
    assert second["tail_coast_threshold_audit_table_tex"].read_bytes() == first_bytes["threshold_table"]
    assert second["tail_coast_branch_audit_table_tex"].read_bytes() == first_bytes["branch_table"]

    metadata = json.loads(second["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["optimization_rerun"] is False
    assert metadata["uses_recorded_artifacts_only"] is True
    assert metadata["high_fidelity_claim"] is False
    assert metadata["branch_control_replay"] is False
    assert metadata["fuel_optimality_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    assert metadata["row_count"] == 8
    assert metadata["tail_coast_threshold_audit_row_count"] == 5
    assert metadata["tail_coast_branch_audit_row_count"] == 1
    assert "no trajectory optimization" in metadata["source_mode"]
    assert "accepted branch controls are not persisted" in " ".join(metadata["interpretation_limits"])
    assert len(metadata["input_artifacts"]) == 10

    ledger = pd.read_csv(second["ledger_path"])
    assert len(ledger) == 8
    threshold = pd.read_csv(second["threshold_audit_path"])
    assert len(threshold) == 5
    branch = pd.read_csv(second["branch_audit_path"])
    assert len(branch) == 1

    table = second["claim_evidence_ledger_table_tex"].read_text(encoding="utf-8")
    assert "catalog\\_dro\\_tail\\_coast\\_all\\_one\\_two\\_segment\\_t5\\_all\\_configured" in table
    assert "No sampled-method or QAOA superiority" in table
