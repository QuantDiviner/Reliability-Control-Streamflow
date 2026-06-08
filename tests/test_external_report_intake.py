import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / ".codex" / "skills" / "research-navigator" / "scripts"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


intake = load_module("external_report_intake_test", "external_report_intake.py")
arbiter = load_module("fpa_arbiter_test", "fpa_arbiter.py")


def test_required_field_report_valid(tmp_path):
    report = tmp_path / "draft.md"
    json_out = tmp_path / "draft.json"
    report.write_text(
        "\n".join(
            [
                "# Revision Plan",
                "",
                "Recommendation: ACCEPT",
                "Estimated_Effort: LOW",
                "Confidence: HIGH",
                "",
                "Details: " + "x" * 260,
            ]
        ),
        encoding="utf-8",
    )

    ok, result, exit_code = intake.validate_by_schema(
        "fpa_revision_plan_draft",
        report,
        json_out,
        [],
        200,
    )

    assert ok is True
    assert exit_code == 0
    assert result["status"] == "VALID"
    assert json_out.exists()


def test_missing_raw_report_is_not_repaired(tmp_path, monkeypatch):
    missing = tmp_path / "missing.md"
    json_out = tmp_path / "missing.json"

    def fail_repair(*_args, **_kwargs):
        raise AssertionError("raw missing reports must not enter repair")

    monkeypatch.setattr(intake, "run_repair", fail_repair)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_report_intake.py",
            "--report",
            str(missing),
            "--schema",
            "fpa_revision_plan_draft",
            "--repair-if-needed",
            "--json-out",
            str(json_out),
        ],
    )

    assert intake.main() == 1
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "RAW_MISSING"
    assert payload["intake"]["repair_attempts"] == []


def test_schema_repair_is_validated_and_written_in_place(tmp_path, monkeypatch):
    report = tmp_path / "draft.md"
    json_out = tmp_path / "draft.json"
    report.write_text(
        "\n".join(
            [
                "# Revision Plan",
                "",
                "Recommendation: ACCEPT",
                "Effort: LOW",
                "Confidence: HIGH",
                "",
                "Details: " + "x" * 260,
            ]
        ),
        encoding="utf-8",
    )

    def fake_repair(schema, report_path, validation, required_fields, attempt, reviewer, decision_id):
        report_path.write_text(
            report_path.read_text(encoding="utf-8").replace("Effort: LOW", "Estimated_Effort: LOW"),
            encoding="utf-8",
        )
        return {
            "attempt": attempt,
            "ok": True,
            "message": "fixture repair",
            "provider": "codex",
            "tier": "3",
            "model": "fixture",
            "effort": "high",
            "repair_mode": "in_place_report_path",
            "repair_output_path": str(report_path),
            "repair_output_sha256": intake.sha256_file(report_path),
            "reviewer": reviewer,
            "decision_id": decision_id,
        }

    monkeypatch.setattr(intake, "run_repair", fake_repair)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_report_intake.py",
            "--report",
            str(report),
            "--schema",
            "fpa_revision_plan_draft",
            "--reviewer",
            "codex",
            "--decision-id",
            "unit",
            "--repair-if-needed",
            "--json-out",
            str(json_out),
        ],
    )

    assert intake.main() == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "VALID"
    assert ".raw." in payload["intake"]["raw_backup_path"]
    assert payload["intake"]["repair_attempts"][0]["post_validation_status"] == "VALID"
    assert payload["intake"]["repair_attempts"][0]["repair_output_path"] == str(report)
    assert not list(tmp_path.glob("*.repair*.md"))
    assert "Estimated_Effort: LOW" in report.read_text(encoding="utf-8")


def test_failed_repair_exhausts_without_overwriting_raw_report(tmp_path, monkeypatch):
    report = tmp_path / "draft.md"
    json_out = tmp_path / "draft.json"
    original = "\n".join(
        [
            "# Revision Plan",
            "",
            "Recommendation: ACCEPT",
            "Effort: LOW",
            "Confidence: HIGH",
            "",
            "Details: " + "x" * 260,
        ]
    )
    report.write_text(original, encoding="utf-8")

    def fake_repair(schema, report_path, validation, required_fields, attempt, reviewer, decision_id):
        report_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "attempt": attempt,
            "ok": True,
            "message": "fixture invalid repair",
            "provider": "codex",
            "tier": "3",
            "model": "fixture",
            "effort": "high",
            "repair_mode": "in_place_report_path",
            "repair_output_path": str(report_path),
            "repair_output_sha256": intake.sha256_file(report_path),
            "reviewer": reviewer,
            "decision_id": decision_id,
        }

    monkeypatch.setattr(intake, "run_repair", fake_repair)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_report_intake.py",
            "--report",
            str(report),
            "--schema",
            "fpa_revision_plan_draft",
            "--reviewer",
            "codex",
            "--decision-id",
            "unit",
            "--repair-if-needed",
            "--json-out",
            str(json_out),
        ],
    )

    assert intake.main() == 2
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "FORMAT_BLOCKED_RETRY_EXHAUSTED"
    assert payload["intake"]["repair_attempts"][0]["post_validation_status"] == "SCHEMA_FAILED"
    assert report.read_text(encoding="utf-8") == original
    assert Path(payload["intake"]["raw_backup_path"]).read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.repair*.md"))


def test_raw_backup_path_changes_when_raw_content_changes(tmp_path, monkeypatch):
    report = tmp_path / "draft.md"
    json_out = tmp_path / "draft.json"

    def fake_repair(schema, report_path, validation, required_fields, attempt, reviewer, decision_id):
        report_path.write_text(
            report_path.read_text(encoding="utf-8").replace("Effort:", "Estimated_Effort:"),
            encoding="utf-8",
        )
        return {
            "attempt": attempt,
            "ok": True,
            "message": "fixture repair",
            "provider": "codex",
            "tier": "3",
            "model": "fixture",
            "effort": "high",
            "repair_mode": "in_place_report_path",
            "repair_output_path": str(report_path),
            "repair_output_sha256": intake.sha256_file(report_path),
            "reviewer": reviewer,
            "decision_id": decision_id,
        }

    monkeypatch.setattr(intake, "run_repair", fake_repair)

    backup_paths = []
    for effort in ("LOW", "MEDIUM"):
        raw = "\n".join(
            [
                "# Revision Plan",
                "",
                "Recommendation: ACCEPT",
                f"Effort: {effort}",
                "Confidence: HIGH",
                "",
                "Details: " + "x" * 260,
            ]
        )
        report.write_text(raw, encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "external_report_intake.py",
                "--report",
                str(report),
                "--schema",
                "fpa_revision_plan_draft",
                "--reviewer",
                "codex",
                "--decision-id",
                f"unit_{effort}",
                "--repair-if-needed",
                "--json-out",
                str(json_out),
            ],
        )
        assert intake.main() == 0
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        backup_paths.append(Path(payload["intake"]["raw_backup_path"]))

    assert backup_paths[0] != backup_paths[1]
    assert "Effort: LOW" in backup_paths[0].read_text(encoding="utf-8")
    assert "Effort: MEDIUM" in backup_paths[1].read_text(encoding="utf-8")


def test_fpa_policy_failure_blocks_repair(tmp_path, monkeypatch):
    report = tmp_path / "claude.md"
    json_out = tmp_path / "claude.json"
    report.write_text("# FPA\n\n" + "x" * 260, encoding="utf-8")

    def fake_validate(schema, report_path, json_path, required_fields, min_bytes):
        assert schema == "fpa_layer1"
        return False, {"status": "POLICY_FAILED", "output_zone_violations": ["presentation-only"]}, 2

    def fail_repair(*_args, **_kwargs):
        raise AssertionError("FPA policy failures must not be auto-repaired")

    monkeypatch.setattr(intake, "validate_by_schema", fake_validate)
    monkeypatch.setattr(intake, "run_repair", fail_repair)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_report_intake.py",
            "--report",
            str(report),
            "--schema",
            "fpa_layer1",
            "--reviewer",
            "claude",
            "--decision-id",
            "unit",
            "--repair-if-needed",
            "--json-out",
            str(json_out),
        ],
    )

    assert intake.main() == 2
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "UNRECOVERABLE_POLICY_VIOLATION"
    assert payload["intake"]["repair_attempts"] == []


def test_fpa_intake_rejects_repair_side_report_path(tmp_path, monkeypatch):
    report = tmp_path / "20260526_000000_unit_fpa_R1_claude.format_repaired.md"
    json_out = tmp_path / "claude.json"
    report.write_text("# FPA\n\nRecommendation: A\n\n" + "x" * 260, encoding="utf-8")

    def fail_repair(*_args, **_kwargs):
        raise AssertionError("FPA repair side reports must not enter repair")

    monkeypatch.setattr(intake, "run_repair", fail_repair)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_report_intake.py",
            "--report",
            str(report),
            "--schema",
            "fpa_layer1",
            "--reviewer",
            "claude",
            "--decision-id",
            "unit",
            "--repair-if-needed",
            "--json-out",
            str(json_out),
        ],
    )

    assert intake.main() == 2
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "BANNED_FPA_REPAIR_PATH"
    assert payload["intake"]["report_path"] == str(report)
    assert payload["intake"]["repair_attempts"] == []


def test_strict_fpa_parser_rejects_repair_side_report_path(tmp_path):
    report = tmp_path / "20260526_000000_unit_fpa_R1_claude.repair1.md"
    report.write_text("# FPA\n\nRecommendation: A\n\n" + "x" * 260, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "parse_ai_report.py"),
            str(report),
            "--strict-fpa-policy",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    assert "format-repair side document" in proc.stderr


def test_fpa_arbiter_prefers_intake_reviewer_metadata_and_raw_report_path(tmp_path, monkeypatch, capsys):
    parsed = tmp_path / "reused_artifact.json"
    raw_report = tmp_path / "20260526_000000_unit_fpa_R1_codex.md"
    raw_report.write_text("# FPA Report\n\n" + "x" * 260, encoding="utf-8")
    parsed.write_text(
        json.dumps(
            {
                "recommendation": "A",
                "fix_type": "TEXT",
                "revision_items": [],
                "output_zone_violations": [],
                "q2_schema_errors": [],
                "intake": {"reviewer": "codex", "report_path": str(raw_report)},
            }
        ),
        encoding="utf-8",
    )
    verdicts = tmp_path / "verdicts.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fpa_arbiter.py",
            "--reports",
            str(parsed),
            "--verdicts",
            str(verdicts),
            "--required-reviewers",
            "codex",
            "--chain-exit",
            "A",
            "--dry-run",
        ],
    )

    arbiter.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["individual_ratings"] == {"codex": "A"}
    assert payload["report_paths"] == [str(raw_report)]
    assert payload["parsed_artifact_paths"] == [str(parsed)]


def write_reused_fpa_artifact(
    path: Path,
    reviewer: str = "codex",
    status: str = "VALID",
    decision_id: str = "reuse_guard_unit",
    raw_text: str | None = None,
):
    raw_report = path.with_suffix(".md")
    raw_report.write_text(raw_text or ("# FPA Report\n\n" + "x" * 260), encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "recommendation": "A",
                "fix_type": "TEXT",
                "confidence": "HIGH",
                "revision_items": [],
                "output_zone_violations": [],
                "q2_schema_errors": [],
                "intake": {
                    "status": status,
                    "reviewer": reviewer,
                    "decision_schema": "fpa_layer1",
                    "decision_id": decision_id,
                    "report_path": str(raw_report),
                    "report_sha256": intake.sha256_file(raw_report),
                },
            }
        ),
        encoding="utf-8",
    )


def run_fpa_chain_for_reuse_test(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    fake_ai = tmp_path / "fake_ai.sh"
    fake_ai.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_ai.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "AI_DECIDE_OVERRIDE": str(fake_ai),
            "FPA_ARBITER_DRY_RUN": "true",
            "FPA_VERDICTS_PATH": str(tmp_path / "verdicts.jsonl"),
        }
    )
    return subprocess.run(
        [
            "bash",
            str(SCRIPTS_DIR / "fpa_review_chain.sh"),
            "--decision-id",
            "reuse_guard_unit",
            "--output-dir",
            str(tmp_path),
            *args,
            "--",
            "UNIT=1",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_fpa_chain_rejects_reused_artifact_reviewer_mismatch(tmp_path):
    parsed = tmp_path / "reused.json"
    write_reused_fpa_artifact(parsed, reviewer="claude")

    proc = run_fpa_chain_for_reuse_test(
        tmp_path,
        "--retry-reviewer",
        "claude",
        "--reuse-parsed",
        f"codex={parsed}",
    )

    assert proc.returncode != 0
    assert "reviewer mismatch" in proc.stderr
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_fpa_chain_rejects_reuse_of_retry_reviewer(tmp_path):
    parsed = tmp_path / "reused.json"
    write_reused_fpa_artifact(parsed, reviewer="claude")

    proc = run_fpa_chain_for_reuse_test(
        tmp_path,
        "--retry-reviewer",
        "claude",
        "--reuse-parsed",
        f"claude={parsed}",
    )

    assert proc.returncode != 0
    assert "also --retry-reviewer" in proc.stderr
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_fpa_chain_rejects_non_valid_reused_artifact(tmp_path):
    parsed = tmp_path / "reused.json"
    write_reused_fpa_artifact(parsed, reviewer="codex", status="FORMAT_BLOCKED")

    proc = run_fpa_chain_for_reuse_test(
        tmp_path,
        "--retry-reviewer",
        "claude",
        "--reuse-parsed",
        f"codex={parsed}",
    )

    assert proc.returncode != 0
    assert "was not accepted by external_report_intake.py" in proc.stderr
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_fpa_chain_rejects_reused_artifact_with_repair_side_report_path(tmp_path):
    parsed = tmp_path / "20260526_000000_unit_fpa_R1_codex.format_repaired.json"
    write_reused_fpa_artifact(parsed, reviewer="codex")

    proc = run_fpa_chain_for_reuse_test(
        tmp_path,
        "--retry-reviewer",
        "claude",
        "--reuse-parsed",
        f"codex={parsed}",
    )

    assert proc.returncode != 0
    assert "format-repair side document" in proc.stderr
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_fpa_chain_rejects_reused_artifact_decision_mismatch(tmp_path):
    parsed = tmp_path / "reused.json"
    write_reused_fpa_artifact(parsed, reviewer="codex", decision_id="other_decision")

    proc = run_fpa_chain_for_reuse_test(
        tmp_path,
        "--retry-reviewer",
        "claude",
        "--reuse-parsed",
        f"codex={parsed}",
    )

    assert proc.returncode != 0
    assert "decision mismatch" in proc.stderr
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_fpa_chain_rejects_reused_artifact_raw_hash_mismatch(tmp_path):
    parsed = tmp_path / "reused.json"
    raw = parsed.with_suffix(".md")
    write_reused_fpa_artifact(parsed, reviewer="codex")
    raw.write_text("# FPA Report\n\nchanged " + "x" * 260, encoding="utf-8")

    proc = run_fpa_chain_for_reuse_test(
        tmp_path,
        "--retry-reviewer",
        "claude",
        "--reuse-parsed",
        f"codex={parsed}",
    )

    assert proc.returncode != 0
    assert "raw report hash mismatch" in proc.stderr
    assert not (tmp_path / "verdicts.jsonl").exists()
