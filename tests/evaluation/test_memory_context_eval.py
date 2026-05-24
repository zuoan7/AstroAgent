import subprocess
import sys
from pathlib import Path

from scripts.evaluation.evaluate_memory_context import run_evaluation
from scripts.evaluation.evaluate_memory_context_v2 import run_evaluation as run_v2_evaluation


def test_memory_context_eval_writes_reports(tmp_path):
    dataset = Path("data/eval/memory/memory_context_eval_v1.json")
    output_root = tmp_path / "memory_context_reports"

    report = run_evaluation(
        dataset_path=dataset,
        output_root=output_root,
        tenant_id="memory_eval_test",
        max_tokens=4000,
    )

    metrics_path = Path(report["metrics_path"])
    summary_path = Path(report["summary_path"])
    overall = report["overall_metrics"]

    assert metrics_path.exists()
    assert summary_path.exists()
    assert overall["scenario_count"] == 10
    assert 0.0 <= overall["memory_hit_rate"] <= 1.0
    assert "tool_evidence_reuse_rate" in overall
    assert report["per_scenario"][0]["context_text"]


def test_memory_context_eval_v2_meets_current_functional_gates(tmp_path):
    dataset = Path("data/eval/memory/memory_context_eval_v2.json")
    output_root = tmp_path / "memory_context_v2_reports"

    report = run_v2_evaluation(
        dataset_path=dataset,
        output_root=output_root,
        tenant_id="memory_eval_v2_test",
        budgets=[1000],
    )

    overall = report["overall_metrics"]
    assert Path(report["metrics_path"]).exists()
    assert Path(report["summary_path"]).exists()
    assert overall["scenario_count"] == 12
    assert overall["memory_hit_rate"] == 1.0
    assert overall["tool_evidence_reuse_rate"] == 1.0
    assert overall["paraphrase_hit_rate"] == 1.0
    assert overall["fresh_evidence_primary_rate"] == 1.0
    assert overall["wrong_tool_evidence_injection_rate"] == 0.0
    assert overall["stale_tool_evidence_present_rate"] == 0.0
    assert overall["harmful_wrong_injection_rate"] <= 0.25
    assert overall["harmful_stale_evidence_present_rate"] <= 0.25


def test_memory_context_eval_v2_cli_runs_without_import_cycle(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/evaluate_memory_context_v2.py",
            "--output-root",
            str(tmp_path / "cli_reports"),
            "--tenant-id",
            "memory_eval_v2_cli_test",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Memory context v2 stress evaluation complete" in result.stdout
    assert "tool_evidence_reuse_rate: 100.00%" in result.stdout
