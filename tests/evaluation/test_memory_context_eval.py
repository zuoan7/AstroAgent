from pathlib import Path

from scripts.evaluation.evaluate_memory_context import run_evaluation


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
