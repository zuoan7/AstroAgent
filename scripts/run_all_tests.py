#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "test_reports"


def ensure_reports_dir():
    REPORTS_DIR.mkdir(exist_ok=True)


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    duration = time.time() - start

    return {
        "description": description,
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "duration_sec": round(duration, 2),
        "stdout": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
        "stderr": result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr,
        "success": result.returncode == 0,
    }


def run_pytest_suite(path, description, report_name, with_coverage=False):
    cmd = [
        sys.executable, "-m", "pytest",
        path, "-v",
        "--tb=short",
        f"--junitxml={REPORTS_DIR}/{report_name}_junit.xml",
        "-x",
    ]
    if with_coverage:
        cmd.extend([
            "--cov=src",
            f"--cov-report=xml:{REPORTS_DIR}/{report_name}_coverage.xml",
            "--cov-report=term-missing",
        ])
    return run_command(cmd, description)


def run_unit_tests():
    return run_command(
        [sys.executable, "-m", "pytest",
         "tests/unit/", "-v",
         "--tb=short",
         f"--junitxml={REPORTS_DIR}/unit_junit.xml",
         f"--cov=src", f"--cov-report=xml:{REPORTS_DIR}/unit_coverage.xml",
         f"--cov-report=term-missing",
         "-x"],
        "单元测试"
    )


def run_regression_tests():
    return run_pytest_suite(
        "tests/regression/",
        "回归兼容测试",
        "regression",
        with_coverage=True,
    )


def run_integration_tests():
    return run_pytest_suite(
        "tests/integration/",
        "集成测试",
        "integration",
        with_coverage=True,
    )


def run_boundary_tests():
    return run_pytest_suite(
        "tests/boundary/",
        "边界测试",
        "boundary",
        with_coverage=False,
    )


def run_performance_tests():
    return run_pytest_suite(
        "tests/performance/",
        "性能测试",
        "performance",
        with_coverage=False,
    )


def run_evaluation_tests():
    return run_pytest_suite(
        "tests/evaluation/",
        "评估测试",
        "evaluation",
        with_coverage=False,
    )


def generate_report(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_tests = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total_tests - passed
    total_duration = sum(r["duration_sec"] for r in results)

    report = {
        "report_info": {
            "timestamp": timestamp,
            "project": "AstroAgent",
            "version": "1.0.0",
        },
        "summary": {
            "total_suites": total_tests,
            "passed_suites": passed,
            "failed_suites": failed,
            "total_duration_sec": round(total_duration, 2),
            "pass_rate": f"{passed/total_tests*100:.1f}%" if total_tests > 0 else "0%",
        },
        "suites": [],
    }

    for r in results:
        suite = {
            "name": r["description"],
            "command": r["command"],
            "success": r["success"],
            "return_code": r["return_code"],
            "duration_sec": r["duration_sec"],
        }

        if not r["success"]:
            suite["error_preview"] = r["stderr"][-1000:] if r["stderr"] else ""

        report["suites"].append(suite)

    report_path = REPORTS_DIR / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report, report_path)
    return report


def print_report(report, report_path):
    print("\n" + "=" * 70)
    print("  天文Agent 测试报告")
    print("=" * 70)

    info = report["report_info"]
    print(f"\n  项目: {info['project']} v{info['version']}")
    print(f"  时间: {info['timestamp']}")

    summary = report["summary"]
    print(f"\n  测试套件总数: {summary['total_suites']}")
    print(f"  通过套件数:   {summary['passed_suites']}")
    print(f"  失败套件数:   {summary['failed_suites']}")
    print(f"  通过率:       {summary['pass_rate']}")
    print(f"  总耗时:       {summary['total_duration_sec']}s")

    print(f"\n  各套件详情:")
    print(f"  {'-'*60}")

    for suite in report["suites"]:
        status = "✅ 通过" if suite["success"] else "❌ 失败"
        print(f"  {status} | {suite['name']} | {suite['duration_sec']}s")

        if not suite["success"] and suite.get("error_preview"):
            print(f"         错误预览: {suite['error_preview'][:200]}")

    print(f"\n  报告已保存至: {report_path}")
    print("=" * 70)


def main():
    ensure_reports_dir()

    print("🚀 天文Agent 全面测试套件")
    print(f"   项目根目录: {PROJECT_ROOT}")
    print(f"   报告目录:   {REPORTS_DIR}")

    results = []

    test_suites = [
        ("1", "unit", "单元测试", run_unit_tests),
        ("2", "regression", "回归兼容测试", run_regression_tests),
        ("3", "integration", "集成测试", run_integration_tests),
        ("4", "boundary", "边界测试", run_boundary_tests),
        ("5", "performance", "性能测试", run_performance_tests),
        ("6", "evaluation", "评估测试", run_evaluation_tests),
    ]

    if len(sys.argv) > 1:
        selected = set(sys.argv[1:])
        test_suites = [
            (n, key, d, f)
            for n, key, d, f in test_suites
            if n in selected or key in selected
        ]
    else:
        test_suites = [
            (n, key, d, f)
            for n, key, d, f in test_suites
            if key in {"unit", "regression", "integration"}
        ]

    for num, key, desc, func in test_suites:
        result = func()
        results.append(result)

    report = generate_report(results)

    if report["summary"]["failed_suites"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
