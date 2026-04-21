#!/usr/bin/env python3
"""Guard against reintroducing legacy short-term memory callers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FILES = [
    ROOT / "src/memory/short_term_memory",
    ROOT / "src/memory/adapters",
    ROOT / "src/memory/adapters/short_term_memory_adapter.py",
    ROOT / "src/memory/context_builder.py",
    ROOT / "src/memory/summarizer.py",
]
FORBIDDEN_PATTERNS = [
    "ShortTermMemory",
    "src.memory.short_term_memory",
    ".add_message(",
    ".add_tool_call(",
    ".get_context(",
    ".get_recent_messages(",
]
CHECK_PATHS = [
    ROOT / "src/memory",
    ROOT / "src/agent",
    ROOT / "src/api",
    ROOT / "tests",
]


def iter_python_files(path: Path):
    if path.is_file() and path.suffix == ".py":
        yield path
        return
    for item in path.rglob("*.py"):
        yield item


def main() -> int:
    failures: list[str] = []
    for path in FORBIDDEN_FILES:
        if path.exists():
            failures.append(f"legacy file still exists: {path.relative_to(ROOT)}")

    for base in CHECK_PATHS:
        for path in iter_python_files(base):
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in text:
                    failures.append(
                        f"{path.relative_to(ROOT)} contains forbidden pattern {pattern!r}"
                    )

    if failures:
        print("Memory migration guard failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Memory migration guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
