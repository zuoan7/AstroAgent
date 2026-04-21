#!/usr/bin/env python3
"""Guard against reintroducing removed memory callers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED_SESSION_MODULE = "short" + "_term" + "_memory"
REMOVED_SESSION_CLASS = "Short" + "Term" + "Memory"
REMOVED_LONG_TERM_MANAGER = "Long" + "Term" + "Memory" + "Manager"
REMOVED_LONG_TERM_MANAGER_MODULE = "src.memory.long" + "_term" + "_memory.manager"
FORBIDDEN_FILES = [
    ROOT / "src/memory" / REMOVED_SESSION_MODULE,
    ROOT / "src/memory/adapters",
    ROOT / "src/memory/adapters" / f"{REMOVED_SESSION_MODULE}_adapter.py",
    ROOT / "src/memory/context_builder.py",
    ROOT / "src/memory/summarizer.py",
    ROOT / "src/memory/long_term_memory/manager.py",
]
FORBIDDEN_PATTERNS = [
    REMOVED_SESSION_CLASS,
    f"src.memory.{REMOVED_SESSION_MODULE}",
    REMOVED_LONG_TERM_MANAGER,
    REMOVED_LONG_TERM_MANAGER_MODULE,
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
