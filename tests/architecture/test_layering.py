from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def _python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def _assert_no_forbidden_imports(root: Path, forbidden_prefixes: tuple[str, ...]):
    violations: list[str] = []
    for path in _python_files(root):
        for module in _imports(path):
            if module.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert violations == []


def _assert_text_absent(path: Path, patterns: tuple[str, ...]):
    text = path.read_text(encoding="utf-8")
    violations = [
        f"{path.relative_to(REPO_ROOT)} references {pattern}"
        for pattern in patterns
        if pattern in text
    ]
    assert violations == []


def _assert_token_absent_under(root: Path, token: str):
    violations: list[str] = []
    for path in _python_files(root):
        text = path.read_text(encoding="utf-8")
        if token in text:
            violations.append(f"{path.relative_to(REPO_ROOT)} references {token}")
    assert violations == []


def test_tools_layer_does_not_import_upper_layers():
    _assert_no_forbidden_imports(
        REPO_ROOT / "src" / "tools",
        ("src.skills", "src.agent"),
    )


def test_skills_layer_does_not_import_agent_layer():
    _assert_no_forbidden_imports(
        REPO_ROOT / "src" / "skills",
        ("src.agent",),
    )


def test_capabilities_layer_does_not_import_agent_layer():
    _assert_no_forbidden_imports(
        REPO_ROOT / "src" / "capabilities",
        ("src.agent",),
    )


def test_transport_layer_does_not_import_upper_layers():
    _assert_no_forbidden_imports(
        REPO_ROOT / "src" / "transport",
        ("src.skills", "src.tools", "src.agent"),
    )


def test_transport_and_tools_do_not_define_legacy_tool_call_names():
    forbidden_names = {
        "call_" + "tool",
        "async_" + "call_" + "tool",
        "call_" + "tools_" + "parallel",
    }
    violations: list[str] = []

    for root in (REPO_ROOT / "src" / "tools", REPO_ROOT / "src" / "transport"):
        for path in _python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name in forbidden_names:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} defines {node.name}"
                    )

    assert violations == []


def test_skill_handlers_do_not_import_transport():
    _assert_no_forbidden_imports(
        REPO_ROOT / "src" / "skills" / "handlers",
        ("src.transport",),
    )


def test_upper_layers_do_not_parse_mcp_envelopes_directly():
    for root in (
        REPO_ROOT / "src" / "agent",
        REPO_ROOT / "src" / "skills",
        REPO_ROOT / "src" / "capabilities",
    ):
        _assert_no_forbidden_imports(root, ("src.transport.mcp.envelope",))


def test_upper_layers_and_tests_do_not_reference_raw_tool_envelope_storage():
    forbidden_token = "raw_" + "envelope"
    for root in (
        REPO_ROOT / "src" / "agent",
        REPO_ROOT / "src" / "skills",
        REPO_ROOT / "src" / "capabilities",
        REPO_ROOT / "tests",
    ):
        _assert_token_absent_under(root, forbidden_token)


def test_toolkit_does_not_expose_legacy_runtime_api():
    kit_path = REPO_ROOT / "src" / "tools" / "kit.py"
    tree = ast.parse(kit_path.read_text(encoding="utf-8"))

    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    legacy_runtime_name = "Tool" + "Runtime"
    assert legacy_runtime_name not in class_names
    assert (
        not {
            "with_context",
            "call_tool",
            "async_call_tool",
            "call_tools_parallel",
        }
        & function_names
    )


def test_tool_registry_and_definition_do_not_export_legacy_schema_api():
    _assert_text_absent(
        REPO_ROOT / "src" / "tools" / "registry.py",
        ("TOOL_INPUT" + "_MODELS", "TOOL_OUTPUT" + "_MODELS"),
    )
    _assert_text_absent(
        REPO_ROOT / "src" / "tools" / "definition.py",
        ("param" + "_names", "legacy callers"),
    )


def test_skill_definition_does_not_export_legacy_input_field_api():
    _assert_text_absent(
        REPO_ROOT / "src" / "skills" / "definition.py",
        ("param" + "_names", "legacy callers"),
    )


def test_tool_registry_does_not_define_schema_models():
    registry_path = REPO_ROOT / "src" / "tools" / "registry.py"
    tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    schema_bases = {"BaseModel", "RootModel"}
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in schema_bases:
                violations.append(node.name)
            elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                if base.value.id in schema_bases:
                    violations.append(node.name)

    assert violations == []


def test_legacy_import_paths_are_removed_from_code_and_tests():
    legacy_imports = tuple(
        "src." + suffix
        for suffix in (
            "core.mcp_protocol",
            "skills.mcp_client",
            "agent.skill_manager",
            "skills.router",
            "skills.executor",
            "skills.skill_handlers",
            "tools.catalog",
            "tools.runtime",
            "capabilities.registry",
            "agent.models.skill_result",
            "agent.models.capability_decision",
            "agent.param_parser",
        )
    )
    violations: list[str] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "tests"):
        for path in _python_files(root):
            if path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            for legacy_import in legacy_imports:
                if legacy_import in text:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} references {legacy_import}"
                    )
    assert violations == []
