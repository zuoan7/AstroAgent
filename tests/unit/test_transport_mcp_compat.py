from __future__ import annotations

import ast
from pathlib import Path

from src.transport.mcp.client import MCPClient, _AsyncBridge
from src.transport.mcp.sse import parse_sse_response


def test_mcp_client_symbols_live_in_transport_layer():
    assert MCPClient.__module__ == "src.transport.mcp.client"
    assert _AsyncBridge.__module__ == "src.transport.mcp.client"


def test_mcp_client_exposes_invoke_backend_contract():
    client = MCPClient()

    assert callable(client.invoke)
    assert callable(client.ainvoke)
    assert callable(client.invoke_parallel)
    assert callable(client.ainvoke_parallel)


def test_parse_sse_response_reads_first_json_data_line():
    payload = parse_sse_response(
        "event: message\n" 'data: {"jsonrpc": "2.0", "result": {"status": "ok"}}\n\n'
    )

    assert payload == {"jsonrpc": "2.0", "result": {"status": "ok"}}


def test_parse_sse_response_returns_none_for_invalid_payload():
    assert parse_sse_response("event: ping\n\n") is None
    assert parse_sse_response("data: {not-json}") is None


def test_transport_mcp_does_not_import_upper_layers():
    forbidden_prefixes = ("src.skills", "src.tools", "src.agent")
    transport_root = Path(__file__).parents[2] / "src" / "transport" / "mcp"

    for path in transport_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)

        assert not [
            module
            for module in imported_modules
            if module.startswith(forbidden_prefixes)
        ], f"{path} imports forbidden upper-layer modules"
