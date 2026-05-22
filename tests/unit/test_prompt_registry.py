import pytest

from src.agent.prompts import PromptRenderError, PromptRenderer, get_prompt_renderer


def test_react_prompt_keeps_langchain_placeholders():
    prompt = get_prompt_renderer().render("react.main")

    assert "{input}" in prompt
    assert "{agent_scratchpad}" in prompt
    assert "{tools}" in prompt
    assert "天文助手" in prompt


def test_budgeted_prompt_preserves_required_query():
    prompt = get_prompt_renderer().render_sections(
        "direct.simple_qa",
        {
            "query": "唯一重要的问题",
            "user_profile": "画像" * 500,
            "chat_history": "历史" * 500,
            "rag_context": "知识" * 5000,
        },
    )

    assert "唯一重要的问题" in prompt
    assert len(prompt) <= 6000


def test_missing_required_variable_fails():
    with pytest.raises(PromptRenderError):
        get_prompt_renderer().render("direct.no_tool_answer", {})


def test_manifest_can_be_loaded_from_custom_path(tmp_path):
    root = tmp_path / "prompts"
    root.mkdir()
    manifest = root / "manifest.yaml"
    (root / "body.txt").write_text("Hello {{ name }}", encoding="utf-8")
    manifest.write_text(
        "prompts:\n"
        "  demo:\n"
        "    version: v1\n"
        "    template: body.txt\n"
        "    required_vars: [name]\n",
        encoding="utf-8",
    )

    renderer = PromptRenderer(str(manifest))

    assert renderer.render("demo", {"name": "AstroAgent"}) == "Hello AstroAgent"
    assert renderer.version("demo") == "v1"
