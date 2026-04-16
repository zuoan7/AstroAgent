from unittest.mock import MagicMock, patch
import re

from src.agent.skill_manager import SkillManager
from src.skills.registry import SkillSpec
from src.skills.router import AstronomySkillRouter
from src.skills import registry


class FakeTool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description


def test_skill_sets_are_consistent():
    router = AstronomySkillRouter()
    registry.validate_skill_registry(handler_names=set(router._handlers.keys()))

    registry_skill_names = {spec.skill_name for spec in registry.get_skill_specs()}
    router_skill_names = set(router.list_skills().keys())
    assert router_skill_names == registry_skill_names

    with patch("src.agent.skill_manager.AstronomySkillRouter") as mock_router_cls, \
         patch("src.agent.skill_manager.Tool", FakeTool):
        mock_router = MagicMock()
        mock_router_cls.return_value = mock_router

        manager = SkillManager()
        tool_names = {tool.name for tool in manager.get_langchain_tools()}

    registry_tool_names = {spec.langchain_tool_name for spec in registry.get_skill_specs()}
    assert tool_names - {"RAGRetrieve"} == registry_tool_names

    router.shutdown()


def test_weather_lookup_param_compatibility():
    with patch("src.agent.skill_manager.AstronomySkillRouter") as mock_router_cls, \
         patch("src.agent.skill_manager.Tool", FakeTool):
        mock_router = MagicMock()
        mock_router.call.return_value = "ok"
        mock_router_cls.return_value = mock_router

        manager = SkillManager()
        tools = {tool.name: tool for tool in manager.get_langchain_tools()}
        weather_tool = tools["WeatherLookup"]

        assert weather_tool.func('{"city": "北京"}') == "ok"
        assert weather_tool.func('{"location": "北京"}') == "ok"
        assert weather_tool.func('{"city": "北京", "extensions": "all"}') == "ok"

    assert mock_router.call.call_args_list[0].kwargs == {
        "city": "北京",
        "location": None,
        "extensions": "all",
    }
    assert mock_router.call.call_args_list[1].kwargs == {
        "city": "北京",
        "location": None,
        "extensions": "all",
    }
    assert mock_router.call.call_args_list[2].kwargs == {
        "city": "北京",
        "location": None,
        "extensions": "all",
    }


def test_simple_skill_returns_raw_without_truncation():
    router = AstronomySkillRouter()
    raw = "x" * 2400

    with patch.object(router, "call_mcp_tool", return_value=raw) as mock_call:
        result = router.call("weather-lookup", city="北京")

    assert result == raw
    mock_call.assert_called_once_with("get_weather", city="北京")
    router.shutdown()


def test_registry_only_change_can_register_simple_skill(monkeypatch):
    fake_spec = SkillSpec(
        skill_name="fake-skill",
        langchain_tool_name="FakeSkill",
        summary="fake summary",
        description="fake description",
        route_type="simple",
        mcp_tool_name="fake_tool",
        param_names=["query"],
    )
    original_specs = registry.get_skill_specs()

    monkeypatch.setattr(
        registry,
        "get_skill_specs",
        lambda: original_specs + [fake_spec],
    )

    router = AstronomySkillRouter()
    assert "fake-skill" in router.list_skills()

    with patch.object(router, "call_mcp_tool", return_value="fake-result") as mock_call:
        assert router.call("fake-skill", query="demo") == "fake-result"
        mock_call.assert_called_once_with("fake_tool", query="demo")

    with patch("src.agent.skill_manager.AstronomySkillRouter") as mock_router_cls, \
         patch("src.agent.skill_manager.Tool", FakeTool):
        mock_router_cls.return_value = MagicMock()
        manager = SkillManager()
        tool_names = {tool.name for tool in manager.get_langchain_tools()}

    assert "FakeSkill" in tool_names
    router.shutdown()


def test_readme_skill_table_matches_registry():
    with open("/home/gmy/AstroAgent/README.md", "r", encoding="utf-8") as f:
        readme = f.read()

    match = re.search(
        r"### 高层技能集合\s+当前系统向 Agent 暴露的高层技能如下：\s+(.*?)\n## ",
        readme,
        re.DOTALL,
    )
    assert match is not None

    readme_tool_names = set(re.findall(r"\| `([^`]+)` \|", match.group(1)))
    assert readme_tool_names == {"RAGRetrieve", *registry.list_langchain_tool_names()}


def test_validate_skill_registry_rejects_handler_mismatch():
    fake_handler_spec = SkillSpec(
        skill_name="missing-handler-skill",
        langchain_tool_name="MissingHandlerSkill",
        summary="missing handler",
        description="missing handler description",
        route_type="handler",
    )

    try:
        registry.validate_skill_registry(specs=[fake_handler_spec], handler_names=set())
    except ValueError as exc:
        assert "未在 SKILL_HANDLERS 中注册" in str(exc)
    else:
        raise AssertionError("expected registry validation to fail")
