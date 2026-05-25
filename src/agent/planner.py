"""Planned 路径规划器，按任务画像生成 ExecutionPlan/WorkflowGraph，并为 DAG 节点补齐技能或原子工具参数。"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from src.agent.models.execution_plan import ExecutionPlan
from src.capabilities.plan import PlanStep
from src.agent.models.workflow_graph import WorkflowGraph
from src.agent.prompts import get_prompt_renderer
from src.agent.skill_param_builder import SkillParamBuilder
from src.capabilities.param_builder import CapabilityParamBuilder
from src.capabilities.plan_adapter import CapabilityPlanAdapter
from src.core.config import settings
from src.skills import registry
from src.tools.registry import get_default_tool_registry
from src.tools.selector import ToolSelector


class Planner:
    """
    Planner for planned tasks.

    当前优先输出 WorkflowGraph（plan_graph），ExecutionPlan 仅保留兼容表示。
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        """初始化 Planner 的依赖、配置和内部状态。"""
        self._llm = llm
        self._param_builder = SkillParamBuilder(None)
        self._skill_registry = registry.get_default_skill_registry()
        self._tool_registry = get_default_tool_registry()
        self._capability_plan_adapter = CapabilityPlanAdapter()
        self._tool_selector = ToolSelector()

    def plan(
        self,
        *,
        query: str,
        route_decision: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> ExecutionPlan:
        """返回 ExecutionPlan 的旧版兼容入口。

        新 planned 主路径应优先使用 `plan_graph_for_profile()` 获取 WorkflowGraph；
        本方法仅为旧调用方/旧序列化输出提供兼容表示。
        """
        graph = self.plan_graph(
            query=query,
            route_decision=route_decision,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return ExecutionPlan.from_workflow_graph(
            graph,
            task_type=getattr(route_decision, "task_type", None),
        )

    def plan_graph(
        self,
        *,
        query: str,
        route_decision: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> WorkflowGraph:
        """返回 WorkflowGraph 的 planned 主路径入口。

        初版复用现有模板/通用规划逻辑，但对外直接返回 WorkflowGraph，
        使 graph 成为 planned 路径的优先计划表达。
        """
        plan = self._resolve_plan_from_fields(
            query=query,
            task_type=getattr(
                route_decision, "task_type", "observation_recommendation"
            ),
            output_schema=getattr(
                route_decision, "expected_output_schema", "generic_answer_v1"
            ),
            capability_hints=list(
                getattr(route_decision, "capability_hints", []) or []
            ),
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return WorkflowGraph.from_execution_plan(plan)

    def plan_graph_for_profile(
        self,
        *,
        query: str,
        profile: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> WorkflowGraph:
        """根据 TaskProfile 返回 WorkflowGraph 的 planned 主路径入口。"""
        plan = self._resolve_plan_from_fields(
            query=query,
            task_type=getattr(profile, "task_type", "observation_recommendation"),
            output_schema=getattr(
                profile, "expected_output_schema", "generic_answer_v1"
            ),
            capability_hints=list(getattr(profile, "capability_hints", []) or []),
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return WorkflowGraph.from_execution_plan(plan)

    def _resolve_plan_from_fields(
        self,
        *,
        query: str,
        task_type: str,
        output_schema: str,
        capability_hints: List[str],
        chat_history: str = "",
        user_profile: str = "",
    ) -> ExecutionPlan:
        """按任务字段依次尝试模板计划、通用计划和 LLM fallback 计划。"""
        plan = self._build_template_plan(
            query=query,
            task_type=task_type,
            output_schema=output_schema,
            capability_hints=capability_hints,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        if plan.steps:
            return plan

        generic_plan = self._build_generic_plan(
            query=query,
            task_type=task_type,
            output_schema=output_schema,
            capability_hints=capability_hints,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        if generic_plan.steps:
            return generic_plan

        llm_plan = self._build_llm_fallback_plan(
            query=query,
            task_type=task_type,
            output_schema=output_schema,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return llm_plan or generic_plan

    def _build_template_plan(
        self,
        *,
        query: str,
        task_type: str,
        output_schema: str,
        capability_hints: List[str],
        chat_history: str = "",
        user_profile: str = "",
    ) -> ExecutionPlan:
        """为已知任务类型构造确定性模板计划。"""
        skill_set = set(capability_hints)
        steps: List[PlanStep] = []
        rationale = ""

        def make_step(**kwargs: Any) -> PlanStep:
            """用当前查询上下文创建模板计划步骤。"""
            return self._make_step(
                query=query,
                chat_history=chat_history,
                user_profile=user_profile,
                **kwargs,
            )

        if task_type == "observation_recommendation":
            rationale = "观测推荐以 observation-planner 为聚合入口；按目标约束补充事件、深空或位置证据。"
            context_group = "observation_context"
            if self._observation_plan_needs_event_step(query):
                steps.append(
                    make_step(
                        planner_source="template",
                        id="event_context",
                        title="查询天象事件",
                        description="获取计划时段内的主要天象",
                        skill="celestial-events-forecast",
                        purpose="为多日或周末观测计划补充天象事件",
                        success_criteria="返回指定时段的天象事件摘要",
                        evidence_key="celestial_events",
                        required=False,
                        fallback_strategy="continue",
                        parallel_group=context_group,
                        retry_policy=1,
                        timeout_ms=12000,
                    )
                )
            steps.append(
                make_step(
                    planner_source="template",
                    id="observation_plan",
                    title="生成观测计划",
                    description="基于地点和时间生成可执行的观测建议",
                    skill="observation-planner",
                    purpose="生成目标、时段和实用观测安排",
                    success_criteria="返回指定日期地点的观测计划或目标建议",
                    evidence_key="observation_plan",
                    fallback_strategy="react_fallback",
                    parallel_group=context_group,
                    retry_policy=1,
                    timeout_ms=12000,
                )
            )
            if self._observation_plan_needs_deep_sky_step(query):
                steps.append(
                    make_step(
                        planner_source="template",
                        id="deep_sky_context",
                        title="补充深空目标资料",
                        description="补充深空目标资料、器材约束和观测提示",
                        skill="deep-sky-observing-guide",
                        purpose="为设备约束、深空目标或观测顺序提供目标证据",
                        success_criteria="返回目标基础信息与观测建议",
                        evidence_key="deep_sky_object",
                        required=False,
                        fallback_strategy="continue",
                        parallel_group=context_group,
                        retry_policy=1,
                        timeout_ms=15000,
                    )
                )
            if self._observation_plan_needs_position_step(query):
                steps.append(
                    make_step(
                        planner_source="template",
                        id="position_context",
                        title="补充天体位置",
                        description="补充月亮、行星或目标的时段位置",
                        skill="celestial-position-calculator",
                        purpose="为观测顺序和目标选择提供位置证据",
                        success_criteria="返回目标位置或可见性数据",
                        evidence_key="celestial_position",
                        required=False,
                        fallback_strategy="continue",
                        parallel_group=context_group,
                        retry_policy=1,
                        timeout_ms=12000,
                    )
                )
        elif task_type == "celestial_event_analysis":
            rationale = "天象分析以天象事件检索为核心，必要时补充观测条件。"
            event_group = (
                "event_context"
                if ("weather-lookup" in skill_set or "天气" in query)
                else None
            )
            steps.append(
                make_step(
                    planner_source="template",
                    id="event_forecast",
                    title="查询天象事件",
                    description="获取指定时段内的主要天象与事件信息",
                    skill="celestial-events-forecast",
                    purpose="获取用户关心时间范围内的天象事件",
                    success_criteria="返回周/月范围内的天象事件摘要",
                    evidence_key="celestial_events",
                    fallback_strategy="react_fallback",
                    parallel_group=event_group,
                    retry_policy=1,
                    timeout_ms=12000,
                )
            )
            if "weather-lookup" in skill_set or "天气" in query:
                steps.append(
                    make_step(
                        planner_source="template",
                        id="event_weather",
                        title="补充天气条件",
                        description="补充该地点的观测天气信息",
                        skill="weather-lookup",
                        purpose="补充天象观测的天气可行性",
                        success_criteria="返回指定城市天气或云量",
                        evidence_key="weather",
                        required=False,
                        fallback_strategy="continue",
                        parallel_group="event_context",
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )
        elif task_type == "deep_sky_guidance":
            rationale = (
                "深空指导以目标资料为核心；涉及今晚可见性或目标比较时补充位置计算。"
            )
            deep_sky_group = (
                "deep_sky_context"
                if self._deep_sky_needs_position_step(query)
                else None
            )
            steps.append(
                make_step(
                    planner_source="template",
                    id="deep_sky_guide",
                    title="生成深空观测指导",
                    description="输出目标可见性、设备建议与观测提示",
                    skill="deep-sky-observing-guide",
                    purpose="获取深空目标资料并生成观测建议",
                    success_criteria="返回目标基础信息、观测条件和器材建议",
                    evidence_key="deep_sky_object",
                    fallback_strategy="react_fallback",
                    parallel_group=deep_sky_group,
                    retry_policy=1,
                    timeout_ms=15000,
                )
            )
            if self._deep_sky_needs_position_step(query):
                steps.append(
                    make_step(
                        planner_source="template",
                        id="deep_sky_position",
                        title="补充可见性计算",
                        description="补充目标在指定时间地点的可见性或高度信息",
                        skill="celestial-position-calculator",
                        purpose="判断深空目标在给定时间地点是否适合观测",
                        success_criteria="返回目标位置或可见性数据",
                        evidence_key="celestial_position",
                        required=False,
                        fallback_strategy="continue",
                        parallel_group="deep_sky_context",
                        retry_policy=1,
                        timeout_ms=12000,
                    )
                )
        elif task_type == "astrophotography_advice":
            rationale = "摄影建议先计算拍摄参数；用户关心拍摄可行性或天气时再补充条件。"
            weather_relevant = self._photography_weather_relevant(query, skill_set)
            parallel_group = "imaging_context" if weather_relevant else None
            steps.append(
                make_step(
                    planner_source="template",
                    id="photo_settings",
                    title="计算摄影参数",
                    description="根据目标和器材估算拍摄参数",
                    skill="astrophotography-calculator",
                    purpose="计算目标、焦距、支架和曝光相关摄影参数",
                    success_criteria="返回单张曝光、ISO/光圈或器材相关建议",
                    evidence_key="astrophotography_settings",
                    fallback_strategy="react_fallback",
                    parallel_group=parallel_group,
                    retry_policy=1,
                    timeout_ms=15000,
                )
            )
            if self._photography_needs_deep_sky_context(query):
                steps.append(
                    make_step(
                        planner_source="template",
                        id="photo_target_context",
                        title="补充拍摄目标资料",
                        description="获取深空拍摄目标的亮度、类型和观测特点",
                        skill="deep-sky-observing-guide",
                        purpose="为深空目标曝光和累计时长建议提供目标证据",
                        success_criteria="返回目标基础信息与观测提示",
                        evidence_key="deep_sky_object",
                        required=False,
                        fallback_strategy="continue",
                        parallel_group=parallel_group,
                        retry_policy=1,
                        timeout_ms=15000,
                    )
                )
            if weather_relevant:
                steps.append(
                    make_step(
                        planner_source="template",
                        id="photo_weather",
                        title="查询摄影天气",
                        description="判断云量、透明度等是否适合拍摄",
                        skill="weather-lookup",
                        purpose="判断拍摄当晚天气是否适合执行",
                        success_criteria="返回指定城市天气、云量或透明度相关信息",
                        evidence_key="weather",
                        parallel_group="imaging_context",
                        required=False,
                        fallback_strategy="continue",
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )

        return ExecutionPlan(
            task_type=task_type,
            output_schema=output_schema,
            steps=steps,
            planner_type="template",
            rationale=rationale,
            planner_version=str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
            schema_version=str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            budget_policy_version=str(
                getattr(settings, "BUDGET_POLICY_VERSION", "budget_v1")
            ),
        )

    def _build_generic_plan(
        self,
        *,
        query: str,
        task_type: str,
        output_schema: str,
        capability_hints: List[str],
        chat_history: str,
        user_profile: str,
    ) -> ExecutionPlan:
        """根据能力提示或工具选择器构造通用计划。"""
        steps: List[PlanStep] = []
        capability_hints = list(dict.fromkeys(capability_hints))
        if not capability_hints:
            selected_tool = self._tool_selector.select(query)
            if selected_tool is not None:
                steps.append(
                    self._make_tool_step(
                        query=query,
                        planner_source="generic",
                        id="tool_1",
                        title=f"执行 {selected_tool.tool_name}",
                        description=f"调用 {selected_tool.tool_name} 获取回答所需信息",
                        tool_name=selected_tool.tool_name,
                        params=selected_tool.params,
                        purpose=f"调用 {selected_tool.tool_name} 获取回答证据",
                        success_criteria="返回可用于回答用户问题的工具结果",
                        evidence_key=selected_tool.tool_name,
                        fallback_strategy="react_fallback",
                        retry_policy=1,
                        timeout_ms=10000,
                    )
                )

        parallel_group = "generic_parallel" if len(capability_hints) > 1 else None
        for index, capability_name in enumerate(capability_hints, start=1):
            if self._skill_registry.has_skill(capability_name):
                steps.append(
                    self._make_step(
                        query=query,
                        chat_history=chat_history,
                        user_profile=user_profile,
                        planner_source="generic",
                        id=f"tool_{index}",
                        title=f"执行 {capability_name}",
                        description=f"调用 {capability_name} 获取回答所需信息",
                        skill=capability_name,
                        purpose=f"调用 {capability_name} 获取回答证据",
                        success_criteria="返回可用于回答用户问题的工具结果",
                        evidence_key=capability_name,
                        fallback_strategy="react_fallback",
                        parallel_group=parallel_group,
                        retry_policy=1,
                        timeout_ms=10000,
                    )
                )
                continue

            if self._tool_registry.has_tool(capability_name):
                steps.append(
                    self._make_tool_step(
                        query=query,
                        planner_source="generic",
                        id=f"tool_{index}",
                        title=f"执行 {capability_name}",
                        description=f"调用 {capability_name} 获取回答所需信息",
                        tool_name=capability_name,
                        purpose=f"调用 {capability_name} 获取回答证据",
                        success_criteria="返回可用于回答用户问题的工具结果",
                        evidence_key=capability_name,
                        fallback_strategy="react_fallback",
                        parallel_group=parallel_group,
                        retry_policy=1,
                        timeout_ms=10000,
                    )
                )

        planner_type = "template"
        rationale = "按已识别技能生成通用执行计划。"
        if not steps and self._llm is not None:
            planner_type = "llm_fallback"
            rationale = (
                "当前未命中专用模板，但保留 LLM Planner 扩展位。"
                f" chat_history={bool(chat_history)}, user_profile={bool(user_profile)}, query={query[:80]}"
            )

        return ExecutionPlan(
            task_type=task_type,
            output_schema=output_schema,
            steps=steps,
            planner_type=planner_type,
            rationale=rationale,
            planner_version=str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
            schema_version=str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            budget_policy_version=str(
                getattr(settings, "BUDGET_POLICY_VERSION", "budget_v1")
            ),
        )

    def _build_llm_fallback_plan(
        self,
        *,
        query: str,
        task_type: str,
        output_schema: str,
        chat_history: str,
        user_profile: str,
    ) -> Optional[ExecutionPlan]:
        """在模板和通用计划未命中时调用 LLM 生成受控 fallback 计划。"""
        if (
            not getattr(settings, "ENABLE_LLM_PLANNER_FALLBACK", False)
            or self._llm is None
        ):
            return None

        skill_definitions = registry.get_skill_definitions()
        allowed_skills = {definition.name for definition in skill_definitions}
        skills_text = "\n".join(
            f"- {definition.name}: {definition.summary}"
            for definition in skill_definitions
        )
        prompt = get_prompt_renderer().render(
            "planned.workflow_planner",
            {
                "task_type": task_type,
                "skills_text": skills_text,
                "user_profile_available": bool(user_profile),
                "chat_history_available": bool(chat_history),
                "query": query,
            },
        )

        try:
            raw = self._invoke_llm(prompt)
            payload = self._extract_json_object(raw)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        steps_payload = payload.get("steps") or []
        if not isinstance(steps_payload, list):
            return None

        steps: list[PlanStep] = []
        seen_skills: set[str] = set()
        for index, item in enumerate(steps_payload[:4], start=1):
            if not isinstance(item, dict):
                continue
            skill = str(item.get("skill") or "").strip()
            if skill not in allowed_skills or skill in seen_skills:
                continue
            seen_skills.add(skill)
            steps.append(
                self._make_step(
                    query=query,
                    planner_source="llm_fallback",
                    id=f"llm_tool_{index}",
                    title=f"执行 {skill}",
                    description=str(item.get("reason") or f"调用 {skill} 获取信息"),
                    skill=skill,
                    params=self._sanitize_step_params(skill, item.get("params")),
                    purpose=str(item.get("reason") or f"调用 {skill} 获取信息"),
                    success_criteria="返回可用于回答用户问题的工具结果",
                    evidence_key=skill,
                    fallback_strategy=(
                        "react_fallback"
                        if bool(item.get("required", True))
                        else "continue"
                    ),
                    required=bool(item.get("required", True)),
                    retry_policy=1,
                    timeout_ms=12000,
                )
            )

        if not steps:
            return None

        return ExecutionPlan(
            task_type=task_type,
            output_schema=output_schema,
            steps=steps,
            planner_type="llm_fallback",
            rationale=str(payload.get("rationale") or "LLM fallback generated plan."),
            planner_version=str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
            schema_version=str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            budget_policy_version=str(
                getattr(settings, "BUDGET_POLICY_VERSION", "budget_v1")
            ),
        )

    def _invoke_llm(self, prompt: str) -> str:
        """同步调用底层 LLM 并抽取文本内容。"""
        result = self._llm.invoke(prompt)
        return getattr(result, "content", None) or str(result)

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[dict[str, Any]]:
        """从 LLM 输出中提取 JSON 对象。"""
        text = (raw or "").strip()
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        elif not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def _make_step(
        self,
        *,
        query: str,
        chat_history: str = "",
        user_profile: str = "",
        planner_source: str,
        id: str,
        title: str,
        description: str,
        skill: str,
        params: Optional[dict[str, Any]] = None,
        purpose: str = "",
        success_criteria: str = "",
        fallback_strategy: str = "",
        evidence_key: str = "",
        depends_on: Optional[list[str]] = None,
        required: bool = True,
        parallel_group: Optional[str] = None,
        retry_policy: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> PlanStep:
        """创建高层技能计划步骤并补齐参数。"""
        step_params = (
            self._sanitize_step_params(skill, params)
            if params is not None
            else self._build_step_params(
                skill,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )
        )
        operation = step_params.get("operation")
        return self._capability_plan_adapter.make_skill_step(
            skill_name=skill,
            id=id,
            title=title,
            description=description,
            operation=operation,
            params=step_params,
            purpose=purpose,
            success_criteria=success_criteria,
            fallback_strategy=fallback_strategy,
            evidence_key=evidence_key,
            depends_on=list(depends_on or []),
            planner_source=planner_source,
            required=required,
            parallel_group=parallel_group,
            retry_policy=retry_policy,
            timeout_ms=timeout_ms,
        )

    def _make_tool_step(
        self,
        *,
        query: str,
        planner_source: str,
        id: str,
        title: str,
        description: str,
        tool_name: str,
        params: Optional[dict[str, Any]] = None,
        purpose: str = "",
        success_criteria: str = "",
        fallback_strategy: str = "",
        evidence_key: str = "",
        depends_on: Optional[list[str]] = None,
        required: bool = True,
        parallel_group: Optional[str] = None,
        retry_policy: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> PlanStep:
        """创建原子工具计划步骤并补齐参数。"""
        step_params = (
            dict(params)
            if isinstance(params, dict)
            else self._build_tool_step_params(tool_name, query)
        )
        return self._capability_plan_adapter.make_tool_step(
            tool_name=tool_name,
            id=id,
            title=title,
            description=description,
            params=step_params,
            purpose=purpose,
            success_criteria=success_criteria,
            fallback_strategy=fallback_strategy,
            evidence_key=evidence_key,
            depends_on=list(depends_on or []),
            planner_source=planner_source,
            required=required,
            parallel_group=parallel_group,
            retry_policy=retry_policy,
            timeout_ms=timeout_ms,
        )

    def _build_step_params(
        self,
        skill_name: str,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> dict[str, Any]:
        """为高层技能计划步骤构建参数。"""
        try:
            builder = CapabilityParamBuilder(
                self._param_builder,
                chat_history=chat_history,
                user_profile=user_profile,
            )
            return builder.build_for_capability("skill", skill_name, query)
        except Exception:
            return {}

    def _build_tool_step_params(
        self,
        tool_name: str,
        query: str,
    ) -> dict[str, Any]:
        """为原子工具计划步骤构建参数。"""
        try:
            return CapabilityParamBuilder.build_atomic_tool_params(tool_name, query)
        except Exception:
            return {}

    def _sanitize_step_params(
        self,
        skill_name: str,
        params: Any,
    ) -> dict[str, Any]:
        """按技能参数白名单清洗 LLM 生成的步骤参数。"""
        if not isinstance(params, dict):
            return {}
        try:
            definition = registry.get_skill_definition(skill_name)
        except Exception:
            return {}

        allowed = set(definition.input_field_names)
        candidate = {
            key: value
            for key, value in params.items()
            if key in allowed and value is not None
        }
        try:
            payload = definition.input_model.model_validate(candidate)
        except Exception:
            return {}
        return payload.model_dump(exclude_none=True)

    def _photography_weather_relevant(self, query: str, skill_set: set[str]) -> bool:
        """判断摄影建议是否需要补充天气条件。"""
        if "weather-lookup" in skill_set:
            return True

        weather_terms = (
            "天气",
            "云量",
            "云多",
            "多云",
            "晴",
            "雨",
            "风",
            "湿度",
            "雾霾",
            "透明度",
            "视宁度",
            "结露",
            "露水",
        )
        if any(term in query for term in weather_terms):
            return True

        observability_terms = (
            "适合拍",
            "能拍",
            "能不能拍",
            "好不好拍",
            "拍摄条件",
            "出片",
        )
        time_terms = (
            "今晚",
            "明晚",
            "今天",
            "明天",
            "本周末",
            "周末",
        )
        location_terms = (
            "北京",
            "上海",
            "广州",
            "深圳",
            "苏州",
            "杭州",
            "成都",
            "南京",
            "武汉",
        )
        has_explicit_when_or_where = any(term in query for term in time_terms) or any(
            term in query for term in location_terms
        )
        return has_explicit_when_or_where and any(
            term in query for term in observability_terms
        )

    def _observation_plan_needs_event_step(self, query: str) -> bool:
        """判断观测计划是否需要天象事件步骤。"""
        return any(
            token in query
            for token in (
                "周末两晚",
                "这个周末两晚",
                "未来三天",
                "哪天更适合",
                "周末晚上",
            )
        )

    def _observation_plan_needs_deep_sky_step(self, query: str) -> bool:
        """判断观测计划是否需要深空资料步骤。"""
        if any(token in query for token in ("看不清", "临时改看", "改看什么")):
            return False
        has_equipment_constraint = bool(
            re.search(r"\b\d{1,2}\s*x\s*\d{2}\b", query, re.IGNORECASE)
        ) or bool(re.search(r"\d{1,2}\s*寸", query))
        return (
            any(
                token in query
                for token in (
                    "双筒",
                    "DOB",
                    "深空",
                    "星云",
                    "星系",
                    "星团",
                    "从月亮到深空",
                )
            )
            or has_equipment_constraint
        )

    def _observation_plan_needs_position_step(self, query: str) -> bool:
        """判断观测计划是否需要天体位置步骤。"""
        target_count = sum(
            1
            for target in ("月亮", "月球", "木星", "土星", "火星", "金星", "水星")
            if target in query
        )
        has_sequence_intent = any(
            token in query
            for token in ("先看", "先观测", "优先看", "观测顺序", "顺序", "还是")
        )
        return target_count >= 2 and has_sequence_intent

    def _deep_sky_needs_position_step(self, query: str) -> bool:
        """判断深空指导是否需要可见性计算步骤。"""
        has_time = any(
            token in query
            for token in ("今晚", "明晚", "这周末", "周末", "今天", "明天")
        )
        has_visibility = any(
            token in query for token in ("能看到", "能看见", "适合", "哪个更", "差别")
        )
        return has_time and has_visibility

    def _photography_needs_deep_sky_context(self, query: str) -> bool:
        """判断摄影建议是否需要深空目标背景资料。"""
        return bool(
            re.search(
                r"\b(M\s?\d{1,3}|NGC\s?\d{1,5}|IC\s?\d{1,5})\b", query, re.IGNORECASE
            )
        ) or any(token in query for token in ("猎户座大星云", "仙女座星系"))
