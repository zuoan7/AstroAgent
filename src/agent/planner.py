from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.workflow_graph import WorkflowGraph
from src.agent.skill_param_builder import SkillParamBuilder
from src.core.config import settings
from src.skills import registry


class Planner:
    """
    Planner for planned tasks.

    当前优先输出 WorkflowGraph（plan_graph），ExecutionPlan 仅保留兼容表示。
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        self._llm = llm
        self._param_builder = SkillParamBuilder(None)

    def plan(
        self,
        *,
        query: str,
        route_decision: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> ExecutionPlan:
        """Deprecated compatibility entry returning ExecutionPlan.

        新 planned 主路径应优先使用 `plan_graph()` 获取 WorkflowGraph；
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
        """Primary planned-path entry returning WorkflowGraph.

        初版复用现有模板/通用规划逻辑，但对外直接返回 WorkflowGraph，
        使 graph 成为 planned 路径的优先计划表达。
        """
        plan = self._resolve_plan(
            query=query,
            route_decision=route_decision,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return WorkflowGraph.from_execution_plan(plan)

    def _resolve_plan(
        self,
        *,
        query: str,
        route_decision: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> ExecutionPlan:
        task_type = getattr(route_decision, "task_type", "observation_recommendation")
        output_schema = getattr(route_decision, "expected_output_schema", "generic_answer_v1")
        matched_skills = list(getattr(route_decision, "matched_skills", []) or [])

        plan = self._build_template_plan(
            query=query,
            task_type=task_type,
            output_schema=output_schema,
            matched_skills=matched_skills,
        )
        if plan.steps:
            return plan

        generic_plan = self._build_generic_plan(
            query=query,
            task_type=task_type,
            output_schema=output_schema,
            matched_skills=matched_skills,
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
        matched_skills: List[str],
    ) -> ExecutionPlan:
        skill_set = set(matched_skills)
        steps: List[PlanStep] = []
        rationale = ""

        if task_type == "observation_recommendation":
            rationale = "观测推荐通常需要先获取环境条件，再生成目标与时段建议。"
            if "weather-lookup" in skill_set or "天气" in query or not skill_set:
                steps.append(
                    self._make_step(
                        query=query,
                        planner_source="template",
                        id="weather_context",
                        title="查询天气条件",
                        description="获取当前位置或目标地点的天气与云量信息",
                        skill="weather-lookup",
                        purpose="确认观测条件中的天气、云量和风等环境因素",
                        success_criteria="返回指定城市的天气或云量信息",
                        evidence_key="weather",
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )
            steps.append(
                self._make_step(
                    query=query,
                    planner_source="template",
                    id="observation_plan",
                    title="生成观测计划",
                    description="基于地点和时间生成可执行的观测建议",
                    skill="observation-planner",
                    purpose="生成目标、时段和实用观测安排",
                    success_criteria="返回指定日期地点的观测计划或目标建议",
                    evidence_key="observation_plan",
                    retry_policy=1,
                    timeout_ms=12000,
                )
            )
        elif task_type == "celestial_event_analysis":
            rationale = "天象分析以天象事件检索为核心，必要时补充观测条件。"
            steps.append(
                self._make_step(
                    query=query,
                    planner_source="template",
                    id="event_forecast",
                    title="查询天象事件",
                    description="获取指定时段内的主要天象与事件信息",
                    skill="celestial-events-forecast",
                    purpose="获取用户关心时间范围内的天象事件",
                    success_criteria="返回周/月范围内的天象事件摘要",
                    evidence_key="celestial_events",
                    retry_policy=1,
                    timeout_ms=12000,
                )
            )
            if "weather-lookup" in skill_set or "天气" in query:
                steps.append(
                    self._make_step(
                        query=query,
                        planner_source="template",
                        id="event_weather",
                        title="补充天气条件",
                        description="补充该地点的观测天气信息",
                        skill="weather-lookup",
                        purpose="补充天象观测的天气可行性",
                        success_criteria="返回指定城市天气或云量",
                        evidence_key="weather",
                        required=False,
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )
        elif task_type == "deep_sky_guidance":
            rationale = "深空指导需要目标观测建议，可选补充天气判断。"
            steps.append(
                self._make_step(
                    query=query,
                    planner_source="template",
                    id="deep_sky_guide",
                    title="生成深空观测指导",
                    description="输出目标可见性、设备建议与观测提示",
                    skill="deep-sky-observing-guide",
                    purpose="获取深空目标资料并生成观测建议",
                    success_criteria="返回目标基础信息、观测条件和器材建议",
                    evidence_key="deep_sky_object",
                    retry_policy=1,
                    timeout_ms=15000,
                )
            )
            if "weather-lookup" in skill_set or any(token in query for token in ("今晚", "天气", "云量")):
                steps.append(
                    self._make_step(
                        query=query,
                        planner_source="template",
                        id="deep_sky_weather",
                        title="补充天气条件",
                        description="补充当前地点的天空条件判断",
                        skill="weather-lookup",
                        purpose="补充深空观测的天气条件",
                        success_criteria="返回指定城市天气或云量",
                        evidence_key="weather",
                        required=False,
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )
        elif task_type == "astrophotography_advice":
            rationale = "摄影建议先计算拍摄参数；用户关心拍摄可行性或天气时再补充条件。"
            weather_relevant = self._photography_weather_relevant(query, skill_set)
            parallel_group = "imaging_context" if weather_relevant else None
            steps.append(
                self._make_step(
                    query=query,
                    planner_source="template",
                    id="photo_settings",
                    title="计算摄影参数",
                    description="根据目标和器材估算拍摄参数",
                    skill="astrophotography-calculator",
                    purpose="计算目标、焦距、支架和曝光相关摄影参数",
                    success_criteria="返回单张曝光、ISO/光圈或器材相关建议",
                    evidence_key="astrophotography_settings",
                    parallel_group=parallel_group,
                    retry_policy=1,
                    timeout_ms=15000,
                )
            )
            if weather_relevant:
                steps.append(
                    self._make_step(
                        query=query,
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
        matched_skills: List[str],
        chat_history: str,
        user_profile: str,
    ) -> ExecutionPlan:
        steps: List[PlanStep] = []
        for index, skill_name in enumerate(matched_skills, start=1):
            steps.append(
                self._make_step(
                    query=query,
                    planner_source="generic",
                    id=f"tool_{index}",
                    title=f"执行 {skill_name}",
                    description=f"调用 {skill_name} 获取回答所需信息",
                    skill=skill_name,
                    purpose=f"调用 {skill_name} 获取回答证据",
                    success_criteria="返回可用于回答用户问题的工具结果",
                    evidence_key=skill_name,
                    parallel_group="generic_parallel" if len(matched_skills) > 1 else None,
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
        if not getattr(settings, "ENABLE_LLM_PLANNER_FALLBACK", False) or self._llm is None:
            return None

        skill_specs = registry.get_skill_specs()
        allowed_skills = {spec.skill_name for spec in skill_specs}
        skills_text = "\n".join(
            f"- {spec.skill_name}: {spec.summary}" for spec in skill_specs
        )
        prompt = (
            "你是 AstroAgent 的结构化计划生成器。只输出 JSON 对象，不要输出解释文字。\n"
            "仅当用户问题确实需要工具时选择步骤；只能使用给定 skills，禁止发明工具。\n"
            "输出最多 4 个步骤，按执行顺序排列。\n\n"
            f"task_type: {task_type}\n"
            f"可用 skills:\n{skills_text}\n\n"
            "输出 JSON schema:\n"
            "{\n"
            '  "steps": [\n'
            '    {"skill": "weather-lookup", "required": true, "reason": "查询云量", "params": {"city": "北京"}}\n'
            "  ],\n"
            '  "rationale": "简短计划理由"\n'
            "}\n\n"
            f"用户画像可用性: {bool(user_profile)}\n"
            f"历史对话可用性: {bool(chat_history)}\n"
            f"用户问题: {query}\n"
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
        result = self._llm.invoke(prompt)
        return getattr(result, "content", None) or str(result)

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[dict[str, Any]]:
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
        required: bool = True,
        parallel_group: Optional[str] = None,
        retry_policy: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> PlanStep:
        step_params = (
            self._sanitize_step_params(skill, params)
            if params is not None
            else self._build_step_params(skill, query)
        )
        return PlanStep(
            id=id,
            kind="tool",
            title=title,
            description=description,
            skill=skill,
            params=step_params,
            purpose=purpose,
            success_criteria=success_criteria,
            fallback_strategy=fallback_strategy,
            evidence_key=evidence_key,
            planner_source=planner_source,
            required=required,
            parallel_group=parallel_group,
            retry_policy=retry_policy,
            timeout_ms=timeout_ms,
        )

    def _build_step_params(self, skill_name: str, query: str) -> dict[str, Any]:
        try:
            return self._param_builder.build(skill_name, query)
        except Exception:
            return {}

    def _sanitize_step_params(
        self,
        skill_name: str,
        params: Any,
    ) -> dict[str, Any]:
        if not isinstance(params, dict):
            return {}
        try:
            spec = registry.get_skill_spec(skill_name)
        except Exception:
            return {}

        allowed = set(spec.param_names)
        candidate = {
            key: value
            for key, value in params.items()
            if key in allowed and value is not None
        }
        if spec.special_handling:
            candidate = spec.special_handling(candidate)

        normalized: dict[str, Any] = {}
        for name in spec.param_names:
            value = candidate.get(name)
            if value is None:
                continue
            converter = spec.type_conversions.get(name)
            if converter is not None:
                try:
                    value = converter(value)
                except Exception:
                    continue
            normalized[name] = value
        return normalized

    def _photography_weather_relevant(self, query: str, skill_set: set[str]) -> bool:
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
