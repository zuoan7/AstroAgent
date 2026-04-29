from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.agent.executor import EventCallback, StepExecutor
from src.agent.planner import Planner
from src.agent.models.execution_plan import ExecutionPlan
from src.agent.param_parser import ParamParser
from src.agent.policies.budget_policy import RequestBudgetTracker
from src.agent.policies.fallback_policy import FallbackPolicy
from src.agent.request_router import RouteDecision
from src.agent.models.skill_result import SkillResult
from src.agent.models.final_response import FinalResponse
from src.agent.response_synthesizer import ResponseSynthesizer
from src.core.config import settings
from src.skills import registry


class TaskOrchestrator:
    """Direct and planned execution paths for low-latency online requests.

    .. deprecated::
        Phase 8 起，主路径已迁移至 ExecutionEngine（ENABLE_UNIFIED_EXECUTION_ENGINE=True）。
        本类保留为兼容层（flag=False 或 ExecutionEngine 未注入时的回退路径）。
        当前去向：
        1. `run()` 作为旧 direct/planned 外部调用入口继续保留；
        2. `build_execution_plan()` 仅供 legacy planned 展示/执行链路复用；
        3. 新代码不得新增对本类的内部主路径依赖。

        删除条件：
        - StreamingService 不再需要 legacy orchestrator 回退；
        - 外部调用方迁移到 ExecutionEngine / TaskProfile / ExecutionDecision；
        - 兼容测试与 flag=False 回退链路退场。
    """

    def __init__(
        self,
        skill_manager: Any,
        rag_retriever: Any,
        llm: Any,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        planner: Optional[Planner] = None,
        executor: Optional[StepExecutor] = None,
        fallback_policy: Optional[FallbackPolicy] = None,
    ) -> None:
        self._skill_manager = skill_manager
        self._rag = rag_retriever
        self._llm = llm
        self._synthesizer = response_synthesizer or ResponseSynthesizer(llm=llm)
        self._planner = planner or Planner(llm=llm)
        self._executor = executor or StepExecutor(skill_manager=skill_manager)
        self._fallback_policy = fallback_policy or FallbackPolicy()

    async def run(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
        execution_plan: Optional[ExecutionPlan] = None,
        event_callback: Optional[EventCallback] = None,
    ) -> FinalResponse:
        """Deprecated compatibility entry for legacy direct/planned callers.

        新主路径应使用 `ExecutionEngine.run()`。本方法仅用于：
        - `ENABLE_UNIFIED_EXECUTION_ENGINE=False` 的兼容回退
        - 尚未迁移的外部 direct/planned 调用
        - 历史测试基线
        """
        budget_tracker = RequestBudgetTracker()
        self._synthesizer._budget_tracker = budget_tracker
        if decision.route == "direct_task":
            return await self._run_direct_task(
                decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )

        if decision.route == "planned_task":
            return await self._run_planned_task(
                decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
                execution_plan=execution_plan,
                event_callback=event_callback,
                budget_tracker=budget_tracker,
            )

        raise ValueError(f"unsupported orchestrated route: {decision.route}")

    async def _run_direct_task(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> FinalResponse:
        if decision.task_type == "smalltalk":
            return self._synthesizer.synthesize_smalltalk(
                self._smalltalk_reply(query)
            )

        if decision.task_type == "single_tool_lookup":
            return await self._run_tool_task(decision, query)

        if decision.task_type == "simple_qa":
            return await self._run_simple_qa(
                query, chat_history=chat_history, user_profile=user_profile
            )

        raise ValueError(f"unsupported direct task type: {decision.task_type}")

    async def _run_planned_task(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
        execution_plan: Optional[ExecutionPlan] = None,
        event_callback: Optional[EventCallback] = None,
        budget_tracker: Optional[RequestBudgetTracker] = None,
    ) -> FinalResponse:
        plan = execution_plan or self.build_execution_plan(
            decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        if not plan.steps:
            raise ValueError(f"no planned-task steps resolved for {decision.task_type}")

        outcome = await self._executor.execute(
            plan,
            query=query,
            param_builder=self._build_skill_params,
            event_callback=event_callback,
            budget_tracker=budget_tracker,
        )
        fallback_decision = self._fallback_policy.decide_for_execution(
            outcome=outcome,
            plan=plan,
        )

        return self._synthesizer.synthesize(
            query=query,
            task_type=decision.task_type,
            output_schema=decision.expected_output_schema,
            skill_results=outcome.skill_results,
            chat_history=chat_history,
            user_profile=user_profile,
            route=decision.route,
            execution_plan=plan.to_dict(),
            execution_trace=[step.to_dict() for step in outcome.step_results],
            route_decision=decision.to_meta(),
            fallback_path=[fallback_decision.to_dict()] if fallback_decision else [],
            budget_usage=budget_tracker.snapshot() if budget_tracker else None,
            versions={
                "router_policy_version": str(
                    getattr(settings, "ROUTER_POLICY_VERSION", "router_v1")
                ),
                "planner_version": str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
                "schema_version": str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
                "synth_prompt_version": str(
                    getattr(settings, "SYNTH_PROMPT_VERSION", "synth_prompt_v2")
                ),
                "fallback_policy_version": self._fallback_policy.version,
                "budget_policy_version": (
                    budget_tracker.budget.policy_version if budget_tracker else "budget_v1"
                ),
            },
        )

    def build_execution_plan(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> ExecutionPlan:
        """Deprecated compatibility helper.

        新 planned 主路径优先走 `Planner.plan_graph()` / `ExecutionEngine.preview_plan()`；
        StreamingService 在 unified engine 可用时不应再调用本方法。

        当前仅保留给：
        - TaskOrchestrator.run() 的 legacy planned 执行链路
        - StreamingService 在 legacy orchestrator 回退模式下的 planned 展示
        """
        return self._planner.plan(
            query=query,
            route_decision=decision,
            chat_history=chat_history,
            user_profile=user_profile,
        )

    async def _run_simple_qa(
        self,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> FinalResponse:
        retrieval = self._rag.retrieve(query, fast_mode=True)
        context = retrieval.get("context", "")
        prompt = (
            "你是天文助手。请基于给定知识，用简洁直接的中文回答。\n"
            "如果知识不足，要明确说明不确定，不要编造。\n\n"
            f"用户画像：\n{user_profile[:400]}\n\n"
            f"最近对话：\n{chat_history[:800]}\n\n"
            f"知识：\n{context[:2400]}\n\n"
            f"问题：{query}\n\n回答："
        )
        answer = await asyncio.to_thread(self._invoke_llm, prompt)

        response = self._synthesizer.synthesize_qa(
            query=query,
            answer=answer,
            rag_context=context,
            retrieval=retrieval,
        )
        return response

    async def _run_tool_task(self, decision: RouteDecision, query: str) -> FinalResponse:
        skill_name = decision.matched_skills[0]
        params = self._build_skill_params(skill_name, query)
        result: SkillResult = await asyncio.to_thread(
            self._skill_manager.call_skill,
            skill_name,
            **params,
        )

        response = self._synthesizer.synthesize_direct(
            query=query,
            task_type=decision.task_type,
            skill_results=[result],
        )
        return response

    def _invoke_llm(self, prompt: str) -> str:
        if getattr(self._synthesizer, "_budget_tracker", None):
            self._synthesizer._budget_tracker.register_context_chars(len(prompt))
            self._synthesizer._budget_tracker.register_llm_call()
        result = self._llm.invoke(prompt)
        return getattr(result, "content", None) or str(result)

    def _smalltalk_reply(self, query: str) -> str:
        normalized = (query or "").strip().lower()
        if "谢谢" in query or "thanks" in normalized:
            return "不客气。如果你想查天象、天气或观测计划，直接告诉我时间和地点即可。"
        if "在吗" in query:
            return "在。可以直接问我天文知识、今晚观测目标、天气或观测计划。"
        return "你好，我可以帮你查询天象、观测条件、天体位置和天文知识。"

    def _build_skill_params(self, skill_name: str, query: str) -> Dict[str, Any]:
        parsed = ParamParser.parse(query)
        if self._is_structured_skill_payload(parsed, query):
            return self._finalize_skill_params(skill_name, parsed)

        if skill_name == "weather-lookup":
            return self._finalize_skill_params(skill_name, {"city": query.strip()})
        if skill_name == "observation-planner":
            return self._finalize_skill_params(
                skill_name,
                {
                    "location": self._extract_location(query) or query.strip(),
                    "date": self._extract_date(query),
                },
            )
        if skill_name == "deep-sky-observing-guide":
            return self._finalize_skill_params(
                skill_name,
                {
                    "target": self._extract_target(query) or query.strip(),
                    "observer_location": self._extract_location(query),
                    "date": self._extract_date(query),
                    "equipment": self._extract_equipment(query),
                },
            )
        if skill_name == "celestial-events-forecast":
            start_date, end_date = self._extract_event_range(query)
            return self._finalize_skill_params(
                skill_name,
                {
                    "event_type": self._extract_event_type(query),
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        if skill_name == "astrophotography-calculator":
            return self._finalize_skill_params(
                skill_name,
                {
                    "target": self._extract_target(query) or query.strip(),
                    "camera": self._extract_camera(query) or "未指定相机",
                    "location": self._extract_location(query),
                    "date": self._extract_date(query),
                },
            )
        if skill_name == "celestial-position-calculator":
            return self._finalize_skill_params(
                skill_name,
                {
                    "target": self._extract_target(query) or query.strip(),
                    "location": self._extract_location(query),
                    "datetime": self._extract_datetime(query),
                },
            )

        spec = registry.get_skill_spec(skill_name)
        fallback = (
            {spec.param_names[0]: query.strip()}
            if len(spec.param_names) == 1
            else {}
        )
        return self._finalize_skill_params(skill_name, fallback)

    def _is_structured_skill_payload(
        self,
        parsed: Dict[str, Any],
        query: str,
    ) -> bool:
        if not isinstance(parsed, dict) or not parsed:
            return False
        if set(parsed.keys()) != {"query"}:
            return True
        return str(parsed.get("query", "")).strip() != query.strip()

    def _finalize_skill_params(
        self,
        skill_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        spec = registry.get_skill_spec(skill_name)
        normalized = dict(spec.defaults or {})
        candidate = dict(params or {})
        if spec.special_handling:
            candidate = spec.special_handling(candidate)
        for name in spec.param_names:
            value = candidate.get(name)
            if value is not None:
                normalized[name] = value
        return normalized

    def _skills_for_task_type(self, task_type: str) -> List[str]:
        mapping = {
            "observation_recommendation": [
                "weather-lookup",
                "observation-planner",
            ],
            "celestial_event_analysis": ["celestial-events-forecast"],
            "deep_sky_guidance": ["deep-sky-observing-guide"],
            "astrophotography_advice": [
                "astrophotography-calculator",
                "weather-lookup",
            ],
        }
        return list(mapping.get(task_type, []))

    def _extract_location(self, query: str) -> Optional[str]:
        for city in ("北京", "上海", "广州", "深圳", "苏州", "杭州", "成都", "南京", "武汉"):
            if city in query:
                return city
        return None

    def _extract_target(self, query: str) -> Optional[str]:
        catalog_match = re.search(r"\b(M\d{1,3}|NGC\s?\d{1,4})\b", query, re.IGNORECASE)
        if catalog_match:
            return catalog_match.group(1).upper().replace(" ", "")
        for target in ("木星", "土星", "火星", "金星", "月球", "太阳", "M31", "M42", "猎户座大星云"):
            if target in query:
                return target
        return None

    def _extract_date(self, query: str) -> Optional[str]:
        for token in ("今天", "明天", "今晚", "明晚", "本周末", "下周一"):
            if token in query:
                return token
        return None

    def _extract_event_range(self, query: str) -> tuple[Optional[str], Optional[str]]:
        month_match = re.search(r"(\d{4})年(\d{1,2})月", query)
        if month_match:
            year = int(month_match.group(1))
            month = int(month_match.group(2))
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year, month + 1, 1) - timedelta(days=1)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if "本月" in query:
            today = datetime.now()
            start = today.replace(day=1)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if any(token in query for token in ("未来一周", "未来7天", "本周天象", "这周天象", "一周天象")):
            start = datetime.now()
            end = start + timedelta(days=7)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        date_value = self._extract_date(query)
        return date_value, None

    def _extract_datetime(self, query: str) -> Optional[str]:
        match = re.search(r"\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2})?", query)
        if match:
            return match.group(0)
        return self._extract_date(query)

    def _extract_equipment(self, query: str) -> Optional[str]:
        for equipment in ("双筒", "双筒望远镜", "小折射镜", "8寸望远镜", "赤道仪", "三脚架"):
            if equipment in query:
                return equipment
        return None

    def _extract_camera(self, query: str) -> Optional[str]:
        for camera in ("Sony", "Canon", "Nikon", "ZWO", "QHY", "相机"):
            if camera in query:
                return camera
        return None

    def _extract_event_type(self, query: str) -> Optional[str]:
        for event_type in ("流星雨", "月食", "日食", "行星合月", "冲日", "掩星"):
            if event_type in query:
                return event_type
        return None
