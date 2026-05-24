"""记忆选择任务画像。

本文件定义 memory 层统一使用的 TaskContextProfile，用来把 Agent 路由画像、
短期检索焦点和长期记忆注入策略收敛到同一组字段，避免各子系统重复推断
task_type、场景、焦点实体和 freshness intent。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


CONTEXT_SCENES = {"observation", "computation", "learning_qa", "debugging", "general"}
MEMORY_TASK_TYPES = {"observation", "learning", "qa", "general"}


@dataclass(frozen=True)
class TaskContextProfile:
    """记忆选择专用任务画像，供短期 planner 和长期 injector 共同消费。"""

    query: str = ""
    task_type: str = "general"
    context_scene: str = "general"
    intent: str = "general"
    locations: Tuple[str, ...] = field(default_factory=tuple)
    targets: Tuple[str, ...] = field(default_factory=tuple)
    tool_types: Tuple[str, ...] = field(default_factory=tuple)
    freshness_intent: str = "neutral"
    capability_hints: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    source: str = "memory_rule"
    focus_drifted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 trace/API 可直接返回的稳定字典。"""

        return {
            "query": self.query,
            "task_type": self.task_type,
            "context_scene": self.context_scene,
            "intent": self.intent,
            "locations": list(self.locations),
            "targets": list(self.targets),
            "tool_types": list(self.tool_types),
            "freshness_intent": self.freshness_intent,
            "capability_hints": list(self.capability_hints),
            "confidence": self.confidence,
            "source": self.source,
            "focus_drifted": self.focus_drifted,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        query: str = "",
        task_type: Optional[str] = None,
        context_scene: Optional[str] = None,
    ) -> "TaskContextProfile":
        """从调用方传入的 dict 恢复画像，并用显式参数补齐缺省值。"""

        raw_task_type = task_type or str(data.get("task_type") or "")
        hints = _normalize_hints(data.get("capability_hints"))
        resolved_task_type = normalize_memory_task_type(
            raw_task_type,
            query=query or str(data.get("query") or ""),
            capability_hints=hints,
        )
        resolved_scene = normalize_context_scene(
            context_scene or str(data.get("context_scene") or ""),
            task_type=raw_task_type or resolved_task_type,
            query=query or str(data.get("query") or ""),
            capability_hints=hints,
        )
        metadata = dict(data.get("metadata") or {})
        if raw_task_type and raw_task_type != resolved_task_type:
            metadata.setdefault("raw_task_type", raw_task_type)
        return cls(
            query=query or str(data.get("query") or ""),
            task_type=resolved_task_type,
            context_scene=resolved_scene,
            intent=str(data.get("intent") or "general"),
            locations=_normalize_tuple(data.get("locations")),
            targets=_normalize_tuple(data.get("targets")),
            tool_types=_normalize_tuple(data.get("tool_types")),
            freshness_intent=str(data.get("freshness_intent") or "neutral"),
            capability_hints=hints,
            confidence=_clamp_float(data.get("confidence"), 0.0),
            source=str(data.get("source") or "memory_profile"),
            focus_drifted=bool(data.get("focus_drifted")),
            metadata=metadata,
        )

    @classmethod
    def from_task_profile(
        cls,
        task_profile: Any,
        query: str = "",
        context_scene: Optional[str] = None,
        intent: str = "general",
        locations: Iterable[str] = (),
        targets: Iterable[str] = (),
        tool_types: Iterable[str] = (),
        freshness_intent: str = "neutral",
        focus_drifted: bool = False,
    ) -> "TaskContextProfile":
        """把 Agent 层 TaskProfile 适配成 memory 层任务画像。"""

        raw_task_type = str(getattr(task_profile, "task_type", "") or "")
        hints = _normalize_hints(
            getattr(task_profile, "capability_hints", None)
            or getattr(task_profile, "matched_skills", None)
        )
        resolved_task_type = normalize_memory_task_type(
            raw_task_type,
            query=query,
            capability_hints=hints,
        )
        resolved_scene = normalize_context_scene(
            context_scene,
            task_type=raw_task_type or resolved_task_type,
            query=query,
            capability_hints=hints,
        )
        metadata = {
            "raw_task_type": raw_task_type,
            "legacy_route": getattr(task_profile, "legacy_route", ""),
            "router_source": getattr(task_profile, "router_source", ""),
        }
        return cls(
            query=query,
            task_type=resolved_task_type,
            context_scene=resolved_scene,
            intent=intent,
            locations=_normalize_tuple(locations),
            targets=_normalize_tuple(targets),
            tool_types=_normalize_tuple(tool_types or hints),
            freshness_intent=freshness_intent or "neutral",
            capability_hints=hints,
            confidence=_clamp_float(getattr(task_profile, "confidence", 0.0), 0.0),
            source="agent_task_profile",
            focus_drifted=focus_drifted,
            metadata={key: value for key, value in metadata.items() if value},
        )

    @classmethod
    def from_memory_inputs(
        cls,
        query: str,
        task_type: Optional[str] = None,
        context_scene: Optional[str] = None,
        intent: str = "general",
        locations: Iterable[str] = (),
        targets: Iterable[str] = (),
        tool_types: Iterable[str] = (),
        freshness_intent: str = "neutral",
        capability_hints: Any = None,
        focus_drifted: bool = False,
        source: str = "memory_rule",
    ) -> "TaskContextProfile":
        """从 memory 现有入参和短期 focus 构造兜底画像。"""

        hints = _normalize_hints(capability_hints)
        resolved_task_type = normalize_memory_task_type(
            task_type,
            query=query,
            capability_hints=hints,
        )
        resolved_scene = normalize_context_scene(
            context_scene,
            task_type=task_type or resolved_task_type,
            query=query,
            capability_hints=hints,
        )
        return cls(
            query=query or "",
            task_type=resolved_task_type,
            context_scene=resolved_scene,
            intent=intent or "general",
            locations=_normalize_tuple(locations),
            targets=_normalize_tuple(targets),
            tool_types=_normalize_tuple(tool_types or hints),
            freshness_intent=freshness_intent or "neutral",
            capability_hints=hints,
            confidence=0.0,
            source=source,
            focus_drifted=focus_drifted,
        )

    def with_focus(
        self,
        locations: Iterable[str] = (),
        targets: Iterable[str] = (),
        tool_types: Iterable[str] = (),
        freshness_intent: Optional[str] = None,
        focus_drifted: Optional[bool] = None,
    ) -> "TaskContextProfile":
        """返回合并短期 focus 信息后的画像副本。"""

        return TaskContextProfile(
            query=self.query,
            task_type=self.task_type,
            context_scene=self.context_scene,
            intent=self.intent,
            locations=_merge_tuple(self.locations, locations),
            targets=_merge_tuple(self.targets, targets),
            tool_types=_merge_tuple(self.tool_types, tool_types),
            freshness_intent=freshness_intent or self.freshness_intent,
            capability_hints=self.capability_hints,
            confidence=self.confidence,
            source=self.source,
            focus_drifted=self.focus_drifted
            if focus_drifted is None
            else bool(focus_drifted),
            metadata=dict(self.metadata or {}),
        )


def coerce_task_context_profile(
    value: Any,
    query: str = "",
    task_type: Optional[str] = None,
    context_scene: Optional[str] = None,
) -> Optional[TaskContextProfile]:
    """把 Optional[TaskContextProfile|dict|TaskProfile] 统一转换为画像。"""

    if value is None:
        return None
    if isinstance(value, TaskContextProfile):
        profile = value
        if (
            (query and profile.query != query)
            or (task_type and profile.task_type != task_type)
            or (context_scene and profile.context_scene != context_scene)
        ):
            profile = TaskContextProfile.from_mapping(
                {**profile.to_dict(), "query": query},
                query=query,
                task_type=task_type or profile.task_type,
                context_scene=context_scene or profile.context_scene,
            )
        return profile
    if isinstance(value, Mapping):
        return TaskContextProfile.from_mapping(
            value,
            query=query,
            task_type=task_type,
            context_scene=context_scene,
        )
    if hasattr(value, "task_type"):
        return TaskContextProfile.from_task_profile(
            value,
            query=query,
            context_scene=context_scene,
        )
    return None


def normalize_memory_task_type(
    task_type: Optional[str],
    query: str = "",
    capability_hints: Iterable[str] = (),
) -> str:
    """把 Agent/短期场景任务类型规整为长期记忆权重可识别的类型。"""

    raw = str(task_type or "").strip().lower()
    hints = " ".join(str(item).lower() for item in capability_hints or ())
    text = f"{raw} {hints} {query or ''}".lower()
    if raw in MEMORY_TASK_TYPES:
        return raw
    if raw == "learning_qa":
        return "learning"
    if raw in {"simple_qa", "direct_answer_no_tool", "qa"}:
        return "qa"
    if raw in {"smalltalk", "clarification", "general"}:
        return "general"
    if any(token in text for token in ["observation", "weather", "visibility", "观测", "天气", "可见"]):
        return "observation"
    if any(token in text for token in ["learn", "learning", "explain", "什么是", "为什么", "科普"]):
        return "learning"
    if any(token in text for token in ["astronomy", "celestial", "星", "天文"]):
        return "qa"
    return "general"


def normalize_context_scene(
    context_scene: Optional[str],
    task_type: Optional[str] = None,
    query: str = "",
    capability_hints: Iterable[str] = (),
) -> str:
    """把显式场景、任务类型和能力提示规整为短期上下文装配场景。"""

    raw_scene = str(context_scene or "").strip().lower()
    if raw_scene in CONTEXT_SCENES:
        return raw_scene
    raw_task = str(task_type or "").strip().lower()
    if raw_task in CONTEXT_SCENES:
        return raw_task
    hints = " ".join(str(item).lower() for item in capability_hints or ())
    text = f"{raw_task} {hints} {query or ''}".lower()
    if any(token in text for token in ["debug", "traceback", "exception", "报错", "失败", "修复"]):
        return "debugging"
    if any(token in text for token in ["compute", "calculate", "formula", "计算", "公式", "推导", "参数"]):
        return "computation"
    if any(token in text for token in ["observation", "weather", "visibility", "观测", "天气", "可见", "升起", "落下"]):
        return "observation"
    if any(token in text for token in ["learn", "explain", "simple_qa", "什么是", "为什么", "科普", "学习"]):
        return "learning_qa"
    return "general"


def _normalize_hints(value: Any) -> Tuple[str, ...]:
    """把 dict/list/str 形式的 capability hints 归一为去重字符串元组。"""

    if isinstance(value, Mapping):
        raw_items = []
        for key in ["capability_hints", "matched_skills", "tools", "tool_types"]:
            item = value.get(key)
            if isinstance(item, (list, tuple, set)):
                raw_items.extend(item)
            elif item:
                raw_items.append(item)
        for key in ["context_scene", "task_type", "tool_type"]:
            if value.get(key):
                raw_items.append(value[key])
    elif isinstance(value, str):
        raw_items = [value]
    else:
        raw_items = list(value or [])
    return _normalize_tuple(raw_items)


def _normalize_tuple(value: Any) -> Tuple[str, ...]:
    """把任意可迭代值归一为稳定去重元组。"""

    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    seen = set()
    normalized = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return tuple(normalized)


def _merge_tuple(left: Iterable[str], right: Iterable[str]) -> Tuple[str, ...]:
    """合并两个字符串集合并保持左侧优先顺序。"""

    return _normalize_tuple([*(left or ()), *(right or ())])


def _clamp_float(value: Any, default: float = 0.0) -> float:
    """把置信度等输入安全限制在 0 到 1 区间。"""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(parsed, 1.0))


__all__ = [
    "TaskContextProfile",
    "coerce_task_context_profile",
    "normalize_context_scene",
    "normalize_memory_task_type",
]
