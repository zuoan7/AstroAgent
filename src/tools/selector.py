from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

KNOWN_CITIES = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "苏州",
    "杭州",
    "成都",
    "南京",
    "武汉",
    "西安",
    "重庆",
    "天津",
    "青岛",
    "厦门",
)

DEFAULT_WEATHER_CITY = "北京"


@dataclass(frozen=True)
class ToolSelectionDecision:
    mode: str
    tool_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "tool_name": self.tool_name,
            "params": dict(self.params),
            "confidence": self.confidence,
            "reason": self.reason,
        }


class AtomicToolParamAdapter:
    """Deterministic parameter adapter for stable atomic MCP tools."""

    @classmethod
    def build(
        cls,
        tool_name: str,
        query: str,
        *,
        explicit_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(explicit_params, dict):
            return dict(explicit_params)

        if tool_name == "web_search":
            return {
                "query": cls._extract_web_search_query(query),
                "max_results": 5,
            }
        if tool_name == "get_nasa_apod":
            return {
                "date": cls._extract_iso_date(query),
                "hd": False,
            }
        if tool_name == "get_weather":
            return {
                "city": cls._extract_weather_city(query) or DEFAULT_WEATHER_CITY,
                "extensions": "all",
            }
        return {}

    @staticmethod
    def _extract_web_search_query(query: str) -> str:
        cleaned = (query or "").strip()
        cleaned = re.sub(r"^(请|帮我|麻烦)?\s*(查一下|查询|搜索|搜一下)", "", cleaned)
        cleaned = re.sub(r"(是什么|有哪些|怎么样)[？?。]*$", "", cleaned)
        return cleaned.strip(" ？?。") or (query or "").strip()

    @staticmethod
    def _extract_iso_date(query: str) -> Optional[str]:
        text = query or ""
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            return match.group(0)

        today = datetime.now()
        if any(token in text for token in ("今天", "今日", "今晚", "每日天文图")):
            return today.strftime("%Y-%m-%d")
        if "昨天" in text:
            return (today - timedelta(days=1)).strftime("%Y-%m-%d")
        if "明天" in text:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        return None

    @staticmethod
    def _extract_weather_city(query: str) -> Optional[str]:
        text = query or ""
        coord_match = re.search(
            r"(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*[,，]\s*(-?\d{2,3}(?:\.\d+)?)(?!\d)",
            text,
        )
        if coord_match:
            return f"{coord_match.group(1)},{coord_match.group(2)}"
        for city in KNOWN_CITIES:
            if city in text:
                return city
        return None


class ToolSelector:
    """Rule-only selector for stable atomic MCP tools.

    v1 intentionally covers only high-confidence direct lookups and does not
    call an LLM.
    """

    def select(
        self,
        query: str,
        *,
        profile: Optional[Any] = None,
    ) -> Optional[ToolSelectionDecision]:
        text = (query or "").strip()
        if not text:
            return None

        hinted = self._select_from_hints(text, profile)
        if hinted is not None:
            return hinted

        if self._is_apod_lookup(text):
            return self._decision(
                "get_nasa_apod",
                text,
                confidence=0.9,
                reason="matched_apod_lookup_rule",
            )

        if self._is_web_search_lookup(text):
            return self._decision(
                "web_search",
                text,
                confidence=0.88,
                reason="matched_fresh_external_search_rule",
            )

        if self._is_weather_lookup(text):
            return self._decision(
                "get_weather",
                text,
                confidence=0.86,
                reason="matched_weather_lookup_rule",
            )

        return None

    def _select_from_hints(
        self,
        query: str,
        profile: Optional[Any],
    ) -> Optional[ToolSelectionDecision]:
        if profile is None:
            return None

        hints: list[str] = []
        for attr in (
            "capability_hints",
            "tool_necessity_allowed_skill_hints",
        ):
            for value in list(getattr(profile, attr, []) or []):
                if isinstance(value, str) and value not in hints:
                    hints.append(value)

        for hint in hints:
            tool_name = self._normalize_tool_hint(hint)
            if tool_name:
                return self._decision(
                    tool_name,
                    query,
                    confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
                    reason="matched_atomic_tool_hint",
                )
        return None

    @staticmethod
    def _normalize_tool_hint(hint: str) -> str:
        if hint in {"get_weather", "get_nasa_apod", "web_search"}:
            return hint
        return ""

    @staticmethod
    def _is_apod_lookup(text: str) -> bool:
        has_reference = "APOD" in text or "每日天文图" in text
        if not has_reference:
            return False
        if any(token in text for token in ("什么意思", "怎么理解")):
            return False
        return any(
            token in text
            for token in (
                "今天",
                "今日",
                "昨天",
                "明天",
                "日期",
                "查",
                "查询",
                "图片",
                "是什么",
            )
        ) or bool(re.search(r"\d{4}-\d{2}-\d{2}", text))

    @staticmethod
    def _is_web_search_lookup(text: str) -> bool:
        if any(token in text for token in ("天象", "流星雨", "月食", "日食", "合月")):
            return False
        has_freshness = any(
            token in text for token in ("最近", "最新", "新闻", "新结果", "新发现")
        )
        has_external_topic = any(
            token in text
            for token in ("天文", "韦布", "詹姆斯 Webb", "JWST", "发现", "结果")
        )
        return has_freshness and has_external_topic

    @staticmethod
    def _is_weather_lookup(text: str) -> bool:
        if any(token in text for token in ("同时", "并且", "以及", "天象")):
            return False
        if "并" in text and any(token in text for token in ("摄影", "计划", "安排")):
            return False

        has_place = any(city in text for city in KNOWN_CITIES) or bool(
            re.search(r"\d{1,2}(?:\.\d+)?\s*[,，]\s*\d{2,3}(?:\.\d+)?", text)
        )
        has_when = any(
            token in text for token in ("今晚", "明晚", "今天", "明天", "当前", "现在")
        )
        has_weather = any(
            token in text
            for token in (
                "天气",
                "云量",
                "云多",
                "云少",
                "下雨",
                "降雨",
                "观测条件",
                "适合观星",
                "适合出门观星",
                "适合架望远镜",
            )
        )
        return has_place and has_when and has_weather

    @staticmethod
    def _decision(
        tool_name: str,
        query: str,
        *,
        confidence: float,
        reason: str,
    ) -> ToolSelectionDecision:
        return ToolSelectionDecision(
            mode="atomic_tool",
            tool_name=tool_name,
            params=AtomicToolParamAdapter.build(tool_name, query),
            confidence=confidence,
            reason=reason,
        )
