from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FinalResponse:
    answer: str
    summary: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    structured_payload: Optional[Dict[str, Any]] = None
    route: str = ""
    task_type: str = ""
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    execution_plan: Optional[Dict[str, Any]] = None
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    execution_events: List[Dict[str, Any]] = field(default_factory=list)
    route_decision: Optional[Dict[str, Any]] = None
    latency_profile: Optional[Dict[str, Any]] = None
    fallback_path: List[Dict[str, Any]] = field(default_factory=list)
    audit_metadata: Optional[Dict[str, Any]] = None
    budget_usage: Optional[Dict[str, Any]] = None
    versions: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "answer": self.answer,
            "summary": self.summary,
            "sources": self.sources,
            "tools_used": self.tools_used,
            "confidence": self.confidence,
            "route": self.route,
            "task_type": self.task_type,
        }
        if self.structured_payload:
            result["structured_payload"] = self.structured_payload
        if self.memory_hits:
            result["memory_hits"] = self.memory_hits
        if self.execution_plan:
            result["execution_plan"] = self.execution_plan
        if self.execution_trace:
            result["execution_trace"] = self.execution_trace
        if self.execution_events:
            result["execution_events"] = self.execution_events
        if self.route_decision:
            result["route_decision"] = self.route_decision
        if self.latency_profile:
            result["latency_profile"] = self.latency_profile
        if self.fallback_path:
            result["fallback_path"] = self.fallback_path
        if self.audit_metadata:
            result["audit_metadata"] = self.audit_metadata
        if self.budget_usage:
            result["budget_usage"] = self.budget_usage
        if self.versions:
            result["versions"] = self.versions
        return result

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "route": self.route,
            "task_type": self.task_type,
            "sources": self.sources,
            "tools_used": self.tools_used,
            "memory_hits": self.memory_hits,
            "execution_plan": self.execution_plan,
            "execution_trace": self.execution_trace,
            "execution_events": self.execution_events,
            "route_decision": self.route_decision,
            "latency_profile": self.latency_profile,
            "fallback_path": self.fallback_path,
            "audit_metadata": self.audit_metadata,
            "budget_usage": self.budget_usage,
            "versions": self.versions,
        }
