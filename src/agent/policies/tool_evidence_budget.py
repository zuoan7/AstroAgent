"""ToolEvidenceCompactor — tool result evidence budget governance.

Phase 4: per-tool cap, total cap, success/source priority, error truncation.
Character-based; no external tokenizer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ToolEvidenceItem:
    """A single tool result as seen by the compactor."""

    tool_name: str
    status: str  # "success" | "error"
    summary: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 50
    raw_size_chars: int = 0
    is_truncated: bool = False
    error_message: str = ""

    @classmethod
    def from_skill_result(cls, sr: Any) -> "ToolEvidenceItem":
        """Build from a SkillResult."""
        summary = str(sr.summary or "") if sr.success else ""
        return cls(
            tool_name=sr.skill_name,
            status="success" if sr.success else "error",
            summary=summary,
            sources=list(sr.sources or []),
            priority=60 if sr.success else 30,
            raw_size_chars=len(summary),
            error_message=str(sr.error_message or ""),
        )


@dataclass
class ToolEvidenceBudgetResult:
    """Result of compacting tool evidence items."""

    text: str = ""
    total_chars_before: int = 0
    total_chars_after: int = 0
    raw_total_chars_before: int = 0
    trimmed_tools: List[str] = field(default_factory=list)
    dropped_tools: List[str] = field(default_factory=list)
    tool_char_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


class ToolEvidenceCompactor:
    """Compacts tool evidence for LLM prompt injection.

    - Caps individual tool summaries.
    - Caps total evidence length.
    - Prioritises successful tools and tools with sources.
    - Truncates error tools aggressively.
    """

    def __init__(
        self,
        *,
        max_single_chars: Optional[int] = None,
        max_total_chars: Optional[int] = None,
        error_max_chars: Optional[int] = None,
        compacted_max_chars: Optional[int] = None,
    ) -> None:
        self._max_single = (
            max_single_chars
            if max_single_chars is not None
            else settings.TOOL_EVIDENCE_MAX_SINGLE_CHARS
        )
        self._max_total = (
            max_total_chars
            if max_total_chars is not None
            else settings.TOOL_EVIDENCE_MAX_TOTAL_CHARS
        )
        self._error_max = (
            error_max_chars
            if error_max_chars is not None
            else settings.TOOL_EVIDENCE_ERROR_MAX_CHARS
        )
        self._compacted_max = (
            compacted_max_chars
            if compacted_max_chars is not None
            else settings.TOOL_EVIDENCE_COMPACTED_MAX_CHARS
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compact_skill_results(
        self,
        skill_results: List[Any],
        max_total_chars: Optional[int] = None,
    ) -> ToolEvidenceBudgetResult:
        """Build items from skill_results and compact them."""
        if not skill_results:
            return ToolEvidenceBudgetResult(text="")

        items = [ToolEvidenceItem.from_skill_result(sr) for sr in skill_results]
        return self.compact_items(items, max_total_chars=max_total_chars)

    def compact_items(
        self,
        items: List[ToolEvidenceItem],
        max_total_chars: Optional[int] = None,
    ) -> ToolEvidenceBudgetResult:
        """Compact *items* into a single evidence text block."""
        budget = (
            max_total_chars
            if max_total_chars is not None
            else self._max_total
        )
        budget = max(budget, 1)

        raw_total_before = sum(
            (len(item.summary or "") + len(item.error_message or ""))
            for item in items
        )
        trimmed: List[str] = []
        dropped: List[str] = []
        counts: Dict[str, int] = {}

        # --- step 1: per-item caps ---
        capped: List[ToolEvidenceItem] = []
        for item in items:
            cap_chars = self._error_max if item.status == "error" else self._max_single
            summary = item.summary or ""
            if len(summary) > cap_chars:
                summary = self._trim_to(summary, cap_chars)
                trimmed.append(item.tool_name)
            err_msg = item.error_message or ""
            if item.status == "error" and len(err_msg) > cap_chars:
                err_msg = self._trim_to(err_msg, cap_chars)
                if item.tool_name not in trimmed:
                    trimmed.append(item.tool_name)
            capped.append(
                ToolEvidenceItem(
                    tool_name=item.tool_name,
                    status=item.status,
                    summary=summary,
                    sources=item.sources,
                    priority=item.priority,
                    raw_size_chars=len(summary),
                    is_truncated=item.tool_name in trimmed,
                    error_message=err_msg,
                )
            )

        # --- step 2: sort by priority ---
        ordered = sorted(
            capped,
            key=lambda item: (
                item.priority + (10 if item.sources else 0),  # sources boost
            ),
            reverse=True,
        )

        # --- step 3: accumulate within budget ---
        lines: List[str] = []
        used = 0

        for item in ordered:
            if item.status == "error":
                err_text = item.error_message or item.summary
                chunk = f"[{item.tool_name}] error\nerror: {err_text}"
            else:
                chunk = f"[{item.tool_name}] success\n{item.summary}"

            join_overhead = 2 if lines else 0  # "\n\n" before this entry
            if used + join_overhead + len(chunk) <= budget:
                lines.append(chunk)
                used += join_overhead + len(chunk)
                counts[item.tool_name] = len(chunk)
            else:
                # Try trimmed version
                remaining = budget - used - join_overhead
                if remaining >= 40:
                    short = self._trim_to(chunk, remaining)
                    lines.append(short)
                    used += join_overhead + len(short)
                    counts[item.tool_name] = len(short)
                    trimmed.append(item.tool_name)
                else:
                    dropped.append(item.tool_name)
                    counts[item.tool_name] = 0

        text = "\n\n".join(lines)

        # --- step 4: optional secondary compaction ---
        if len(text) > self._compacted_max:
            text = self._trim_to(text, self._compacted_max)
            # All tools beyond the compacted max are effectively trimmed
            for item in ordered:
                if counts.get(item.tool_name, 0) > 0 and item.tool_name not in trimmed:
                    trimmed.append(item.tool_name)

        trimmed_dedup = list(dict.fromkeys(trimmed))
        dropped_dedup = list(dict.fromkeys(dropped))
        if trimmed_dedup or dropped_dedup:
            logger.debug(
                "tool_evidence_compactor: trimmed=%s dropped=%s before=%d after=%d budget=%d",
                trimmed_dedup,
                dropped_dedup,
                raw_total_before,
                len(text),
                budget,
            )

        return ToolEvidenceBudgetResult(
            text=text,
            total_chars_before=raw_total_before,
            total_chars_after=len(text),
            raw_total_chars_before=raw_total_before,
            trimmed_tools=trimmed_dedup,
            dropped_tools=dropped_dedup,
            tool_char_counts=counts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."
