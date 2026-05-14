"""PromptBudgetManager — unified prompt-section budget fitting.

Phase 2: character-based budget. No external tokenizer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.core.config import settings
from src.core.logger import logger

SECTION_HEADER_TEMPLATE = "=== {name} ===\n{content}"


@dataclass
class PromptSection:
    """A single prompt section with priority and constraints.

    Attributes:
        name: Section identifier for logging and rendering.
        content: Raw section text.
        priority: Higher = more important when budget is tight.
        required: If True, section is preserved even if trimming is necessary.
        min_chars: Floor below which the section is never trimmed.
        max_chars: Cap applied before global fitting.
        preserve_head: True → keep beginning; False → keep end.
    """

    name: str
    content: str
    priority: int = 50
    required: bool = False
    min_chars: int = 0
    max_chars: Optional[int] = None
    preserve_head: bool = True

    @property
    def size(self) -> int:
        return len(self.content)


@dataclass
class PromptBudgetResult:
    """Result of a budget-fitting operation.

    Attributes:
        text: The final assembled prompt text.
        total_chars_before: Sum of all section char counts before fitting.
        total_chars_after: Final prompt text length.
        trimmed_sections: Names of sections that were partially trimmed.
        dropped_sections: Names of sections that were entirely dropped.
        section_char_counts: Final char count per section.
    """

    text: str = ""
    total_chars_before: int = 0
    total_chars_after: int = 0
    raw_total_chars_before: int = 0
    trimmed_sections: List[str] = field(default_factory=list)
    dropped_sections: List[str] = field(default_factory=list)
    section_char_counts: Dict[str, int] = field(default_factory=dict)


class PromptBudgetManager:
    """Character-based prompt section budget manager.

    Usage::

        mgr = PromptBudgetManager()
        sections = [
            PromptSection("instruction", inst, priority=100, required=True),
            PromptSection("query", query, priority=100, required=True),
            PromptSection("chat_history", hist, priority=60, max_chars=1200),
        ]
        result = mgr.fit_sections(sections, max_chars=6000)
        prompt = result.text
    """

    def __init__(
        self,
        *,
        max_chars: Optional[int] = None,
        section_min_chars: Optional[int] = None,
        log_trimmed: Optional[bool] = None,
    ) -> None:
        self._max_chars = max_chars or settings.PROMPT_MAX_INPUT_CHARS
        self._section_min_chars = (
            section_min_chars
            if section_min_chars is not None
            else settings.PROMPT_SECTION_MIN_CHARS
        )
        self._log_trimmed = (
            log_trimmed
            if log_trimmed is not None
            else settings.PROMPT_BUDGET_LOG_TRIMMED
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_sections(
        self,
        sections: Sequence[PromptSection],
        max_chars: Optional[int] = None,
    ) -> PromptBudgetResult:
        """Fit *sections* into *max_chars* by priority.

        Strategy:
        1. Apply per-section max_chars caps first.
        2. Sort by priority (descending), required sections always first.
        3. Accumulate sections; when budget is exhausted, trim/drop low-priority
           sections.
        """
        budget = max_chars if max_chars is not None else self._max_chars
        budget = max(budget, 1)

        # --- step 1: per-section cap ---
        raw_total_before = sum(len(sec.content or "") for sec in sections)
        capped: List[PromptSection] = []
        for sec in sections:
            content = sec.content or ""
            if sec.max_chars is not None and len(content) > sec.max_chars:
                content = self.fit_text(
                    sec.name, content, sec.max_chars, sec.preserve_head
                )
            capped.append(
                PromptSection(
                    name=sec.name,
                    content=content,
                    priority=sec.priority,
                    required=sec.required,
                    min_chars=sec.min_chars,
                    preserve_head=sec.preserve_head,
                )
            )

        total_before = sum(sec.size for sec in capped)

        # --- step 2: sort ---
        ordered = sorted(
            capped,
            key=lambda s: (0 if s.required else 1, -s.priority),
        )

        # --- step 3: accumulate ---
        rendered: List[str] = []
        trimmed: List[str] = []
        dropped: List[str] = []
        final_sizes: Dict[str, int] = {}

        def _current_total() -> int:
            return len("\n\n".join(rendered)) if rendered else 0

        for sec in ordered:
            # Build a candidate block
            header = SECTION_HEADER_TEMPLATE.format(name=sec.name, content="")
            header_chars = len(header)
            sep_overhead = 2 if rendered else 0  # "\n\n" before this section
            remaining = budget - _current_total() - sep_overhead

            if sec.required:
                if remaining <= header_chars:
                    # Barely any room — use header-only
                    block = SECTION_HEADER_TEMPLATE.format(name=sec.name, content="")
                    rendered.append(block)
                    final_sizes[sec.name] = 0
                    trimmed.append(sec.name)
                    continue

                available = remaining - header_chars
                content_chars = len(sec.content)
                if content_chars <= available:
                    block = SECTION_HEADER_TEMPLATE.format(name=sec.name, content=sec.content)
                    rendered.append(block)
                    final_sizes[sec.name] = content_chars
                else:
                    fit_chars = max(self._section_min_chars, sec.min_chars, available)
                    bare = self._trim_to(sec.content, fit_chars, sec.preserve_head)
                    block = SECTION_HEADER_TEMPLATE.format(name=sec.name, content=bare)
                    rendered.append(block)
                    final_sizes[sec.name] = len(bare)
                    trimmed.append(sec.name)
                continue

            # --- non-required sections ---
            if remaining <= self._section_min_chars + header_chars:
                dropped.append(sec.name)
                final_sizes[sec.name] = 0
                continue

            available = remaining - header_chars
            content_chars = len(sec.content)
            if content_chars <= available:
                block = SECTION_HEADER_TEMPLATE.format(name=sec.name, content=sec.content)
                rendered.append(block)
                final_sizes[sec.name] = content_chars
            else:
                fit_chars = max(self._section_min_chars, sec.min_chars, min(available, content_chars))
                if fit_chars < self._section_min_chars:
                    dropped.append(sec.name)
                    final_sizes[sec.name] = 0
                    continue
                bare = self._trim_to(sec.content, fit_chars, sec.preserve_head)
                block = SECTION_HEADER_TEMPLATE.format(name=sec.name, content=bare)
                rendered.append(block)
                final_sizes[sec.name] = len(bare)
                trimmed.append(sec.name)

        prompt_text = "\n\n".join(rendered)

        # Final safety: if still over budget, trim last non-required section
        while len(prompt_text) > budget and rendered:
            # Remove last rendered block if it's non-required
            last_name = None
            for sec in reversed(ordered):
                name = sec.name
                if name in dropped or name not in final_sizes:
                    continue
                if final_sizes.get(name, 0) == 0:
                    continue
                last_name = name
                break
            if last_name is None:
                break
            # Find and remove last matching rendered block
            for i in range(len(rendered) - 1, -1, -1):
                if rendered[i].startswith(f"=== {last_name} ==="):
                    dropped.append(last_name)
                    final_sizes[last_name] = 0
                    rendered.pop(i)
                    break
            else:
                break
            prompt_text = "\n\n".join(rendered)

        if self._log_trimmed and (trimmed or dropped):
            logger.debug(
                "prompt_budget: trimmed=%s dropped=%s before=%d after=%d budget=%d",
                trimmed,
                dropped,
                total_before,
                len(prompt_text),
                budget,
            )

        return PromptBudgetResult(
            text=prompt_text,
            total_chars_before=total_before,
            total_chars_after=len(prompt_text),
            raw_total_chars_before=raw_total_before,
            trimmed_sections=list(dict.fromkeys(trimmed)),
            dropped_sections=list(dict.fromkeys(dropped)),
            section_char_counts=final_sizes,
        )

    def fit_text(
        self,
        name: str,
        text: str,
        max_chars: int,
        preserve_head: bool = True,
    ) -> str:
        """Trim *text* to at most *max_chars* characters."""
        return self._trim_to(text, max_chars, preserve_head)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to(text: str, max_chars: int, preserve_head: bool) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        if preserve_head:
            return text[: max_chars - 3] + "..."
        return "..." + text[-(max_chars - 3):]
