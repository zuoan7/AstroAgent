from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from src.agent.policies.prompt_budget import PromptBudgetManager, PromptSection
from src.core.config import resolve_path, settings


class PromptRenderError(RuntimeError):
    """Raised when a registered prompt cannot be loaded or rendered."""


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    version: str = ""
    template: str = ""
    required_vars: tuple[str, ...] = ()
    optional_vars: tuple[str, ...] = ()
    output_contract: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    sections: tuple[dict[str, Any], ...] = ()


class PromptRenderer:
    """Registry-backed prompt renderer.

    The renderer intentionally supports a small template language:
    - ``{{ variable }}`` substitution for runtime values.
    - ``{% include "relative/path.md" %}`` for shared prompt fragments.

    Single-brace placeholders such as ``{input}`` are left untouched so LangChain
    PromptTemplate can still bind them later.
    """

    _VAR_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*}}")
    _INCLUDE_RE = re.compile(r'{%\s*include\s+"([^"]+)"\s*%}')

    def __init__(self, manifest_path: Optional[str] = None) -> None:
        self._manifest_path = Path(
            resolve_path(manifest_path or settings.PROMPT_MANIFEST_PATH)
        )
        self._prompt_root = self._manifest_path.parent
        self._specs = self._load_manifest()

    def spec(self, prompt_id: str) -> PromptSpec:
        try:
            return self._specs[prompt_id]
        except KeyError as exc:
            raise PromptRenderError(f"unknown prompt id: {prompt_id}") from exc

    def render(
        self,
        prompt_id: str,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        validate: bool = True,
    ) -> str:
        spec = self.spec(prompt_id)
        if not spec.template:
            raise PromptRenderError(f"prompt has no template: {prompt_id}")
        values = dict(variables or {})
        if validate:
            self._validate_vars(spec, values)
        template = self._read_template(spec.template)
        return self.render_text(template, values)

    def render_text(self, template: str, variables: Mapping[str, Any]) -> str:
        text = self._resolve_includes(template, seen=())

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            value = self._lookup(variables, key)
            if value is None:
                return ""
            return str(value)

        return self._VAR_RE.sub(replace, text)

    def render_sections(
        self,
        prompt_id: str,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        use_budget: Optional[bool] = None,
    ) -> str:
        spec = self.spec(prompt_id)
        if not spec.sections:
            return self.render(prompt_id, variables)

        values = dict(variables or {})
        self._validate_vars(spec, values)
        sections: list[PromptSection] = []
        for item in spec.sections:
            name = str(item.get("name") or "").strip()
            if not name:
                raise PromptRenderError(f"section without name in {prompt_id}")

            content = self._section_content(item, values)
            sections.append(
                PromptSection(
                    name=name,
                    content=content,
                    priority=int(item.get("priority", 50)),
                    required=bool(item.get("required", False)),
                    min_chars=int(item.get("min_chars", 0) or 0),
                    max_chars=item.get("max_chars"),
                    preserve_head=bool(item.get("preserve_head", True)),
                )
            )

        budget_enabled = (
            settings.PROMPT_BUDGET_ENABLED if use_budget is None else use_budget
        )
        if budget_enabled:
            max_chars = spec.budget.get("max_chars")
            result = PromptBudgetManager().fit_sections(
                sections,
                max_chars=int(max_chars) if max_chars else None,
            )
            return result.text

        rendered = []
        for section in sections:
            content = section.content
            if section.max_chars and len(content) > int(section.max_chars):
                content = content[: int(section.max_chars)]
            rendered.append(f"=== {section.name} ===\n{content}")
        return "\n\n".join(rendered)

    def version(self, prompt_id: str) -> str:
        return self.spec(prompt_id).version

    def _load_manifest(self) -> dict[str, PromptSpec]:
        if not self._manifest_path.exists():
            raise PromptRenderError(f"prompt manifest not found: {self._manifest_path}")
        with self._manifest_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        prompts = payload.get("prompts") or {}
        specs: dict[str, PromptSpec] = {}
        for prompt_id, raw in prompts.items():
            item = dict(raw or {})
            specs[prompt_id] = PromptSpec(
                prompt_id=prompt_id,
                version=str(item.get("version") or ""),
                template=str(item.get("template") or ""),
                required_vars=tuple(item.get("required_vars") or ()),
                optional_vars=tuple(item.get("optional_vars") or ()),
                output_contract=str(item.get("output_contract") or ""),
                budget=dict(item.get("budget") or {}),
                sections=tuple(dict(x or {}) for x in item.get("sections") or ()),
            )
        return specs

    def _validate_vars(self, spec: PromptSpec, values: Mapping[str, Any]) -> None:
        missing = [key for key in spec.required_vars if key not in values]
        if missing:
            raise PromptRenderError(
                f"prompt {spec.prompt_id} missing variables: {', '.join(missing)}"
            )

    def _read_template(self, relative_path: str) -> str:
        path = (self._prompt_root / relative_path).resolve()
        if self._prompt_root.resolve() not in path.parents and path != self._prompt_root:
            raise PromptRenderError(f"template path escapes prompt root: {relative_path}")
        if not path.exists():
            raise PromptRenderError(f"template not found: {path}")
        return path.read_text(encoding="utf-8")

    def _resolve_includes(self, text: str, *, seen: tuple[str, ...]) -> str:
        def replace(match: re.Match[str]) -> str:
            relative_path = match.group(1)
            if relative_path in seen:
                chain = " -> ".join((*seen, relative_path))
                raise PromptRenderError(f"recursive prompt include: {chain}")
            included = self._read_template(relative_path)
            return self._resolve_includes(included, seen=(*seen, relative_path))

        return self._INCLUDE_RE.sub(replace, text)

    def _section_content(
        self,
        section: Mapping[str, Any],
        variables: Mapping[str, Any],
    ) -> str:
        if "variable" in section:
            value = self._lookup(variables, str(section["variable"]))
            return "" if value is None else str(value)
        if "template" in section:
            template = self._read_template(str(section["template"]))
            return self.render_text(template, variables)
        if "content" in section:
            return self.render_text(str(section["content"]), variables)
        return ""

    @staticmethod
    def _lookup(values: Mapping[str, Any], key: str) -> Any:
        current: Any = values
        for part in key.split("."):
            if isinstance(current, Mapping):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                return None
        return current


@lru_cache(maxsize=1)
def get_prompt_renderer() -> PromptRenderer:
    return PromptRenderer()
