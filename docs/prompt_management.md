# Prompt Management

AstroAgent 的 LLM 提示词统一由 `config/prompts/manifest.yaml` 注册，由
`src.agent.prompts.PromptRenderer` 渲染。业务代码不应再内联大段 prompt 字符串。

## 目录约定

- `config/prompts/manifest.yaml`：prompt id、版本、模板路径、变量、输出契约和预算配置
- `config/prompts/main.txt`：历史 ReAct 模板，仅作为 registry 不可用时的 legacy fallback
- `config/prompts/shared/`：可复用角色、领域默认值、工具策略、回答风格
- `config/prompts/react/`：ReAct Agent 模板
- `config/prompts/direct/`：direct/simple_qa 和 no-tool 回答模板
- `config/prompts/planned/`：计划生成和答案合成模板
- `config/prompts/router/`：LLM 意图分类模板
- `config/prompts/memory/`：长期记忆与 task_state 抽取模板
- `config/prompts/vision/`：图像理解模板

## 模板语法

Renderer 支持一层轻量模板语法：

- `{{ variable }}`：替换运行时变量
- `{% include "shared/persona.zh.md" %}`：引入共享片段

LangChain ReAct 模板中的 `{input}`、`{tools}`、`{agent_scratchpad}` 使用单花括号，
renderer 会原样保留，交给 `PromptTemplate` 绑定。

## 代码使用

普通模板：

```python
prompt = get_prompt_renderer().render(
    "router.intent_classifier",
    {"query": query, "skills_text": skills_text, ...},
)
```

带预算 section 的模板：

```python
prompt = get_prompt_renderer().render_sections(
    "planned.response_synthesizer",
    {"query": query, "tool_outputs": tool_outputs, ...},
)
```

`render_sections()` 会读取 manifest 中的 section 优先级、required 标记和
`max_chars`，并复用现有 `PromptBudgetManager`。

## 变更规则

1. 新增 LLM 调用时，先在 manifest 注册 prompt id，再创建模板文件。
2. 修改 prompt 时同步更新 manifest 中的 `version`。
3. JSON-only prompt 必须声明 `output_contract`，并在模板里明确“只输出 JSON”。
4. 不要在业务代码里新增超过一两行的 prompt 字符串；短错误提示和用户澄清文本可以保留在代码中。
5. 迁移或新增 prompt 后，补充 `tests/unit/test_prompt_registry.py` 或对应调用点测试。
