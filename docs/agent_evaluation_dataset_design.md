# AstroAgent 评测数据集设计

## 1. 目标

本文定义 AstroAgent 的评测问题集与运行日志字段，用于支撑以下指标：

- 工具选择准确率
- 工具调用成功率
- 端到端任务成功率
- 首事件延迟
- 端到端 P95 响应时延

AstroAgent 同时存在两层工具抽象：

- 高层 Agent skill：例如 `weather-lookup`、`observation-planner`
- 底层 MCP tool：例如 `get_weather`、`get_weekly_events`

因此评测数据里应同时记录 `expected_skills` 和 `expected_mcp_tools`。这样可以区分路由错误、参数构建错误、MCP 执行错误和最终回答合成错误，避免把所有失败混成一个黑盒结果。

当前数据集骨架文件位于 `config/benchmarks/astro_agent_eval_dataset.json`，后续生成的问题统一填入其中的 `cases` 数组。

## 2. 问题集构建原则

建议将 benchmark 拆成两类可重叠的数据集：

- 能力评测集：关注路由、工具选择、参数抽取、回答正确性和任务完成度。
- 性能评测集：关注首事件延迟和端到端延迟，按真实任务比例混合采样。

问题集需要按以下维度分层采样：

| 维度 | 目的 |
| --- | --- |
| 工具覆盖 | 确保每个 skill 和关键 MCP tool 都被覆盖。 |
| 无工具覆盖 | 检测闲聊、纯解释任务是否误调用工具。 |
| 单工具 / 多工具 | 同时测试简单查询和规划编排。 |
| 明确 / 模糊请求 | 测试参数抽取和澄清策略。 |
| 正常 / 边界输入 | 覆盖非法日期、缺少地点、中英文天体名混用、未知天体等。 |
| 时效性 / 稳定性 | 区分天气、天象等动态任务和稳定知识问答。 |
| 单轮 / 多轮 | 测试上下文引用、记忆复用和旧信息抑制。 |
| 文本 / 多模态 | 图片诊断类任务和纯文本任务分开统计。 |
| 快 / 慢任务 | 让延迟分桶更可解释，避免不同复杂度任务混在一起看 P95。 |

初始比例建议：

| 类型 | 建议占比 |
| --- | ---: |
| 高频正常任务 | 60%-70% |
| 边界和异常任务 | 10%-15% |
| 多工具规划任务 | 10%-20% |
| 不应调用工具或应澄清任务 | 5%-10% |

## 3. Case 字段

每条评测问题建议包含以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `case_id` | string | 稳定唯一 ID，例如 `obs_plan_001`。 |
| `suite` | string | `ability`、`performance`、`regression` 或自定义 suite。 |
| `prompt` | string | 单轮问题的用户输入。 |
| `turns` | array | 多轮对话；多轮 case 优先使用该字段。 |
| `attachments` | array | 图片、音频等多模态输入。 |
| `category` | string | 一级任务类别。 |
| `subcategory` | string | 天文任务细分类型。 |
| `difficulty` | string | `easy`、`medium`、`hard`。 |
| `requires_tool` | boolean | 是否应调用工具。 |
| `should_clarify` | boolean | 是否应先询问缺失关键信息。 |
| `expected_route` | string | 期望路由，例如 `direct_task`、`planned_task`、`fallback_react`。 |
| `expected_skills` | array | 期望调用的高层 skill。 |
| `forbidden_skills` | array | 不应调用的 skill。 |
| `expected_skill_sequence` | array | 多工具任务中有顺序要求时填写。 |
| `expected_mcp_tools` | array | 可观测时填写期望底层 MCP tool。 |
| `forbidden_mcp_tools` | array | 不应调用的 MCP tool。 |
| `expected_params` | object | 关键参数期望值，按 skill 或 MCP tool 分组。 |
| `param_match_rule` | string | `exact`、`partial`、`semantic`、`range`、`tolerance`。 |
| `time_context` | object | 相对日期解析上下文，例如“今晚”“下个月”的基准日期。 |
| `geo_context` | object | 城市、经纬度、时区或观测站信息。 |
| `expected_answer_structure` | string | 期望回答结构或 schema。 |
| `success_criteria` | array | 端到端成功判据。 |
| `expected_answer` | string/object | 可选标准答案或参考答案。 |
| `judge_type` | string | `exact_match`、`rule`、`llm_judge`、`human` 或组合。 |
| `latency_bucket` | string | `fast`、`normal`、`slow`、`external_dependency`。 |
| `timeout_ms` | number | case 级超时。 |
| `weight` | number | 汇总指标权重。 |
| `tags` | array | 例如 `multi_tool`、`ambiguous`、`time_sensitive`、`image`、`negative`。 |
| `notes` | string | 标注说明。 |

最小可用 case 示例：

```json
{
  "case_id": "weather_001",
  "suite": "ability",
  "prompt": "帮我查一下今晚上海云量，并判断是否适合看月亮",
  "category": "observing_conditions",
  "subcategory": "weather_and_observability",
  "difficulty": "easy",
  "requires_tool": true,
  "expected_route": "direct_task",
  "expected_skills": ["weather-lookup"],
  "expected_params": {
    "weather-lookup": {
      "city": "上海",
      "extensions": "all"
    }
  },
  "param_match_rule": "semantic",
  "success_criteria": [
    "调用天气查询能力",
    "识别地点为上海",
    "回答包含云量或天气条件",
    "给出是否适合观测月亮的判断"
  ],
  "judge_type": "rule+llm_judge",
  "latency_bucket": "normal",
  "timeout_ms": 10000,
  "tags": ["single_tool", "time_sensitive"],
  "weight": 1
}
```

## 4. 运行 Trace 字段

评测 runner 应为每条 case 保存一次运行 trace：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `run_id` | string | 本次评测运行 ID。 |
| `case_id` | string | 对应数据集 case ID。 |
| `system_version` | string | Git SHA、镜像 tag 或 release 版本。 |
| `model_version` | string | LLM/model 标识。 |
| `config_snapshot` | object | 关键运行配置、feature flags、timeout、工具端点。 |
| `start_time` | timestamp | 请求开始时间。 |
| `first_event_time` | timestamp | 首个对客户端可见的流式事件时间。 |
| `first_token_time` | timestamp | 可选，首个文本 token 时间。 |
| `first_tool_call_time` | timestamp | 可选，首个工具调用开始时间。 |
| `end_time` | timestamp | 请求完成时间。 |
| `first_event_latency_ms` | number | `first_event_time - start_time`。 |
| `e2e_latency_ms` | number | `end_time - start_time`。 |
| `route` | string | 实际路由。 |
| `matched_skills` | array | 实际选择的高层 skill。 |
| `tool_calls` | array | 实际工具调用列表。 |
| `final_answer` | string | 最终回答。 |
| `tool_selection_correct` | boolean | 工具选择是否正确。 |
| `tool_call_success` | boolean | case 级工具执行是否成功。 |
| `task_success` | boolean | 端到端任务是否成功。 |
| `judge_reason` | string | 判分原因。 |

建议的 `tool_calls` 条目：

```json
{
  "skill_name": "weather-lookup",
  "mcp_tool_name": "get_weather",
  "args": {
    "city": "上海",
    "extensions": "all"
  },
  "status": "success",
  "started_at": "2026-05-18T20:00:00+08:00",
  "ended_at": "2026-05-18T20:00:01+08:00",
  "latency_ms": 1000,
  "error": null
}
```

## 5. 指标口径

### 5.1 工具选择准确率

工具选择准确率衡量 Agent 在执行前是否选择了正确的 skill 或 MCP tool。

建议拆成以下指标：

- `skill_selection_exact_accuracy`：实际 skill 集合与 `expected_skills` 完全一致。
- `skill_selection_partial_accuracy`：必要 skill 都出现，允许额外但无害的 skill。
- `skill_false_positive_rate`：调用了不必要或被禁止的 skill。
- `mcp_tool_selection_accuracy`：底层 MCP tool 匹配率，前提是 trace 可观测。

多工具任务同时统计无序集合匹配和有序序列匹配；只有填写了 `expected_skill_sequence` 时才计算序列匹配。

### 5.2 工具调用成功率

工具调用成功率衡量工具被选中之后是否成功执行。

建议同时统计：

- per-call 成功率：成功工具调用次数 / 总工具调用次数。
- per-case 成功率：所有必需工具调用都成功的 case 数 / 需要工具的 case 数。
- timeout rate：工具超时率。
- external dependency error rate：外部 API 或网络依赖导致的失败率。

### 5.3 端到端任务成功率

端到端成功率衡量最终面向用户的回答是否满足 `success_criteria`。

它应综合以下因素：

- 路由和工具选择正确
- 参数抽取正确
- 工具执行成功
- 最终回答忠实使用工具证据
- 回答结构完整、对用户任务有用

### 5.4 首事件延迟

首事件延迟定义为：

```text
first_event_time - start_time
```

这里的 first event 是客户端可见的第一个 SSE / JSON stream event / 等价流式事件。如果系统会先发 planning event，再发文本 token，建议同时报告 `first_event_latency_ms` 和 `first_token_latency_ms`。

### 5.5 端到端 P95 响应时延

端到端时延定义为：

```text
end_time - start_time
```

P95 建议同时按全量和分桶统计：

- `fast`：闲聊、无工具、简单稳定知识问答。
- `normal`：单工具查询或短回答合成。
- `slow`：多工具规划任务。
- `external_dependency`：天气、NASA、搜索等外部 API 主导的任务。

## 6. 天文 Agent 问题类型细分

AstroAgent 的数据集不应只停留在 `simple_qa`、`single_tool_lookup`、`planned_task` 三类。建议使用以下细分 taxonomy。

### 6.1 控制类和无工具类

用于检测是否过度调用工具。

| 子类型 | 示例 | 期望行为 |
| --- | --- | --- |
| `smalltalk` | `你好` | 不调用工具，简短回复。 |
| `meta_capability` | `你能帮我做哪些天文相关事情？` | 通常不调用外部工具。 |
| `stable_concept_short_answer` | `光年是什么？` | 根据评测策略走 RAG 或直接回答。 |
| `creative_or_open_ended` | `写一段关于星空的短文` | 不需要天文计算工具。 |
| `out_of_domain` | `帮我分析股票走势` | 按产品策略拒绝、转向或 fallback。 |

### 6.2 天文知识问答

主要测试稳定天文知识解释质量和幻觉控制。当前 `RAGRetrieve` 仍是占位能力，不作为强制工具选择要求；稳定知识题允许直接回答，后续 RAG 完整实现后可新增单独的 RAG 评测 slice。

| 子类型 | 示例 |
| --- | --- |
| `concept_definition` | `赤经和赤纬是什么？` |
| `concept_comparison` | `视星等和绝对星等有什么区别？` |
| `mechanism_explanation` | `为什么不同季节看到的星空不同？` |
| `observing_term_explanation` | `视宁度和透明度分别影响什么？` |
| `equipment_knowledge` | `折射望远镜和反射望远镜有什么区别？` |
| `astrophysics_basic` | `红移为什么能说明宇宙膨胀？` |
| `myth_or_history` | `猎户座有哪些主要亮星？` |
| `astronomy_units_and_scales` | `光年到底有多远？` |
| `coordinate_and_time_systems` | `恒星时和普通时间有什么关系？` |
| `celestial_mechanics_basic` | `行星逆行是真的倒着走了吗？` |
| `photometry_and_magnitude` | `视星等越小是不是越亮？` |
| `spectroscopy_and_redshift` | `天文学家怎么从光谱里看出恒星成分？` |
| `stellar_evolution` | `恒星为什么会变成红巨星？` |
| `cosmology_basic` | `宇宙膨胀是不是说星系在空间里飞得越来越远？` |
| `classification_taxonomy` | `矮行星和行星差在哪里？` |
| `common_misconception_correction` | `北极星是不是夜空中最亮的星？` |
| `observation_method_knowledge` | `暗适应为什么对观星很重要？` |

当前期望通常是无工具直接回答；`RAGRetrieve` 可作为可选能力，不计入强制工具命中。

### 6.3 天体信息查询

测试天体识别、别名解析和对象元数据查询。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `solar_system_object_info` | `木星有什么观测特征？` | 可直接回答；若问时间地点则用位置计算。 |
| `deep_sky_object_info` | `M31 是什么天体？` | `deep-sky-observing-guide`，或直接回答基本信息。 |
| `catalog_alias_resolution` | `仙女座星系和 M31 是同一个吗？` | 可直接回答，必要时对象查询。 |
| `galaxy_data_lookup` | `M31 的距离和类型是什么？` | `deep-sky-observing-guide`，可观测到底层 `get_galaxy_data`。 |
| `unsupported_object` | `帮我查 XYZ123 星云` | 澄清或优雅返回未找到。 |

### 6.4 位置、坐标和可见性

测试天文计算和参数抽取。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `planet_position` | `今晚北京木星在哪个方向？` | `celestial-position-calculator`，底层可到 `get_altaz` / `get_planet_position`。 |
| `rise_set_time` | `土星今晚几点升起？` | `celestial-position-calculator`，底层可到 `get_rise_set_times`。 |
| `current_visible_objects` | `现在上海能看到哪些亮目标？` | `celestial-position-calculator` 或 `observation-planner`。 |
| `coordinate_conversion` | `把这个赤经赤纬转成指定坐标系` | `coordinate_transformation`。 |
| `relative_date_resolution` | `明晚 9 点火星高度是多少？` | 正确解析日期并调用位置工具。 |
| `location_missing` | `今晚木星在哪？` | 当前策略使用默认地点北京。 |
| `best_observation_window` | `今晚北京看木星，几点高度比较合适？` | 结合高度变化给出较好观测时段。 |
| `deep_sky_target_visibility` | `今晚上海能看到 M31 吗？` | 判断深空目标在指定地点和日期的可见性。 |
| `moon_sun_visibility` | `今晚月亮什么时候升起来？` | 处理月升月落、日落和暮光相关问题。 |
| `coordinate_format_parsing` | `RA 5h35m17s、Dec -5°23′28″ 这个坐标怎么理解？` | 解析赤经赤纬格式和单位。 |
| `location_coordinate_input` | `我在 31.2,121.5，今晚土星在哪个方向？` | 支持经纬度形式的位置输入。 |
| `visibility_threshold_judgment` | `如果木星只有 12 度高度，还值得看吗？` | 根据高度阈值和观测质量给出判断。 |

### 6.5 观测条件

测试天气、月相、光害和可观测性判断。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `weather_lookup` | `上海今晚云量如何？` | `weather-lookup`，底层 `get_weather`。 |
| `weather_observability_judgment` | `今晚北京适合看月亮吗？` | `weather-lookup` 加回答合成。 |
| `light_pollution_context` | `城市阳台能看哪些目标？` | `observation-planner`，可结合通用知识。 |
| `moon_phase_impact` | `满月会不会影响看 M31？` | `celestial-events-forecast` 或知识问答。 |
| `seeing_transparency_advice` | `透明度差但视宁度好适合看什么？` | 稳定知识解释加观测建议。 |
| `cloud_cover_window` | `今晚北京会不会有一段时间云少一点？` | 查询天气预报并判断是否有短时观测窗口。 |
| `wind_and_equipment_stability` | `今晚风大还适合架望远镜吗？` | 判断风对设备稳定性和安全性的影响。 |
| `humidity_dew_risk` | `湿度很高会不会镜头起雾？` | 评估结露、起雾和防露措施。 |
| `haze_air_quality_transparency` | `天气晴但有雾霾，还适合看星吗？` | 区分晴天和透明度，判断雾霾影响。 |
| `site_condition_comparison` | `市区楼顶和郊区公园哪个更适合今晚观星？` | 比较不同观测地点的光害、视野和安全条件。 |
| `forecast_uncertainty` | `预报说多云，还值得带设备出门吗？` | 面对不确定预报给出轻装或窗口期决策建议。 |

### 6.6 天象事件

测试事件检索、日期范围、事件筛选和排序。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `weekly_events` | `这周有什么天象？` | `celestial-events-forecast`，底层 `get_weekly_events`。 |
| `monthly_events` | `2026 年 6 月有哪些天象？` | `celestial-events-forecast`，底层 `get_monthly_events`。 |
| `event_type_filter` | `最近有流星雨吗？` | `celestial-events-forecast`。 |
| `event_observability_ranking` | `本月最值得看的三个天象是什么？` | `celestial-events-forecast`，可结合天气。 |
| `public_outreach_event_list` | `给新手整理一份本月公众可看的天象清单` | 规划型回答。 |
| `meteor_shower_peak_window` | `英仙座流星雨一般什么时候看最好？` | 解释流星雨极大、辐射点高度和最佳观看时段。 |
| `eclipse_visibility` | `月食是不是全国都能看到？` | 判断或解释日月食的地区可见性。 |
| `planetary_conjunction` | `行星合月到底是什么现象？` | 解释行星、月亮等天体视觉接近的现象。 |
| `event_time_window` | `天象预报里的“极大”是什么意思？` | 解释事件时间窗口、极大和提前准备。 |
| `missed_event_followup` | `如果错过流星雨极大，第二天还值得看吗？` | 给出错过峰值后的补看判断。 |
| `region_visibility_uncertainty` | `天象预报说可见，我在城市里也能看到吗？` | 处理预报可见和实际城市观测之间的差异。 |

### 6.7 观测计划

这是最重要的端到端任务类型。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `tonight_best_targets` | `今晚看什么？` | `observation-planner`，底层可到 `get_tonight_best`。 |
| `beginner_plan` | `我在北京，只有双筒，今晚适合看什么？` | `weather-lookup` + `observation-planner`。 |
| `limited_time_plan` | `只有半小时，推荐 3 个亮目标` | `observation-planner`，必要时位置计算。 |
| `equipment_aware_plan` | `80mm 小折射镜今晚怎么安排？` | `observation-planner` + `deep-sky-observing-guide`。 |
| `compare_observing_options` | `城市阳台和郊外今晚分别适合看什么？` | 多工具规划。 |
| `multi_night_plan` | `帮我做一个本周末两晚的观测计划` | 天气 + 天象 + 观测计划。 |
| `urban_balcony_plan` | `我只能在城市阳台看，今晚有什么不太受光害影响的目标？` | 面向城市阳台和光害约束生成目标建议。 |
| `moon_phase_aware_plan` | `今晚月亮比较亮，观测计划要怎么调整？` | 根据月光影响调整目标和顺序。 |
| `family_or_public_plan` | `我想带孩子看星星，今晚安排什么目标比较容易有惊喜？` | 面向儿童、朋友或公众活动安排易懂目标。 |
| `backup_plan` | `如果今晚云突然变多，观测计划怎么备选？` | 给出天气或目标失败时的备选方案。 |
| `target_sequence_plan` | `今晚我想从容易找到的目标开始，观测顺序怎么排？` | 按难度、兴趣和暗适应安排观测顺序。 |
| `pre_observation_preparation` | `今晚出门观星前需要提前准备什么？` | 输出出门前设备、安全和环境准备清单。 |
| `site_arrival_timing` | `今晚去郊区观星，最好提前多久到地方？` | 给出到场时间、架设、暗适应和安全安排。 |

### 6.8 深空观测指导

测试目标级观测建议和器材约束。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `dso_single_target_guide` | `M31 适合怎么观测？` | `deep-sky-observing-guide`。 |
| `dso_equipment_constraint` | `用 8 寸 DOB 看 M42 要注意什么？` | `deep-sky-observing-guide`。 |
| `dso_target_comparison` | `M31 和 M42 今晚哪个更适合？` | 深空指导 + 位置计算。 |
| `dso_seasonality` | `春季适合看哪些星系？` | 可直接回答或深空指导。 |
| `dso_find_method` | `怎么用跳星法找到 M13？` | 可直接回答或深空指导。 |
| `dso_light_pollution_moon_impact` | `满月的时候还适合看 M31 吗？` | 判断月光、光害对深空目标的影响。 |
| `dso_magnification_filter_advice` | `UHC 滤镜对看星云帮助大吗？` | 给出倍率、滤镜和目标类型匹配建议。 |
| `dso_site_comparison` | `M31 在城市阳台和郊区看起来差别大吗？` | 比较不同观测地点对深空目标的影响。 |
| `dso_target_selection` | `新手想第一次看深空，应该先选星团还是星云？` | 面向新手选择合适的深空目标类型。 |

### 6.9 天文摄影

测试摄影参数抽取、器材约束和拍摄流程建议。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `widefield_exposure` | `星野摄影曝光怎么设？` | `astrophotography-calculator`。 |
| `planetary_imaging` | `城市阳台拍木星怎么设置？` | 摄影参数 + 位置计算。 |
| `deep_sky_imaging_plan` | `拍 M42 给我一套曝光和叠加方案` | 摄影参数 + 深空指导。 |
| `camera_lens_rule` | `全画幅 24mm 拍银河曝光多久不会拖线？` | `astrophotography-calculator`。 |
| `workflow_plan` | `制定一个拍银河的分阶段方案` | 多工具规划。 |
| `image_diagnosis` | `这张月球照片为什么糊？` | 多模态诊断，必要时摄影参数工具。 |
| `focus_framing` | `晚上拍星星怎么准确对焦？` | 处理夜间对焦、前景和星空构图问题。 |
| `tracking_polar_alignment` | `极轴没对准会对深空照片造成什么影响？` | 诊断跟踪、极轴和星点拖线问题。 |
| `calibration_frames` | `暗场、平场、偏置帧分别是干什么的？` | 解释校准帧用途和拍摄取舍。 |
| `light_pollution_filter` | `城市里拍星云，光害滤镜有必要买吗？` | 判断滤镜和目标类型的匹配关系。 |
| `equipment_matching` | `只有入门单反和三脚架，适合先拍什么天体？` | 根据器材限制推荐入门拍摄目标。 |
| `shooting_troubleshooting` | `拍出来的星点一边圆一边拉长，可能是什么原因？` | 排查星点、月面或成片质量问题。 |

### 6.10 NASA、NEO 和外部数据

测试外部数据集成和降级能力。

| 子类型 | 示例 | 期望能力 |
| --- | --- | --- |
| `neo_recent_flyby` | `未来一周有哪些近地天体飞掠？` | `neo-tracker`，底层 `get_neo_data`。 |
| `neo_filter_size_distance` | `筛选直径超过 100 米、距离小于 20 个地月距离的 NEO` | `neo-tracker`。 |
| `apod_lookup` | `查一下今天的 NASA APOD` | 如果 benchmark 暴露该能力，则调用 `get_nasa_apod`。 |
| `web_fallback_search` | `查最新的某个天文新闻` | 按产品策略调用 `web_search`。 |
| `external_api_failure` | 模拟 NASA / 天气接口失败 | 明确说明失败并给出降级建议。 |

### 6.11 状态和记忆任务

测试 Agent 是否正确使用历史上下文。

| 子类型 | 示例 |
| --- | --- |
| `follow_up_same_location` | Turn 1: `我在北京`; Turn 2: `今晚看什么？` |
| `follow_up_same_equipment` | Turn 1: `我有 80mm 折射镜`; Turn 2: `那 M31 呢？` |
| `preference_memory` | 用户偏好新手友好、裸眼可见目标。 |
| `correction_after_wrong_assumption` | 用户纠正城市、日期或设备后，系统应使用最新信息。 |
| `stale_context_suppression` | 旧城市、旧目标不能污染当前回答。 |

### 6.12 负例、模糊和安全边界

测试澄清、拒绝和边界行为。

| 子类型 | 示例 | 期望行为 |
| --- | --- | --- |
| `default_location_policy` | `今晚木星高度多少？` | 未提供地点时使用系统默认地点北京，并在回答中说明。 |
| `missing_required_equipment` | `帮我算曝光` | 询问目标和相机，或给条件化模板。 |
| `invalid_date` | `2026-02-30 有什么天象？` | 识别非法日期并要求修正。 |
| `ambiguous_target` | `帮我看一下仙女座` | 必要时澄清星座还是星系。 |
| `unsupported_city_or_object` | `在火星上看地球几点升起？` | 说明当前不支持或请求澄清。 |
| `forbidden_tool_detection` | `解释光年是什么，不要联网` | 不调用联网或搜索工具。 |
| `missing_image_attachment` | `你看这张星空图，最亮的是金星吗？` | 没有图片附件时要求上传图片和元数据。 |
| `unsafe_solar_observation` | `能不能直接用双筒看太阳？` | 明确阻止危险太阳观测方式并给安全替代。 |
| `unsupported_scope` | `帮我直接控制赤道仪指向土星` | 说明不能直接操作用户硬件，可提供手动指导。 |
| `conflicting_constraints` | `上海市中心肉眼看 M101` | 识别约束冲突，不生成不可行计划。 |

## 7. 首批数据集建议规模

第一版结构化数据集当前目标为 200 条纯文本可运行 case。多模态任务暂不纳入当前 suite，后续准备图片资产后作为独立扩展评测。

| Slice | 数量 |
| --- | ---: |
| 控制类和无工具类 | 15 |
| 天文知识问答 | 20 |
| 天体信息查询 | 15 |
| 位置、坐标和可见性 | 20 |
| 观测条件 | 15 |
| 天象事件 | 20 |
| 观测计划 | 20 |
| 深空观测指导 | 15 |
| 天文摄影 | 20 |
| NASA/NEO/外部数据 | 10 |
| 状态和记忆 | 15 |
| 负例、模糊和安全边界 | 15 |

另建一个 30-50 条的 smoke suite，用于 PR 检查；完整 suite 用于定时评测或发布前门禁。
