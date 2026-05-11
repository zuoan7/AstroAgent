# Memory Context V2 Stress Evaluation Summary

Dataset: `memory_context_eval_v2`

## Overall Metrics
| Metric | Value |
|---|---:|
| scenario_count | 12 |
| primary_max_tokens | 1000 |
| memory_hit_rate | 100.00% |
| tool_evidence_reuse_rate | 100.00% |
| paraphrase_hit_rate | 100.00% |
| stale_evidence_avoidance_rate | 0.00% |
| noise_robustness_score | 0.000 |
| wrong_tool_injection_rate | 100.00% |
| avg_expected_tool_rank | 1.2 |
| avg_context_token_saving | 23.51% |
| context_build_latency_avg_ms | 2.45 ms |
| context_build_latency_p95_ms | 3.57 ms |

## Budget Sweep Results
| max_tokens | Memory Hit | Tool Evidence Reuse | Irrelevant Injection | Wrong Tool Injection | Avg Expected Tool Rank |
|---:|---:|---:|---:|---:|---:|
| 300 | 100.00% | 100.00% | 38.19% | 100.00% | 1.20 |
| 600 | 100.00% | 100.00% | 38.19% | 100.00% | 1.20 |
| 1000 | 100.00% | 100.00% | 38.19% | 100.00% | 1.20 |
| 2000 | 100.00% | 100.00% | 38.19% | 100.00% | 1.20 |

## Paraphrase Robustness
| Scenario | Passed | Total | Rate |
|---|---:|---:|---:|
| vague_weather_beijing_001 | 4 | 4 | 100.00% |
| long_noise_m42_advice_002 | 3 | 3 | 100.00% |
| multi_tool_beijing_result_003 | 3 | 3 | 100.00% |
| stale_weather_update_004 | 3 | 3 | 100.00% |
| paraphrase_message_only_009 | 4 | 4 | 100.00% |
| m42_m31_photo_conflict_010 | 3 | 3 | 100.00% |
| event_time_ambiguous_012 | 3 | 3 | 100.00% |

## Noise Robustness
| Scenario | Memory Hit | Irrelevant Injection | Wrong Tool Injected |
|---|---:|---:|---:|
| long_noise_m42_advice_002 | yes | 25.00% | yes |
| long_noise_neo_priority_011 | yes | 0.00% | yes |

## Stale Evidence Avoidance
| Scenario | Avoided | Fresh Rank | Stale Rank | Stale Hits |
|---|---:|---:|---:|---|
| stale_weather_update_004 | no | 1 | 2 | 18:00 旧结果, 多云, 云量 75% |
| stale_exposure_update_005 | no | 1 | 3 | 旧参数, ISO 1600, 120 秒 |
| stale_position_update_006 | no | 1 | 2 | 旧地点, 上海, 18 度 |
| event_time_ambiguous_012 | no | 1 | 2 | 英仙座流星雨, 8 月 13 日, Perseids stale event |

## Failure Cases
| Scenario | Reasons | Wrong Hits | Irrelevant Hits |
|---|---|---|---|
| vague_weather_beijing_001 | wrong_tool_injected, irrelevant_memory_injected | 上海, 云量 72%, M42 | 近地小行星 |
| long_noise_m42_advice_002 | wrong_tool_injected, irrelevant_memory_injected | M31, 上海, 90 秒 | M31 |
| multi_tool_beijing_result_003 | wrong_tool_injected, irrelevant_memory_injected | 上海, 云量 68%, M31 | 上海只是备选, M31 |
| stale_weather_update_004 | wrong_tool_injected, stale_evidence_not_avoided, irrelevant_memory_injected | 18:00 旧结果, 多云, 不适合 | M31 摄影参数 |
| stale_exposure_update_005 | wrong_tool_injected, stale_evidence_not_avoided | 旧参数, ISO 1600, 120 秒, M31 | none |
| stale_position_update_006 | wrong_tool_injected, stale_evidence_not_avoided, irrelevant_memory_injected | 旧地点, 上海, 18 度 | 流星雨 |
| tiny_budget_weather_007 | wrong_tool_injected, irrelevant_memory_injected | M31, 上海, 60 秒 | M31 上海 |
| tiny_budget_task_state_008 | irrelevant_memory_injected | none | 土星冲日 |
| paraphrase_message_only_009 | irrelevant_memory_injected | none | 近地小行星, 相机曝光参数 |
| m42_m31_photo_conflict_010 | wrong_tool_injected, irrelevant_memory_injected | M31, 上海, 90 秒 | M31 上海 |
| long_noise_neo_priority_011 | wrong_tool_injected | weather not NEO, 北京, 晴 | none |
| event_time_ambiguous_012 | wrong_tool_injected, stale_evidence_not_avoided | 英仙座流星雨, 8 月 13 日, Perseids stale event | none |

## Recommendations
- RetrievalPlanner 需要更强的实体约束或负向过滤，避免问北京时注入上海、问 M42 时注入 M31。
- 对同一工具/实体的新旧结果应建立 freshness 规则，优先最新证据并压低旧结果。
- 长历史噪声下应限制 recent messages 的无差别注入，优先使用与当前焦点匹配的片段。
- 本次共有 12 个 primary failure case，建议优先查看 metrics.json 中对应 context_text 和 selected_tool_calls。

## Notes
- 本评估为 synthetic stress eval，不调用真实 LLM/MCP。
- 指标只评估 MemoryService.build_context 的上下文构造，不评估最终回答质量。
- stale_evidence_avoidance 使用严格口径：必须命中新证据、新证据 rank 优于旧证据，并且上下文/selected_tool_calls 中不出现旧证据关键词。
- noise_robustness_score 使用 primary budget 下的 memory_hit、irrelevant injection 和 wrong tool injection 合成分数。
