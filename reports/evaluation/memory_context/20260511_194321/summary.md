# Memory Context Evaluation Summary

Dataset: `memory_context_eval_v1`

## Overall Metrics
| Metric | Value |
|---|---:|
| scenario_count | 10 |
| memory_hit_rate | 100.00% |
| avg_context_token_saving | 23.06% |
| avg_tool_summary_compression_rate | 45.60% |
| avg_irrelevant_memory_injection_rate | 10.00% |
| context_build_latency_avg_ms | 2.58 ms |
| context_build_latency_p50_ms | 1.86 ms |
| context_build_latency_p95_ms | 4.80 ms |
| context_build_latency_max_ms | 4.80 ms |
| tool_evidence_reuse_rate | 100.00% |

## Per Scenario Results
| Scenario | Memory Hit | Token Saving | Compression | Irrelevant Injection | Tool Evidence Reused | Latency |
|---|---:|---:|---:|---:|---:|---:|
| obs_weather_followup_001 | yes | 45.64% | 76.26% | 0.00% | yes | 4.80 ms |
| astrophoto_exposure_followup_002 | yes | 36.67% | 56.63% | 0.00% | yes | 2.67 ms |
| deep_sky_equipment_followup_003 | yes | 26.64% | 36.92% | 0.00% | yes | 1.86 ms |
| meteor_shower_time_followup_004 | yes | 24.56% | 40.90% | 0.00% | yes | 1.81 ms |
| neo_priority_followup_005 | yes | 21.05% | 47.45% | 0.00% | yes | 2.91 ms |
| object_altitude_followup_006 | yes | 27.96% | 43.52% | 0.00% | yes | 1.77 ms |
| irrelevant_history_filter_007 | yes | -4.04% | -22.28% | 100.00% | yes | 3.29 ms |
| long_tool_raw_compression_008 | yes | 63.42% | 85.43% | 0.00% | yes | 1.84 ms |
| task_state_context_009 | yes | -1.68% | n/a | 0.00% | n/a | 1.76 ms |
| message_only_followup_010 | yes | -9.57% | n/a | 0.00% | n/a | 3.14 ms |

## Findings
- 表现较好的 scenario: obs_weather_followup_001, astrophoto_exposure_followup_002, deep_sky_equipment_followup_003, meteor_shower_time_followup_004, neo_priority_followup_005, object_altitude_followup_006, long_tool_raw_compression_008, task_state_context_009, message_only_followup_010
- 未达到 memory_hit 阈值的 scenario: 无
- 无关记忆注入: irrelevant_history_filter_007 injected 土星冲日, 近地小行星, 相机曝光参数
- 平均工具摘要压缩率为 45.60%，仅统计包含工具调用的 scenario。
- 低/负压缩率 scenario: irrelevant_history_filter_007 (-22.28%)
- context 构建延迟 avg/p95/max = 2.58/4.80/4.80 ms。

## Notes
- 本评估为 synthetic eval，不调用真实 LLM/MCP。
- 指标只评估 memory context construction，不评估最终回答质量。
- 当前 MemoryService 支持通过 DTO 直接写入 message/tool_call；task_state 通过 update_task_state patch 写入。
- naive_full_context 由原始 message、完整 tool raw_output 和 task_state_patch 拼接得到，并复用 MemoryService token estimator。
