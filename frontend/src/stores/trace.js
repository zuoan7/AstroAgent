import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export const useTraceStore = defineStore('trace', () => {
  const items = ref([])
  const reasoningSummary = ref('')
  const runStartedAt = ref(null)
  const runFinishedAt = ref(null)
  const requestId = ref('')
  const route = ref('')
  const routeReason = ref('')
  const latencyMetrics = ref({})
  const finalMetrics = ref({
    totalDurationSec: null,
    toolCount: 0,
    toolSuccessCount: 0,
    toolErrorCount: 0,
    evidenceCount: 0,
    memoryHitCount: 0,
    confidence: null,
  })

  function normalizeToolName(payload = {}) {
    return (
      payload.tool ||
      payload.tool_name ||
      payload.name ||
      payload.meta?.tool ||
      payload.meta?.tool_name ||
      'unknown_tool'
    )
  }

  function serializeValue(value, fallback = '') {
    if (value === undefined || value === null || value === '') {
      return fallback
    }
    if (typeof value === 'string') {
      return value
    }
    try {
      return JSON.stringify(value, null, 2)
    } catch (_) {
      return String(value)
    }
  }

  function reset() {
    items.value = []
    reasoningSummary.value = ''
    runStartedAt.value = null
    runFinishedAt.value = null
    requestId.value = ''
    route.value = ''
    routeReason.value = ''
    latencyMetrics.value = {}
    finalMetrics.value = {
      totalDurationSec: null,
      toolCount: 0,
      toolSuccessCount: 0,
      toolErrorCount: 0,
      evidenceCount: 0,
      memoryHitCount: 0,
      confidence: null,
    }
  }

  function startRun() {
    runStartedAt.value = Date.now()
    runFinishedAt.value = null
  }

  function startTool(payload) {
    const runId = payload.meta?.run_id || payload.run_id || `${Date.now()}`
    const toolName = normalizeToolName(payload)
    const existing = items.value.find((item) => item.id === runId)
    if (existing) {
      existing.tool = toolName
      existing.input = serializeValue(payload.input, '无输入参数')
      existing.status = 'running'
      return
    }
    items.value.unshift({
      id: runId,
      tool: toolName,
      input: serializeValue(payload.input, '无输入参数'),
      outputSummary: '',
      durationSec: null,
      status: 'running',
      timestamp: payload.timestamp || Date.now() / 1000,
    })
  }

  function endTool(payload) {
    const runId = payload.meta?.run_id || payload.run_id
    const toolName = normalizeToolName(payload)
    const current = items.value.find((item) => item.id === runId)
    if (current) {
      current.tool = toolName
      current.outputSummary = serializeValue(
        payload.output_summary || payload.output,
        '等待工具返回...'
      )
      current.durationSec = payload.meta?.duration_sec ?? null
      current.status = payload.status || 'success'
    } else {
      items.value.unshift({
        id: runId || `${Date.now()}`,
        tool: toolName,
        input: '无输入参数',
        outputSummary: serializeValue(payload.output_summary || payload.output, '等待工具返回...'),
        durationSec: payload.meta?.duration_sec ?? null,
        status: payload.status || 'success',
        timestamp: payload.timestamp || Date.now() / 1000,
      })
    }
  }

  function finishRun(payload) {
    runFinishedAt.value = Date.now()
    requestId.value = payload.request_id || payload.meta?.request_id || requestId.value
    if (payload.latency_metrics?.stages_ms) {
      latencyMetrics.value = payload.latency_metrics.stages_ms
    }
    finalMetrics.value = {
      totalDurationSec: payload.total_duration_sec ?? totalDurationSec.value,
      toolCount: payload.tool_count ?? items.value.length,
      toolSuccessCount:
        payload.tool_success_count ??
        items.value.filter((item) => item.status === 'success').length,
      toolErrorCount:
        payload.tool_error_count ??
        items.value.filter((item) => item.status === 'error').length,
      evidenceCount: payload.evidence_count ?? 0,
      memoryHitCount: payload.memory_hit_count ?? 0,
      confidence: payload.confidence ?? null,
    }
  }

  function setRoute(payload) {
    route.value = payload.route || ''
    routeReason.value = payload.route_reason || ''
  }

  function setLatency(payload) {
    latencyMetrics.value = payload.stages_ms || {}
  }

  const totalDurationSec = computed(() => {
    if (finalMetrics.value.totalDurationSec != null) {
      return finalMetrics.value.totalDurationSec
    }
    if (!runStartedAt.value) {
      return null
    }
    const end = runFinishedAt.value || Date.now()
    return Math.max(0, (end - runStartedAt.value) / 1000)
  })

  const overview = computed(() => ({
    requestId: requestId.value,
    route: route.value,
    routeReason: routeReason.value,
    totalDurationSec: totalDurationSec.value,
    toolCount: finalMetrics.value.toolCount || items.value.length,
    toolSuccessCount:
      finalMetrics.value.toolSuccessCount ||
      items.value.filter((item) => item.status === 'success').length,
    toolErrorCount:
      finalMetrics.value.toolErrorCount ||
      items.value.filter((item) => item.status === 'error').length,
    evidenceCount: finalMetrics.value.evidenceCount,
    memoryHitCount: finalMetrics.value.memoryHitCount,
    confidence: finalMetrics.value.confidence,
  }))

  return {
    items,
    reasoningSummary,
    route,
    routeReason,
    latencyMetrics,
    totalDurationSec,
    overview,
    reset,
    startRun,
    startTool,
    endTool,
    finishRun,
    setRoute,
    setLatency,
  }
})
