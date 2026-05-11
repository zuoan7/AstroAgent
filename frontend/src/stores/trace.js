import { computed, ref } from 'vue'
import { acceptHMRUpdate, defineStore } from 'pinia'

export const useTraceStore = defineStore('trace', () => {
  const items = ref([])
  const events = ref([])
  const textDeltas = ref([])
  const errors = ref([])
  const finalReport = ref(null)
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
    events.value = []
    textDeltas.value = []
    errors.value = []
    finalReport.value = null
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

  function normalizeEvent(event = {}) {
    const eventRequestId = event.request_id || event.meta?.request_id || requestId.value || ''
    const runId = event.run_id || event.meta?.run_id || ''
    if (eventRequestId && !requestId.value) {
      requestId.value = eventRequestId
    }
    return {
      id: event.event_id || `${event.sequence ?? events.value.length}-${event.type}-${Date.now()}`,
      type: event.type || 'unknown',
      sequence: event.sequence ?? events.value.length + 1,
      timestamp: event.timestamp ? event.timestamp * 1000 : Date.now(),
      requestId: eventRequestId,
      runId,
      status: event.status || event.meta?.status || '',
      label: summarizeEvent(event),
      payload: event,
    }
  }

  function recordEvent(event) {
    if (!event || !event.type) {
      return
    }
    const normalized = normalizeEvent(event)
    events.value.push(normalized)
    if (event.type === 'text' || event.type === 'thinking') {
      recordTextDelta(event)
    }
    if (event.type === 'error' || event.type === 'warning') {
      recordError(event)
    }
  }

  function summarizeEvent(event = {}) {
    if (event.type === 'text' || event.type === 'thinking') {
      return previewText(event.content, 120)
    }
    if (event.type === 'route_decision') {
      return `${event.route || 'unknown_route'}: ${event.route_reason || ''}`.trim()
    }
    if (event.type === 'tool_start') {
      return `${normalizeToolName(event)} started`
    }
    if (event.type === 'tool_end') {
      return `${normalizeToolName(event)} ${event.status || 'success'}`
    }
    if (event.type === 'step_start' || event.type === 'step_end') {
      return `${event.step_id || ''} ${event.status || ''}`.trim()
    }
    if (event.type === 'final_answer') {
      return previewText(event.final_answer, 120)
    }
    if (event.type === 'error' || event.type === 'warning') {
      return previewText(event.content, 120)
    }
    return previewText(event.title || event.summary || event.content || event.type, 120)
  }

  function previewText(value, maxLength = 160) {
    const text = serializeValue(value, '').replace(/\s+/g, ' ').trim()
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text
  }

  function recordTextDelta(payload) {
    const content = payload.content || ''
    if (!content) {
      return
    }
    textDeltas.value.push({
      sequence: payload.sequence ?? null,
      runId: payload.run_id || payload.meta?.run_id || '',
      type: payload.type,
      content,
      timestamp: payload.timestamp ? payload.timestamp * 1000 : Date.now(),
    })
  }

  function recordError(payload) {
    errors.value.push({
      type: payload.type,
      content: payload.content || '',
      meta: payload.meta || {},
      timestamp: payload.timestamp ? payload.timestamp * 1000 : Date.now(),
    })
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
    finalReport.value = {
      finalAnswer: payload.final_answer || '',
      sources: payload.sources || [],
      toolsUsed: payload.tools_used || [],
      confidence: payload.confidence ?? null,
      fallbackPath: payload.fallback_path || [],
      routeDecision: payload.route_decision || null,
      budgetUsage: payload.budget_usage || null,
      versions: payload.versions || null,
    }
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

  const eventTypeCounts = computed(() =>
    events.value.reduce((acc, event) => {
      acc[event.type] = (acc[event.type] || 0) + 1
      return acc
    }, {})
  )

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
    eventCount: events.value.length,
    textDeltaCount: textDeltas.value.length,
    errorCount: errors.value.length,
  }))

  return {
    items,
    events,
    textDeltas,
    errors,
    finalReport,
    reasoningSummary,
    route,
    routeReason,
    latencyMetrics,
    totalDurationSec,
    overview,
    eventTypeCounts,
    reset,
    startRun,
    recordEvent,
    startTool,
    endTool,
    finishRun,
    setRoute,
    setLatency,
  }
})

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useTraceStore, import.meta.hot))
}
