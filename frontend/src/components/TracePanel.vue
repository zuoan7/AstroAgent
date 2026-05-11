<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">Timeline</p>
        <h2>执行时间线</h2>
      </div>
      <div class="badge-row">
        <span v-if="overview.route" class="status-pill subtle">{{ overview.route }}</span>
        <span class="status-pill subtle">事件 {{ overview.eventCount || 0 }}</span>
        <span class="status-pill subtle">成功 {{ overview.toolSuccessCount || 0 }}</span>
        <span class="status-pill subtle">失败 {{ overview.toolErrorCount || 0 }}</span>
      </div>
    </div>
    <p v-if="overview.requestId" class="trace-summary mono">Request ID: {{ overview.requestId }}</p>
    <p v-if="overview.routeReason" class="trace-summary">路由原因：{{ overview.routeReason }}</p>
    <p v-if="reasoningSummary" class="trace-summary">{{ reasoningSummary }}</p>
    <div v-if="errors.length" class="trace-errors">
      <article v-for="(error, index) in errors" :key="`${error.timestamp}-${index}`" class="trace-error">
        <strong>{{ error.type }}</strong>
        <p>{{ error.content }}</p>
      </article>
    </div>
    <div v-if="Object.keys(latencyMetrics || {}).length" class="trace-metrics">
      <span
        v-for="(value, key) in latencyMetrics"
        :key="key"
        class="metric-chip"
      >
        {{ key }}: {{ formatMs(value) }}
      </span>
    </div>
    <div class="trace-filter-row">
      <button
        v-for="type in filterTypes"
        :key="type"
        type="button"
        class="tiny-btn"
        :class="{ active: activeType === type }"
        @click="activeType = type"
      >
        {{ typeLabel(type) }} {{ type === 'all' ? events.length : eventTypeCounts[type] || 0 }}
      </button>
    </div>
    <div v-if="finalReport" class="trace-report">
      <h3>最终调用报告</h3>
      <div class="compact-list">
        <div class="compact-item">
          <strong>置信度</strong>
          <p>{{ finalReport.confidence ?? '--' }}</p>
        </div>
        <div class="compact-item">
          <strong>工具</strong>
          <p>{{ finalReport.toolsUsed?.length || 0 }} 次</p>
        </div>
        <div class="compact-item">
          <strong>来源</strong>
          <p>{{ finalReport.sources?.length || 0 }} 条</p>
        </div>
        <div class="compact-item">
          <strong>版本</strong>
          <pre class="mono trace-block">{{ serialize(finalReport.versions || {}) }}</pre>
        </div>
      </div>
    </div>
    <h3 class="trace-section-title">工具聚合</h3>
    <div class="trace-list tool-trace-list">
      <article v-for="item in items" :key="item.id" class="trace-item">
        <div class="trace-top">
          <strong>{{ item.tool || 'unknown_tool' }}</strong>
          <span class="status-pill" :class="item.status">{{ statusLabel(item.status) }}</span>
        </div>
        <p class="trace-label">输入参数</p>
        <pre class="mono trace-block">{{ item.input || '无输入参数' }}</pre>
        <p class="trace-label">返回摘要</p>
        <p>{{ item.outputSummary || '等待工具返回...' }}</p>
        <small>耗时：{{ formatDuration(item.durationSec) }}</small>
      </article>
      <p v-if="!items.length" class="muted">本轮尚未记录工具调用。</p>
    </div>
    <details class="trace-event-dropdown compact-preview" open>
      <summary>
        <span>最近事件摘要</span>
        <span class="status-pill subtle">{{ recentEvents.length }} 条</span>
      </summary>
      <div class="trace-list trace-event-list compact">
        <article v-for="event in recentEvents" :key="`recent-${event.id}`" class="trace-item event-item" :data-event-type="event.type">
          <div class="trace-top">
            <strong>#{{ event.sequence }} {{ event.type }}</strong>
            <span class="status-pill subtle">{{ formatTime(event.timestamp) }}</span>
          </div>
          <p v-if="event.label" class="trace-event-label">{{ event.label }}</p>
          <div class="badge-row event-meta-row">
            <span v-if="event.requestId" class="status-pill subtle">request {{ event.requestId }}</span>
            <span v-if="event.runId" class="status-pill subtle">run {{ event.runId }}</span>
            <span v-if="event.status" class="status-pill" :class="event.status">{{ statusLabel(event.status) }}</span>
          </div>
          <details>
            <summary>查看 payload</summary>
            <pre class="mono trace-block">{{ serialize(event.payload) }}</pre>
          </details>
        </article>
        <p v-if="!recentEvents.length" class="muted">当前过滤条件下没有事件。</p>
      </div>
    </details>
    <details class="trace-event-dropdown">
      <summary>
        <span>完整原始事件流</span>
        <span class="status-pill subtle">{{ filteredEvents.length }} 条</span>
      </summary>
      <p class="muted">展开后按当前过滤条件分页查看完整 SSE 事件，避免一次性撑开页面。</p>
      <div class="trace-list trace-event-list">
        <article v-for="event in visibleEvents" :key="event.id" class="trace-item event-item" :data-event-type="event.type">
          <div class="trace-top">
            <strong>#{{ event.sequence }} {{ event.type }}</strong>
            <span class="status-pill subtle">{{ formatTime(event.timestamp) }}</span>
          </div>
          <p v-if="event.label" class="trace-event-label">{{ event.label }}</p>
          <div class="badge-row event-meta-row">
            <span v-if="event.requestId" class="status-pill subtle">request {{ event.requestId }}</span>
            <span v-if="event.runId" class="status-pill subtle">run {{ event.runId }}</span>
            <span v-if="event.status" class="status-pill" :class="event.status">{{ statusLabel(event.status) }}</span>
          </div>
          <details>
            <summary>查看 payload</summary>
            <pre class="mono trace-block">{{ serialize(event.payload) }}</pre>
          </details>
        </article>
        <p v-if="!filteredEvents.length" class="muted">当前过滤条件下没有事件。</p>
      </div>
      <button
        v-if="filteredEvents.length > visibleLimit"
        type="button"
        class="ghost-btn trace-load-btn"
        @click="visibleLimit += 50"
      >
        再显示 50 条
      </button>
    </details>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  items: { type: Array, default: () => [] },
  events: { type: Array, default: () => [] },
  eventTypeCounts: { type: Object, default: () => ({}) },
  errors: { type: Array, default: () => [] },
  finalReport: { type: Object, default: null },
  reasoningSummary: { type: String, default: '' },
  overview: { type: Object, default: () => ({}) },
  latencyMetrics: { type: Object, default: () => ({}) },
})

const activeType = ref('all')
const primaryTypes = [
  'route_decision',
  'plan_update',
  'step_start',
  'step_end',
  'tool_start',
  'tool_end',
  'evidence_found',
  'text',
  'final_answer',
  'error',
  'warning',
]
const filterTypes = computed(() => [
  'all',
  ...primaryTypes.filter((type) => props.eventTypeCounts[type]),
])
const filteredEvents = computed(() =>
  activeType.value === 'all'
    ? props.events
    : props.events.filter((event) => event.type === activeType.value)
)
const visibleLimit = ref(50)
const visibleEvents = computed(() => filteredEvents.value.slice(0, visibleLimit.value))
const recentEvents = computed(() => filteredEvents.value.slice(-8).reverse())

function formatDuration(value) {
  if (value == null) {
    return '--'
  }
  return `${Number(value).toFixed(2)}s`
}

function formatMs(value) {
  if (value == null) {
    return '--'
  }
  return `${Number(value).toFixed(1)}ms`
}

function formatTime(value) {
  if (!value) {
    return '--'
  }
  return new Date(value).toLocaleTimeString('zh-CN', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function serialize(value) {
  try {
    return JSON.stringify(value, null, 2)
  } catch (_) {
    return String(value)
  }
}

function typeLabel(type) {
  const labels = {
    all: '全部',
    route_decision: '路由',
    plan_update: '计划',
    step_start: '步骤开始',
    step_end: '步骤结束',
    tool_start: '工具开始',
    tool_end: '工具结束',
    evidence_found: '证据',
    text: '文本',
    final_answer: '最终',
    error: '错误',
    warning: '警告',
  }
  return labels[type] || type
}

function statusLabel(status) {
  const labels = {
    pending: '等待中',
    running: '进行中',
    done: '已完成',
    success: '成功',
    error: '失败',
  }
  return labels[status] || status || '--'
}
</script>
