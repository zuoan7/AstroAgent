<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">Timeline</p>
        <h2>执行时间线</h2>
      </div>
      <div class="badge-row">
        <span v-if="overview.route" class="status-pill subtle">{{ overview.route }}</span>
        <span class="status-pill subtle">成功 {{ overview.toolSuccessCount || 0 }}</span>
        <span class="status-pill subtle">失败 {{ overview.toolErrorCount || 0 }}</span>
      </div>
    </div>
    <p v-if="reasoningSummary" class="trace-summary">{{ reasoningSummary }}</p>
    <div v-if="Object.keys(latencyMetrics || {}).length" class="trace-metrics">
      <span
        v-for="(value, key) in latencyMetrics"
        :key="key"
        class="metric-chip"
      >
        {{ key }}: {{ formatMs(value) }}
      </span>
    </div>
    <div class="trace-list">
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
    </div>
  </section>
</template>

<script setup>
defineProps({
  items: { type: Array, default: () => [] },
  reasoningSummary: { type: String, default: '' },
  overview: { type: Object, default: () => ({}) },
  latencyMetrics: { type: Object, default: () => ({}) },
})

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
