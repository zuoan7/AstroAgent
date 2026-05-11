<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">Overview</p>
        <h2>执行总览</h2>
      </div>
    </div>
    <div class="overview-grid">
      <article class="overview-card">
        <small>总耗时</small>
        <strong>{{ formatDuration(overview.totalDurationSec) }}</strong>
      </article>
      <article class="overview-card">
        <small>工具调用</small>
        <strong>{{ overview.toolCount || 0 }}</strong>
      </article>
      <article class="overview-card">
        <small>证据条目</small>
        <strong>{{ overview.evidenceCount || 0 }}</strong>
      </article>
      <article class="overview-card">
        <small>记忆命中</small>
        <strong>{{ overview.memoryHitCount || 0 }}</strong>
      </article>
    </div>
    <div class="plan-list">
      <article v-for="step in steps" :key="step.id" class="plan-item" :data-status="step.status">
        <div class="plan-marker"></div>
        <div>
          <h3>{{ step.title }}</h3>
          <p>{{ step.description }}</p>
          <small class="muted">
            开始 {{ formatTime(step.startedAt) }} · 结束 {{ formatTime(step.finishedAt) }} · 耗时 {{ formatStepDuration(step) }}
          </small>
        </div>
        <span class="status-pill" :class="step.status">{{ statusLabel(step.status) }}</span>
      </article>
    </div>
  </section>
</template>

<script setup>
defineProps({
  steps: { type: Array, default: () => [] },
  overview: { type: Object, default: () => ({}) },
})

function formatDuration(value) {
  if (value == null) {
    return '--'
  }
  return `${Number(value).toFixed(2)}s`
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

function formatStepDuration(step) {
  if (!step.startedAt) {
    return '--'
  }
  const end = step.finishedAt || Date.now()
  return `${Math.max(0, (end - step.startedAt) / 1000).toFixed(2)}s`
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
