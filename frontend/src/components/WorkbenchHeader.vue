<template>
  <header class="hero-card">
    <div>
      <p class="eyebrow">AstroAgent Workbench</p>
      <h1>可观测、可解释、可操作的 Agent 控制台</h1>
      <p class="muted">
        实时展示阶段进度、工具时间线、证据引用、记忆命中与整轮耗时。
      </p>
    </div>
    <div class="hero-meta">
      <div class="hero-summary">
        <span class="status-pill subtle">账号 {{ accountName }}</span>
        <span class="status-pill subtle">User {{ userId }}</span>
        <span class="status-pill subtle">Session {{ sessionId }}</span>
      </div>
      <label class="toggle">
        <input type="checkbox" :checked="disableLongTermMemory" @change="$emit('toggle-memory', $event.target.checked)" />
        <span>本轮禁用长期记忆</span>
      </label>
      <div class="badge-row">
        <span class="status-pill" :class="{ active: isStreaming }">{{ isStreaming ? 'Streaming' : 'Idle' }}</span>
        <span class="status-pill subtle">Vue 3</span>
        <span class="status-pill subtle">Pinia</span>
      </div>
    </div>
  </header>
</template>

<script setup>
defineProps({
  accountName: { type: String, default: 'Demo Account' },
  userId: { type: String, required: true },
  sessionId: { type: String, required: true },
  disableLongTermMemory: { type: Boolean, required: true },
  isStreaming: { type: Boolean, required: true },
})

defineEmits(['toggle-memory'])
</script>
