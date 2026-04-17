<template>
  <section class="panel composer-panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">Mission Console</p>
        <h2>任务输入</h2>
      </div>
      <button class="ghost-btn" @click="$emit('refresh')">刷新记忆面板</button>
    </div>
    <textarea
      :value="modelValue"
      class="query-input"
      placeholder="例如：基于我过去的观测偏好，给我一个今晚的行星观测计划，并说明你调用了哪些工具。"
      @input="$emit('update:modelValue', $event.target.value)"
    />
    <div class="composer-actions">
      <button class="primary-btn" :disabled="disabled" @click="$emit('submit')">执行任务</button>
      <button class="danger-btn" @click="$emit('clear-session')">清空当前用户记忆</button>
    </div>
    <p v-if="error" class="error-text">{{ error }}</p>
  </section>
</template>

<script setup>
defineProps({
  modelValue: { type: String, required: true },
  disabled: { type: Boolean, required: true },
  error: { type: String, default: '' },
})

defineEmits(['update:modelValue', 'submit', 'refresh', 'clear-session'])
</script>
