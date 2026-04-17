<template>
  <nav class="surface-switcher panel">
    <div>
      <p class="eyebrow">Interface</p>
      <h2>版面切换</h2>
    </div>
    <div class="surface-actions">
      <button
        class="switch-btn"
        :class="{ active: mode === 'customer' }"
        @click="$emit('change', 'customer')"
      >
        客户界面
      </button>
      <button
        class="switch-btn"
        :class="{ active: mode === 'workbench' }"
        @click="$emit('change', 'workbench')"
      >
        调试工作台
      </button>
    </div>
  </nav>
</template>

<script setup>
defineProps({
  mode: { type: String, required: true },
})

defineEmits(['change'])
</script>
