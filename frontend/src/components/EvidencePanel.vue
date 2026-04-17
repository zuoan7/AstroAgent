<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">Evidence</p>
        <h2>证据与引用</h2>
      </div>
    </div>
    <div class="evidence-list">
      <article v-for="item in items" :key="item.source_id" class="evidence-item">
        <div class="trace-top">
          <strong>{{ item.title }}</strong>
          <span class="status-pill subtle">{{ item.kind }}</span>
        </div>
        <p>{{ item.snippet }}</p>
        <small v-if="item.reason">{{ item.reason }}</small>
      </article>
    </div>
  </section>
</template>

<script setup>
defineProps({
  items: { type: Array, default: () => [] },
})
</script>
