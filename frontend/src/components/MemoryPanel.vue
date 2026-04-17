<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">Memory</p>
        <h2>用户画像与记忆</h2>
      </div>
      <span class="status-pill subtle">{{ loading ? 'Loading' : 'Ready' }}</span>
    </div>
    <div class="memory-grid">
      <article class="subpanel">
        <h3>画像卡片</h3>
        <p class="mono">{{ profile?.user_id || 'anonymous' }}</p>
        <p>偏好：{{ formatObject(profile?.preferences) }}</p>
        <p>习惯：{{ formatObject(profile?.habits) }}</p>
        <p>约束：{{ (profile?.constraints || []).join('，') || '无' }}</p>
      </article>
      <article class="subpanel">
        <h3>命中解释</h3>
        <div v-if="memoryHits.length" class="compact-list">
          <div v-for="hit in memoryHits" :key="hit.memory_id" class="compact-item">
            <strong>{{ hit.memory_type }}.{{ hit.key }}</strong>
            <p>{{ renderValue(hit.value) }}</p>
            <small>{{ hit.reason }}</small>
          </div>
        </div>
        <p v-else class="muted">本轮尚未使用长期记忆。</p>
      </article>
      <article class="subpanel">
        <h3>短期记忆上下文</h3>
        <p>{{ shortTerm?.summary || '暂无摘要' }}</p>
        <small>会话消息数：{{ shortTerm?.debug?.message_count || 0 }}</small>
      </article>
      <article class="subpanel">
        <h3>统计</h3>
        <p>{{ formatObject(stats) }}</p>
      </article>
    </div>

    <div class="memory-columns">
      <div class="subpanel">
        <h3>正式记忆</h3>
        <div class="memory-list">
          <article v-for="item in memories" :key="item.id" class="memory-item">
            <div class="trace-top">
              <strong>{{ item.memory_type }}.{{ item.key }}</strong>
              <span class="status-pill subtle">{{ item.status }}</span>
            </div>
            <p>{{ renderValue(item.value) }}</p>
            <small>{{ item.source_content_snippet || item.metadata?.raw_content || '无来源片段' }}</small>
            <small>记录原因：{{ item.metadata?.reason || item.source_type }}</small>
            <div class="row-actions">
              <button class="tiny-btn" @click="$emit('confirm-memory', item.id)">确认</button>
              <button class="tiny-btn" @click="$emit('archive-memory', item.id)">停用</button>
              <button class="tiny-btn danger" @click="$emit('delete-memory', item.id)">删除</button>
            </div>
          </article>
        </div>
      </div>
      <div class="subpanel">
        <h3>候选记忆 / 待确认</h3>
        <div class="memory-list">
          <article v-for="item in candidates" :key="item.id" class="memory-item">
            <strong>{{ item.memory_type }}.{{ item.key }}</strong>
            <p>{{ renderValue(item.value) }}</p>
            <div class="row-actions">
              <button class="tiny-btn" @click="$emit('accept-candidate', item.id)">提升</button>
              <button class="tiny-btn danger" @click="$emit('reject-candidate', item.id)">拒绝</button>
            </div>
          </article>
          <article v-for="item in confirmations" :key="item.id" class="memory-item">
            <strong>确认请求</strong>
            <p>{{ item.content }}</p>
            <div class="row-actions">
              <button class="tiny-btn" @click="$emit('resolve-confirmation', item.id, 'confirmed')">确认</button>
              <button class="tiny-btn danger" @click="$emit('resolve-confirmation', item.id, 'rejected')">拒绝</button>
            </div>
          </article>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  profile: { type: Object, default: null },
  stats: { type: Object, default: () => ({}) },
  memories: { type: Array, default: () => [] },
  candidates: { type: Array, default: () => [] },
  confirmations: { type: Array, default: () => [] },
  memoryHits: { type: Array, default: () => [] },
  shortTerm: { type: Object, default: null },
  loading: { type: Boolean, default: false },
})

defineEmits([
  'confirm-memory',
  'archive-memory',
  'delete-memory',
  'accept-candidate',
  'reject-candidate',
  'resolve-confirmation',
])

function renderValue(value) {
  return typeof value === 'string' ? value : JSON.stringify(value)
}

function formatObject(value) {
  if (!value || (Array.isArray(value) && value.length === 0)) {
    return '无'
  }
  return typeof value === 'string' ? value : JSON.stringify(value)
}
</script>
