<template>
  <section class="panel chat-panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">Answer</p>
        <h2>最终回答</h2>
      </div>
      <div class="badge-row">
        <span v-if="finalAnswer?.confidence" class="status-pill subtle">Confidence {{ finalAnswer.confidence }}</span>
        <span v-if="finalAnswer?.total_duration_sec != null" class="status-pill subtle">总耗时 {{ Number(finalAnswer.total_duration_sec).toFixed(2) }}s</span>
      </div>
    </div>
    <p v-if="finalAnswer" class="trace-summary">
      本轮共调用 {{ finalAnswer.tool_count || finalAnswer.tools_used?.length || 0 }} 个工具，
      命中 {{ finalAnswer.memory_hit_count || finalAnswer.memory_hits?.length || 0 }} 条记忆，
      生成 {{ finalAnswer.evidence_count || finalAnswer.sources?.length || 0 }} 条证据。
    </p>
    <div class="chat-feed">
      <article v-for="message in messages" :key="message.id" class="message-card" :data-role="message.role">
        <strong>{{ message.role === 'user' ? 'User' : 'Agent' }}</strong>
        <img
          v-if="message.imageUrl || message.localPreviewUrl"
          class="message-image"
          :src="message.imageUrl || message.localPreviewUrl"
          :alt="message.attachmentName || 'uploaded image'"
        />
        <p v-if="message.attachmentName" class="message-meta">
          {{ message.kind === 'audio' ? '语音文件' : '附件' }}：{{ message.attachmentName }}
        </p>
        <p v-if="message.transcription" class="message-meta">语音转写：{{ message.transcription }}</p>
        <p v-if="message.status === 'uploading'" class="message-meta">图片上传中...</p>
        <p v-else-if="message.status === 'failed'" class="message-meta error-text">图片上传失败</p>
        <p>{{ message.content }}</p>
      </article>
      <article v-if="streamingAnswer" class="message-card" data-role="assistant">
        <strong>Streaming</strong>
        <p>{{ streamingAnswer }}</p>
      </article>
    </div>
  </section>
</template>

<script setup>
defineProps({
  messages: { type: Array, default: () => [] },
  streamingAnswer: { type: String, default: '' },
  finalAnswer: { type: Object, default: null },
})
</script>
