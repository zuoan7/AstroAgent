<template>
  <section class="customer-shell">
    <aside class="panel customer-sidebar">
      <div class="panel-head">
        <div>
          <p class="eyebrow">Accounts</p>
          <h2>客户账号</h2>
        </div>
      </div>

      <div class="inline-form">
        <select :value="activeAccountId" @change="$emit('select-account', $event.target.value)">
          <option v-for="account in accounts" :key="account.id" :value="account.id">
            {{ account.name }}
          </option>
        </select>
        <button class="ghost-btn" @click="createAccount">新增账号</button>
      </div>

      <div class="panel-head customer-subhead">
        <div>
          <p class="eyebrow">Sessions</p>
          <h2>会话列表</h2>
        </div>
        <button class="ghost-btn" @click="createConversation">新建会话</button>
      </div>

      <div class="conversation-list">
        <div
          v-for="conversation in conversations"
          :key="conversation.id"
          class="conversation-item"
          :class="{ active: conversation.id === activeConversationId }"
        >
          <button
            class="conversation-main"
            type="button"
            @click="$emit('select-conversation', conversation.id)"
          >
            <strong>{{ conversation.title }}</strong>
            <small>{{ formatTime(conversation.updatedAt) }}</small>
          </button>
          <button
            class="tiny-btn danger conversation-delete"
            type="button"
            :aria-label="`删除会话 ${conversation.title}`"
            @click="$emit('remove-conversation', conversation.id)"
          >
            删除
          </button>
        </div>
      </div>

      <div class="subpanel">
        <h3>当前上下文</h3>
        <p>账号：{{ activeAccountName }}</p>
        <p class="mono">User ID: {{ userId }}</p>
        <p class="mono">Session ID: {{ sessionId }}</p>
        <button class="danger-btn full-width" @click="$emit('clear-session')">清空当前会话</button>
      </div>
    </aside>

    <section class="panel customer-main">
      <div class="panel-head">
        <div>
          <p class="eyebrow">Client Chat</p>
          <h2>{{ activeConversationTitle }}</h2>
        </div>
        <div class="badge-row">
          <span class="status-pill" :class="{ active: isStreaming }">
            {{ isStreaming ? 'Streaming' : 'Ready' }}
          </span>
        </div>
      </div>

      <div class="chat-feed customer-feed">
        <article
          v-for="message in messages"
          :key="message.id"
          class="message-card"
          :data-role="message.role"
        >
          <strong>{{ message.role === 'user' ? activeAccountName : 'AstroAgent' }}</strong>
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
          <strong>AstroAgent</strong>
          <p>{{ streamingAnswer }}</p>
        </article>
        <article v-if="!messages.length && !streamingAnswer" class="message-card empty-state" data-role="assistant">
          <strong>AstroAgent</strong>
          <p>选择账号和会话后即可开始对话。工作台版面保留了完整的调试追踪视图。</p>
        </article>
      </div>

      <div class="customer-composer">
        <textarea
          :value="modelValue"
          class="query-input"
          placeholder="例如：帮我安排今晚在上海的双筒望远镜观测目标。"
          @input="$emit('update:modelValue', $event.target.value)"
        />
        <div class="upload-row">
          <input
            ref="imageInputRef"
            type="file"
            accept="image/*"
            class="sr-only"
            @change="handleImageChange"
          />
          <input
            ref="audioInputRef"
            type="file"
            accept="audio/*"
            class="sr-only"
            @change="handleAudioChange"
          />
          <details class="attach-menu" :open="menuOpen && !disabled" @toggle="handleMenuToggle">
            <summary class="icon-btn ghost-btn" :class="{ disabled }" aria-label="上传附件">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M21.44 11.05 12.25 20.24a6 6 0 1 1-8.49-8.49l9.2-9.19a4 4 0 0 1 5.65 5.66l-9.2 9.19a2 2 0 0 1-2.82-2.83l8.48-8.48"
                  fill="none"
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="1.8"
                />
              </svg>
            </summary>
            <div class="attach-menu-panel">
              <button class="attach-option" type="button" @click="pickImageFromMenu">上传图片</button>
              <button class="attach-option" type="button" @click="pickAudioFromMenu">上传语音</button>
            </div>
          </details>
          <button
            class="icon-btn ghost-btn"
            type="button"
            :disabled="disabled || !canRecord"
            @click="$emit('toggle-recording')"
            :aria-label="isRecording ? '停止录音' : '开始录音'"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M12 15a3 3 0 0 0 3-3V7a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Z"
                fill="none"
                stroke="currentColor"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="1.8"
              />
              <path
                d="M19 11a7 7 0 0 1-14 0M12 18v3M8 21h8"
                fill="none"
                stroke="currentColor"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="1.8"
              />
            </svg>
          </button>
        </div>
        <p v-if="recordingLabel" class="recording-hint">{{ recordingLabel }}</p>
        <div v-if="imagePreviewUrl" class="pending-preview-card">
          <img class="message-image" :src="imagePreviewUrl" :alt="imageFileName || 'pending image'" />
          <div class="badge-row">
            <span class="status-pill subtle">本地预览</span>
            <span v-if="uploadState === 'uploading'" class="status-pill running">上传中</span>
          </div>
        </div>
        <div v-if="imageFileName || audioFileName" class="attachment-row">
          <div v-if="imageFileName" class="attachment-chip">
            <span>图片：{{ imageFileName }}</span>
            <button class="tiny-btn" type="button" @click="$emit('clear-image')">移除</button>
          </div>
          <div v-if="audioFileName" class="attachment-chip">
            <span>语音：{{ audioFileName }}</span>
            <button class="tiny-btn" type="button" @click="$emit('clear-audio')">移除</button>
          </div>
        </div>
        <div class="composer-actions">
          <button class="primary-btn" :disabled="disabled" @click="$emit('submit')">发送</button>
        </div>
        <p v-if="error" class="error-text">{{ error }}</p>
      </div>
    </section>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'

const imageInputRef = ref(null)
const audioInputRef = ref(null)
const menuOpen = ref(false)

function formatTime(value) {
  if (!value) {
    return '刚刚'
  }
  return new Date(value).toLocaleString('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const props = defineProps({
  accounts: { type: Array, default: () => [] },
  activeAccountId: { type: String, required: true },
  activeAccountName: { type: String, required: true },
  conversations: { type: Array, default: () => [] },
  activeConversationId: { type: String, required: true },
  activeConversationTitle: { type: String, default: '会话' },
  userId: { type: String, required: true },
  sessionId: { type: String, required: true },
  modelValue: { type: String, required: true },
  messages: { type: Array, default: () => [] },
  streamingAnswer: { type: String, default: '' },
  disabled: { type: Boolean, required: true },
  isStreaming: { type: Boolean, required: true },
  error: { type: String, default: '' },
  imageFileName: { type: String, default: '' },
  audioFileName: { type: String, default: '' },
  imagePreviewUrl: { type: String, default: '' },
  uploadState: { type: String, default: 'idle' },
  isRecording: { type: Boolean, required: true },
  canRecord: { type: Boolean, required: true },
  recordingDurationSec: { type: Number, default: 0 },
})

const emit = defineEmits([
  'update:modelValue',
  'submit',
  'select-account',
  'add-account',
  'select-conversation',
  'add-conversation',
  'remove-conversation',
  'clear-session',
  'select-image',
  'select-audio',
  'clear-image',
  'clear-audio',
  'toggle-recording',
])

const recordingLabel = computed(() => {
  if (!props.canRecord) {
    return '当前浏览器不支持录音。'
  }
  if (props.isRecording) {
    return `录音中 ${props.recordingDurationSec}s`
  }
  if (props.audioFileName) {
    return '可直接发送录音，或重新录制。'
  }
  return ''
})

function createAccount() {
  const name = window.prompt('请输入账号名称')
  if (name) {
    emit('add-account', name)
  }
}

function createConversation() {
  const title = window.prompt('请输入会话标题', `会话 ${props.conversations.length + 1}`)
  emit('add-conversation', title || '')
}

function openImagePicker() {
  imageInputRef.value?.click()
}

function openAudioPicker() {
  audioInputRef.value?.click()
}

function handleMenuToggle(event) {
  menuOpen.value = event.target.open
}

function pickImageFromMenu() {
  menuOpen.value = false
  openImagePicker()
}

function pickAudioFromMenu() {
  menuOpen.value = false
  openAudioPicker()
}

function handleImageChange(event) {
  const [file] = event.target.files || []
  if (file) {
    emit('select-image', file)
  }
  event.target.value = ''
}

function handleAudioChange(event) {
  const [file] = event.target.files || []
  if (file) {
    emit('select-audio', file)
  }
  event.target.value = ''
}
</script>
