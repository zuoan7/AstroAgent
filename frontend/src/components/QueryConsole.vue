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
      <button class="primary-btn" :disabled="disabled" @click="$emit('submit')">执行任务</button>
      <button class="danger-btn" @click="$emit('clear-session')">清空当前会话</button>
    </div>
    <p v-if="error" class="error-text">{{ error }}</p>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'

const imageInputRef = ref(null)
const audioInputRef = ref(null)
const menuOpen = ref(false)

const props = defineProps({
  modelValue: { type: String, required: true },
  disabled: { type: Boolean, required: true },
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
  'refresh',
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
